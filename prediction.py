import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


FEATURES = [
    "previous_monthly_spend",
    "previous_average_transaction",
    "previous_transaction_count",
    "previous_late_night_ratio",
    "previous_weekend_ratio",
    "previous_average_time_between_transactions",
    "previous_spending_income_ratio",
    "spending_growth",
]


def prepare_monthly_data(df: pd.DataFrame) -> pd.DataFrame:
    """Build chronological monthly features used by the risk model."""
    if df is None or df.empty:
        return pd.DataFrame()

    data = df.copy()

    if "date_time" not in data.columns:
        return pd.DataFrame()

    data["date_time"] = pd.to_datetime(
        data["date_time"],
        errors="coerce",
    )
    data = data.dropna(
        subset=["date_time", "amount"]
    )

    if data.empty:
        return pd.DataFrame()

    if "is_late_night" not in data.columns:
        data["is_late_night"] = (
            (data["date_time"].dt.hour >= 22)
            | (data["date_time"].dt.hour <= 2)
        ).astype(int)

    if "is_weekend" not in data.columns:
        data["is_weekend"] = (
            data["date_time"].dt.dayofweek >= 5
        ).astype(int)

    if "monthly_income" not in data.columns:
        data["monthly_income"] = 0.0

    data["month"] = data["date_time"].dt.to_period("M")

    monthly = (
        data.groupby(["user_id", "month"])
        .agg(
            monthly_spend=("amount", "sum"),
            average_transaction=("amount", "mean"),
            transaction_count=("amount", "count"),
            late_night_count=("is_late_night", "sum"),
            weekend_count=("is_weekend", "sum"),
            average_time_between_transactions=(
                "time_since_last_txn",
                "mean",
            ),
            monthly_income=("monthly_income", "first"),
        )
        .reset_index()
    )

    monthly["late_night_ratio"] = np.where(
        monthly["transaction_count"] > 0,
        monthly["late_night_count"]
        / monthly["transaction_count"],
        0.0,
    )

    monthly["weekend_ratio"] = np.where(
        monthly["transaction_count"] > 0,
        monthly["weekend_count"]
        / monthly["transaction_count"],
        0.0,
    )

    monthly["spending_income_ratio"] = np.where(
        monthly["monthly_income"] > 0,
        monthly["monthly_spend"]
        / monthly["monthly_income"],
        0.0,
    )

    monthly = monthly.sort_values(
        ["user_id", "month"]
    ).reset_index(drop=True)

    previous_features = [
        "monthly_spend",
        "average_transaction",
        "transaction_count",
        "late_night_ratio",
        "weekend_ratio",
        "average_time_between_transactions",
        "spending_income_ratio",
    ]

    for feature in previous_features:
        monthly[f"previous_{feature}"] = (
            monthly.groupby("user_id")[feature].shift(1)
        )

    previous_spend = (
        monthly.groupby("user_id")["monthly_spend"].shift(1)
    )

    monthly["spending_growth"] = np.where(
        previous_spend > 0,
        monthly["monthly_spend"] / previous_spend - 1,
        np.nan,
    )

    monthly["spending_growth"] = monthly[
        "spending_growth"
    ].clip(-1, 3)

    # Future information is used only to create the training target.
    monthly["future_spending_ratio"] = (
        monthly.groupby("user_id")[
            "spending_income_ratio"
        ].shift(-1)
    )

    monthly["future_spending_growth"] = (
        monthly.groupby("user_id")[
            "spending_growth"
        ].shift(-1)
    )

    monthly["future_late_night_ratio"] = (
        monthly.groupby("user_id")[
            "late_night_ratio"
        ].shift(-1)
    )

    condition_1 = (
        monthly["future_spending_ratio"] > 0.80
    )

    condition_2 = (
        monthly["future_spending_growth"] > 0.25
    )

    condition_3 = (
        (monthly["future_spending_ratio"] > 0.65)
        & (monthly["future_late_night_ratio"] > 0.30)
    )

    monthly["future_overspending"] = (
        condition_1 | condition_2 | condition_3
    ).astype(int)

    return monthly


def train_risk_model(
    df: pd.DataFrame,
    train_before: str = "2026-05",
):
    """
    Train the Random Forest on historical monthly observations.

    Returns:
        model, modeling_data
        or (None, modeling_data) when there is not enough
        class variation to train safely.
    """
    monthly = prepare_monthly_data(df)

    if monthly.empty:
        return None, monthly

    required = FEATURES + [
        "future_spending_ratio",
        "future_spending_growth",
        "future_late_night_ratio",
    ]

    model_data = monthly.dropna(
        subset=required
    ).copy()

    if model_data.empty:
        return None, model_data

    train_data = model_data[
        model_data["month"] < train_before
    ].copy()

    if train_data.empty:
        return None, model_data

    y_train = train_data["future_overspending"]

    # A classifier needs at least two classes.
    if y_train.nunique() < 2:
        return None, model_data

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(
        train_data[FEATURES],
        y_train,
    )

    return model, model_data


def short_history_fallback(
    transactions: pd.DataFrame,
) -> float:
    """
    Transparent fallback for users without enough monthly history.

    This is a heuristic score, NOT an ML prediction.
    """
    if transactions is None or transactions.empty:
        return 0.0

    data = transactions.copy()

    if "amount" not in data.columns:
        return 0.0

    income = float(
        data["monthly_income"].iloc[-1]
        if "monthly_income" in data.columns
        and not data["monthly_income"].empty
        else 0.0
    )

    spending = float(
        data["amount"].sum()
    )

    # With very short history, use recorded spending/income
    # without presenting it as a trained-model probability.
    ratio = (
        spending / income
        if income > 0
        else 0.0
    )

    late_ratio = 0.0

    if "is_late_night" in data.columns:
        late_ratio = float(
            data["is_late_night"].mean()
        )

    score = min(
        55.0,
        max(0.0, ratio * 70.0),
    )

    score += min(
        20.0,
        max(0.0, late_ratio * 40.0),
    )

    return float(
        min(100.0, score)
    )


def predict_user_risk(
    df: pd.DataFrame,
    user_id: int,
    minimum_months: int = 2,
):
    """
    Return a risk result dictionary.

    The result always identifies its source:
        - 'ml_model'
        - 'short_history_fallback'
        - 'insufficient_data'
    """
    if df is None or df.empty:
        return {
            "score": 0.0,
            "source": "insufficient_data",
            "label": "Insufficient data",
            "months": 0,
        }

    user = df[
        df["user_id"] == user_id
    ].copy()

    if user.empty:
        return {
            "score": 0.0,
            "source": "insufficient_data",
            "label": "Insufficient data",
            "months": 0,
        }

    user["date_time"] = pd.to_datetime(
        user["date_time"],
        errors="coerce",
    )

    months = int(
        user["date_time"]
        .dt.to_period("M")
        .nunique()
    )

    # Train on the supplied historical dataset.
    model, model_data = train_risk_model(df)

    user_monthly = prepare_monthly_data(user)

    if (
        months >= minimum_months
        and model is not None
        and not user_monthly.empty
    ):
        latest = user_monthly.iloc[-1:]

        # We need the previous-month feature row for prediction.
        if latest[FEATURES].notna().all(axis=None):
            probability = float(
                model.predict_proba(
                    latest[FEATURES]
                )[0, 1] * 100
            )

            if probability >= 70:
                label = "High risk"
            elif probability >= 40:
                label = "Moderate risk"
            else:
                label = "Lower risk"

            return {
                "score": probability,
                "source": "ml_model",
                "label": label,
                "months": months,
            }

    # Not enough valid model history -> transparent fallback.
    score = short_history_fallback(user)

    if score >= 70:
        label = "High risk"
    elif score >= 40:
        label = "Moderate risk"
    else:
        label = "Lower risk"

    return {
        "score": score,
        "source": "short_history_fallback",
        "label": label,
        "months": months,
    }


def get_feature_importance(
    df: pd.DataFrame,
):
    """Return trained-model feature importance as a dataframe."""
    model, _ = train_risk_model(df)

    if model is None:
        return pd.DataFrame(
            columns=["feature", "importance"]
        )

    result = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": model.feature_importances_,
        }
    )

    return result.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)


if __name__ == "__main__":
    df = pd.read_csv(
        "processed_transactions.csv"
    )

    model, model_data = train_risk_model(df)

    print("\n==============================================")
    print("       EXPENSE INTELLIGENCE")
    print("       FUTURE RISK PREDICTION")
    print("==============================================")

    print(
        f"Transactions loaded: {len(df):,}"
    )
    print(
        f"Users: {df['user_id'].nunique():,}"
    )
    print(
        f"Modeling rows: {len(model_data):,}"
    )

    if model is None:
        print(
            "\nModel was not trained because "
            "there is insufficient class/history variation."
        )
    else:
        print(
            "\nRisk model trained successfully."
        )

        importance = get_feature_importance(df)

        print("\nFeature importance:")
        print(importance.to_string(index=False))

    print("\n==============================================")
