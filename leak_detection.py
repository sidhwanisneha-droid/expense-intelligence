import pandas as pd
import numpy as np


# ============================================================
# EXPENSE INTELLIGENCE
# BEHAVIOR-BASED LEAK DETECTION
# ============================================================

MIN_HISTORY_TRANSACTIONS = 5


def detect_leaks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect behavioral spending leaks for the supplied transactions.

    Minimum history:
    - Fewer than 5 transactions -> no meaningful leak detection.
    - Category spikes require at least 2 months of data.
    """

    columns = [
        "user_id",
        "month",
        "category",
        "leak_type",
        "description",
        "leak_amount",
        "leak_percentage",
        "severity",
        "evidence",
    ]

    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

    data = df.copy()

    if "date_time" not in data.columns or "amount" not in data.columns:
        return pd.DataFrame(columns=columns)

    data["date_time"] = pd.to_datetime(
        data["date_time"],
        errors="coerce",
    )
    data = data.dropna(
        subset=["date_time", "amount"]
    )

    if data.empty or len(data) < MIN_HISTORY_TRANSACTIONS:
        return pd.DataFrame(columns=columns)

    if "category" not in data.columns:
        data["category"] = "uncategorized"

    if "is_late_night" not in data.columns:
        data["is_late_night"] = (
            (data["date_time"].dt.hour >= 22)
            | (data["date_time"].dt.hour <= 2)
        ).astype(int)

    if "monthly_income" not in data.columns:
        data["monthly_income"] = 0.0

    data["month"] = data["date_time"].dt.to_period("M")

    monthly_user = (
        data.groupby(["user_id", "month"])
        .agg(
            monthly_spend=("amount", "sum"),
            transaction_count=("amount", "count"),
            monthly_income=("monthly_income", "first"),
            late_night_count=("is_late_night", "sum"),
        )
        .reset_index()
    )

    monthly_user["spending_income_ratio"] = np.where(
        monthly_user["monthly_income"] > 0,
        monthly_user["monthly_spend"]
        / monthly_user["monthly_income"],
        0.0,
    )

    monthly_user["late_night_ratio"] = np.where(
        monthly_user["transaction_count"] > 0,
        monthly_user["late_night_count"]
        / monthly_user["transaction_count"],
        0.0,
    )

    monthly_user = monthly_user.sort_values(
        ["user_id", "month"]
    ).reset_index(drop=True)

    monthly_user["previous_spend"] = (
        monthly_user
        .groupby("user_id")["monthly_spend"]
        .shift(1)
    )

    monthly_user["previous_late_night_ratio"] = (
        monthly_user
        .groupby("user_id")["late_night_ratio"]
        .shift(1)
    )

    monthly_user["spending_change"] = (
        (
            monthly_user["monthly_spend"]
            - monthly_user["previous_spend"]
        )
        / monthly_user["previous_spend"]
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    micro = data[data["amount"] < 200]

    micro_monthly = (
        micro.groupby(["user_id", "month"])
        .agg(
            micro_transactions=("amount", "count"),
            micro_spend=("amount", "sum"),
        )
        .reset_index()
    )

    monthly_user = monthly_user.merge(
        micro_monthly,
        on=["user_id", "month"],
        how="left",
    )

    monthly_user[
        ["micro_transactions", "micro_spend"]
    ] = monthly_user[
        ["micro_transactions", "micro_spend"]
    ].fillna(0)

    monthly_user["micro_spending_ratio"] = np.where(
        monthly_user["transaction_count"] > 0,
        monthly_user["micro_transactions"]
        / monthly_user["transaction_count"],
        0.0,
    )

    category_monthly = (
        data.groupby(["user_id", "month", "category"])
        .agg(
            category_spend=("amount", "sum"),
            category_transactions=("amount", "count"),
        )
        .reset_index()
    )

    category_monthly = category_monthly.merge(
        monthly_user[
            [
                "user_id",
                "month",
                "monthly_spend",
            ]
        ],
        on=["user_id", "month"],
        how="left",
    )

    category_monthly["category_share"] = np.where(
        category_monthly["monthly_spend"] > 0,
        category_monthly["category_spend"]
        / category_monthly["monthly_spend"],
        0.0,
    )

    category_monthly = category_monthly.sort_values(
        ["user_id", "category", "month"]
    )

    category_monthly["previous_category_spend"] = (
        category_monthly
        .groupby(["user_id", "category"])["category_spend"]
        .shift(1)
    )

    category_monthly["category_growth"] = (
        (
            category_monthly["category_spend"]
            - category_monthly["previous_category_spend"]
        )
        / category_monthly["previous_category_spend"]
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    leaks = []

    # ==========================================================
    # MONTHLY USER BEHAVIOR
    # ==========================================================

    for _, row in monthly_user.iterrows():

        user_id = int(row["user_id"])
        month = str(row["month"])

        # ------------------------------------------------------
        # MICRO-SPENDING
        # ------------------------------------------------------

        if (
            int(row["transaction_count"]) >= 5
            and row["micro_spending_ratio"] >= 0.35
            and row["micro_spend"] >= 1000
        ):
            leaks.append(
                {
                    "user_id": user_id,
                    "month": month,
                    "category": "multiple",
                    "leak_type": "micro_spending",
                    "description":
                        "Small transactions are accumulating into a significant amount.",
                    "leak_amount":
                        round(float(row["micro_spend"]), 2),
                    "leak_percentage":
                        round(
                            float(
                                row["micro_spend"]
                                / row["monthly_spend"]
                                * 100
                            ),
                            2,
                        ),
                    "evidence":
                        f'{int(row["micro_transactions"])} small transactions',
                }
            )

        # ------------------------------------------------------
        # LATE-NIGHT
        # Require current + previous month comparison.
        # ------------------------------------------------------

        if (
            int(row["transaction_count"]) >= 3
            and row["late_night_ratio"] >= 0.30
            and pd.notna(row["previous_late_night_ratio"])
        ):
            leaks.append(
                {
                    "user_id": user_id,
                    "month": month,
                    "category": "multiple",
                    "leak_type": "late_night",
                    "description":
                        "A high proportion of transactions occur during late-night hours.",
                    "leak_amount":
                        round(
                            float(
                                row["monthly_spend"]
                                * row["late_night_ratio"]
                            ),
                            2,
                        ),
                    "leak_percentage":
                        round(
                            float(row["late_night_ratio"] * 100),
                            2,
                        ),
                    "evidence":
                        (
                            f'{row["late_night_ratio"] * 100:.1f}% '
                            "of transactions were late-night"
                        ),
                }
            )

    # ==========================================================
    # CATEGORY PATTERNS
    # ==========================================================

    for _, row in category_monthly.iterrows():

        growth = row["category_growth"]

        if pd.isna(growth):
            continue

        # Category spike: meaningful increase vs previous month.
        if growth >= 0.40:
            leaks.append(
                {
                    "user_id": int(row["user_id"]),
                    "month": str(row["month"]),
                    "category": str(row["category"]),
                    "leak_type": "category_spike",
                    "description":
                        "Spending in this category increased sharply compared with the previous month.",
                    "leak_amount":
                        round(float(row["category_spend"]), 2),
                    "leak_percentage":
                        round(float(row["category_share"] * 100), 2),
                    "evidence":
                        f'Category spending increased {growth * 100:.1f}%',
                }
            )

        # Category concentration: only meaningful with a reasonable
        # amount of transaction history.
        if (
            int(row["category_transactions"]) >= 2
            and row["category_share"] >= 0.35
            and row["category_spend"] >= 3000
        ):
            leaks.append(
                {
                    "user_id": int(row["user_id"]),
                    "month": str(row["month"]),
                    "category": str(row["category"]),
                    "leak_type": "category_concentration",
                    "description":
                        "A large share of monthly spending is concentrated in one category.",
                    "leak_amount":
                        round(float(row["category_spend"]), 2),
                    "leak_percentage":
                        round(float(row["category_share"] * 100), 2),
                    "evidence":
                        f'{row["category_share"] * 100:.1f}% of monthly spending',
                }
            )

    leaks = pd.DataFrame(leaks)

    if leaks.empty:
        return pd.DataFrame(columns=columns)

    leaks = leaks.drop_duplicates(
        subset=[
            "user_id",
            "month",
            "category",
            "leak_type",
        ]
    )

    # ==========================================================
    # SEVERITY
    # ==========================================================

    def get_severity(row):
        percentage = float(row["leak_percentage"])
        amount = float(row["leak_amount"])

        if percentage >= 40 or amount >= 10000:
            return "high"

        if percentage >= 25 or amount >= 5000:
            return "medium"

        return "low"

    leaks["severity"] = leaks.apply(
        get_severity,
        axis=1,
    )

    return leaks[
        columns
    ].sort_values(
        ["severity", "leak_amount"],
        ascending=[True, False],
    ).reset_index(drop=True)


def save_leaks(
    df: pd.DataFrame,
    output_file: str = "leak_summary.csv",
) -> pd.DataFrame:
    """Detect leaks and save the resulting summary CSV."""
    leaks = detect_leaks(df)
    leaks.to_csv(
        output_file,
        index=False,
    )
    return leaks


if __name__ == "__main__":
    input_file = "processed_transactions.csv"

    df = pd.read_csv(input_file)
    leaks = save_leaks(df)

    print("\n==============================================")
    print("       BEHAVIOR-BASED LEAK DETECTION")
    print("==============================================")
    print(f"Transactions loaded: {len(df):,}")
    print(f"Users: {df['user_id'].nunique():,}")
    print(f"Minimum history: {MIN_HISTORY_TRANSACTIONS} transactions")
    print(f"Total behavioral leaks: {len(leaks):,}")

    if not leaks.empty:
        print("\nLeak types:")
        print(leaks["leak_type"].value_counts())

        print("\nSeverity distribution:")
        print(leaks["severity"].value_counts())

    else:
        print("\nNo significant behavioral leaks detected.")

    print("\nSaved as: leak_summary.csv")
    print("==============================================")
