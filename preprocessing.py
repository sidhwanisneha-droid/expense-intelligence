import pandas as pd


def spend_bucket(amount):
    if amount < 200:
        return "low"
    elif amount < 700:
        return "medium"
    return "high"


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare transaction data with the project's existing
    time, spending, transaction-gap, late-night, and income features.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    data = df.copy()

    required = [
        "user_id",
        "date_time",
        "amount",
        "monthly_income",
    ]

    missing = [
        col for col in required
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    data["date_time"] = pd.to_datetime(
        data["date_time"],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            "user_id",
            "date_time",
            "amount",
            "monthly_income",
        ]
    ).copy()

    data["hour"] = data["date_time"].dt.hour

    data["day_of_week"] = (
        data["date_time"].dt.dayofweek
    )

    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    data["week_of_month"] = (
        data["date_time"].dt.day // 7 + 1
    )

    data["spend_bucket"] = (
        data["amount"].apply(spend_bucket)
    )

    data = data.sort_values(
        by=[
            "user_id",
            "date_time",
        ]
    ).reset_index(
        drop=True
    )

    data["time_since_last_txn"] = (
        data.groupby("user_id")["date_time"]
        .diff()
        .dt.total_seconds()
        .div(3600)
        .fillna(0)
    )

    # Existing project definition:
    # late-night = 10 PM through 2 AM.
    data["is_late_night"] = (
        (data["hour"] >= 22)
        | (data["hour"] <= 2)
    ).astype(int)

    data["daily_income"] = (
        data["monthly_income"] / 30
    )

    data["income_spending_ratio"] = (
        data["amount"]
        / data["daily_income"].replace(0, pd.NA)
    ).fillna(0.0)

    return data


def preprocess_file(
    input_file: str = "transactions.csv",
    output_file: str = "processed_transactions.csv",
) -> pd.DataFrame:
    """Load, preprocess, and save transaction data."""
    df = pd.read_csv(input_file)

    processed = preprocess_data(df)

    processed.to_csv(
        output_file,
        index=False,
    )

    return processed


if __name__ == "__main__":
    print("\n========== PREPROCESSING ==========")

    processed = preprocess_file()

    print(
        "Processed transactions:",
        len(processed),
    )

    if not processed.empty:
        print(
            "Users:",
            processed["user_id"].nunique(),
        )

    print(
        "Features created:"
    )

    print(
        [
            "hour",
            "day_of_week",
            "is_weekend",
            "week_of_month",
            "spend_bucket",
            "time_since_last_txn",
            "is_late_night",
            "daily_income",
            "income_spending_ratio",
        ]
    )

    print(
        "\nSaved as: processed_transactions.csv"
    )
    print(
        "============================================"
    )
