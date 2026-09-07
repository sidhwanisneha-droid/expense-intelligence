


def simulate_category_reduction(
    df: pd.DataFrame,
    user_id: int,
    category: str,
    reduction_percent: float,
):
    """
    Simulate monthly and yearly savings from reducing a user's
    spending in one category.

    Returns a dictionary suitable for display in Streamlit.
    """

    if df is None or df.empty:
        return {
            "success": False,
            "message": "No spending data is available.",
        }

    data = df.copy()

    if "date_time" not in data.columns:
        return {
            "success": False,
            "message": "Date information is unavailable.",
        }

    if "user_id" not in data.columns or "category" not in data.columns:
        return {
            "success": False,
            "message": "Required transaction fields are missing.",
        }

    data["date_time"] = pd.to_datetime(
        data["date_time"],
        errors="coerce",
    )

    user_data = data[
        data["user_id"] == user_id
    ].copy()

    category = str(category).strip().lower()

    category_data = user_data[
        user_data["category"].astype(str).str.lower()
        == category
    ]

    if category_data.empty:
        return {
            "success": False,
            "message": (
                f"No spending data found for '{category}'."
            ),
        }

    reduction_percent = max(
        0.0,
        min(
            100.0,
            float(reduction_percent),
        ),
    )

    total_spend = float(
        category_data["amount"].sum()
    )

    valid_dates = user_data["date_time"].dropna()

    if valid_dates.empty:
        months = 1.0
    else:
        days = (
            valid_dates.max()
            - valid_dates.min()
        ).days

        months = max(
            days / 30,
            1.0,
        )

    monthly_spend = (
        total_spend / months
    )

    monthly_saving = (
        monthly_spend
        * reduction_percent
        / 100
    )

    yearly_saving = (
        monthly_saving * 12
    )

    simulated_category_spend = (
        monthly_spend - monthly_saving
    )

    # Overall monthly spending is based on the user's
    # recorded transactions over the available time range.
    total_user_spend = float(
        user_data["amount"].sum()
    )

    monthly_total_spend = (
        total_user_spend / months
    )

    simulated_total_spend = (
        monthly_total_spend
        - monthly_saving
    )

    income = 0.0

    if "monthly_income" in user_data.columns:
        income = float(
            user_data["monthly_income"]
            .dropna()
            .iloc[0]
            if not user_data["monthly_income"]
            .dropna()
            .empty
            else 0.0
        )

    current_ratio = (
        monthly_total_spend / income
        if income > 0
        else 0.0
    )

    simulated_ratio = (
        simulated_total_spend / income
        if income > 0
        else 0.0
    )

    if monthly_saving > 5000:
        insight = (
            "High-impact category: reducing it could "
            "significantly improve your monthly buffer."
        )
    elif monthly_saving > 0:
        insight = (
            "A gradual reduction here can increase your "
            "monthly savings without requiring a complete "
            "change in spending habits."
        )
    else:
        insight = (
            "No reduction is being simulated yet."
        )

    return {
        "success": True,
        "user_id": int(user_id),
        "category": category,
        "reduction_percent": reduction_percent,
        "months": months,
        "current_category_monthly": monthly_spend,
        "simulated_category_monthly": simulated_category_spend,
        "monthly_saving": monthly_saving,
        "yearly_saving": yearly_saving,
        "current_total_monthly": monthly_total_spend,
        "simulated_total_monthly": simulated_total_spend,
        "income": income,
        "current_ratio": current_ratio,
        "simulated_ratio": simulated_ratio,
        "insight": insight,
    }


def simulate_from_processed_file(
    user_id: int,
    category: str,
    reduction_percent: float,
    input_file: str = "processed_transactions.csv",
):
    """Convenience wrapper for command-line use."""
    df = pd.read_csv(input_file)

    return simulate_category_reduction(
        df=df,
        user_id=user_id,
        category=category,
        reduction_percent=reduction_percent,
    )


if __name__ == "__main__":
    print(
        "\n=============================================="
    )
    print(
        "          WHAT-IF SPENDING SIMULATOR"
    )
    print(
        "=============================================="
    )

    try:
        user_id = int(
            input("Enter user ID: ")
        )

        category = input(
            "Enter category to reduce (food/shopping/etc): "
        ).strip().lower()

        reduction_percent = float(
            input(
                "Enter reduction % (e.g., 30): "
            )
        )

        result = simulate_from_processed_file(
            user_id=user_id,
            category=category,
            reduction_percent=reduction_percent,
        )

        if not result["success"]:
            print(
                f"\nNo simulation: {result['message']}"
            )
        else:
            print("\nSimulation Result:")
            print(
                f"Current monthly {category} spending: "
                f"₹{result['current_category_monthly']:,.2f}"
            )
            print(
                f"Simulated monthly {category} spending: "
                f"₹{result['simulated_category_monthly']:,.2f}"
            )
            print(
                f"Estimated monthly savings: "
                f"₹{result['monthly_saving']:,.2f}"
            )
            print(
                f"Estimated yearly savings: "
                f"₹{result['yearly_saving']:,.2f}"
            )
            print(
                f"Current spending/income: "
                f"{result['current_ratio'] * 100:.1f}%"
            )
            print(
                f"Simulated spending/income: "
                f"{result['simulated_ratio'] * 100:.1f}%"
            )
            print(
                f"\nInsight: {result['insight']}"
            )

    except ValueError:
        print(
            "Please enter valid numeric values."
        )

    print(
        "\n=============================================="
    )
