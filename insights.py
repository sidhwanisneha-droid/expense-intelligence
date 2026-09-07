import pandas as pd
from pathlib import Path


def prepare_user_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build the user-level summary used by the insight engine."""
    if df is None or df.empty:
        return pd.DataFrame()

    data = df.copy()

    if "user_id" not in data.columns or "amount" not in data.columns:
        return pd.DataFrame()

    summary = (
        data.groupby("user_id")
        .agg(
            total_spend=("amount", "sum"),
            average_transaction=("amount", "mean"),
            transaction_count=("amount", "count"),
            monthly_income=("monthly_income", "first"),
        )
        .reset_index()
    )

    # Keep the same six-month normalization used by the
    # original project insight engine.
    summary["monthly_spend"] = (
        summary["total_spend"] / 6
    )

    summary["spending_income_ratio"] = (
        summary["monthly_spend"]
        / summary["monthly_income"].replace(0, pd.NA)
    ).fillna(0.0)

    return summary


def _severity_icon(severity: str) -> str:
    if severity == "high":
        return "🔴"
    if severity == "medium":
        return "🟡"
    return "🟢"


def generate_user_insight(
    df: pd.DataFrame,
    leaks: pd.DataFrame,
    clusters: pd.DataFrame,
    user_id: int,
) -> dict:
    """
    Generate a structured personalized insight report for one user.

    Returns a dictionary so app.py can present the insight cleanly
    without parsing the saved text file.
    """
    summary = prepare_user_summary(df)

    empty = {
        "user_id": int(user_id),
        "profile": "New User",
        "monthly_income": 0.0,
        "monthly_spend": 0.0,
        "spending_ratio": 0.0,
        "financial_status": "NO DATA",
        "status_description": "Add spending data to generate personalized insights.",
        "leaks": [],
        "top_category": None,
        "top_category_amount": 0.0,
        "top_category_percentage": 0.0,
        "recommendations": [],
        "overall_insight": "There is not enough spending data to generate a personalized insight.",
    }

    if summary.empty:
        return empty

    user_summary = summary[
        summary["user_id"] == user_id
    ]

    if user_summary.empty:
        return empty

    user = user_summary.iloc[0]

    monthly_income = float(
        user["monthly_income"]
    )
    monthly_spend = float(
        user["monthly_spend"]
    )
    spending_ratio = float(
        user["spending_income_ratio"]
    )

    profile = "Balanced Spender"

    if clusters is not None and not clusters.empty:
        if (
            "user_id" in clusters.columns
            and "user_type" in clusters.columns
        ):
            result = clusters[
                clusters["user_id"] == user_id
            ]
            if not result.empty:
                profile = str(
                    result.iloc[0]["user_type"]
                )

    if spending_ratio >= 0.80:
        financial_status = "HIGH SPENDING"
        status_description = (
            "A large portion of your estimated monthly income "
            "is being used for expenses."
        )
    elif spending_ratio >= 0.60:
        financial_status = "MODERATE SPENDING"
        status_description = (
            "Your spending is taking a noticeable share "
            "of your estimated monthly income."
        )
    else:
        financial_status = "CONTROLLED SPENDING"
        status_description = (
            "Your estimated spending is currently below "
            "60% of your monthly income."
        )

    user_leaks = pd.DataFrame()

    if leaks is not None and not leaks.empty:
        user_leaks = leaks[
            leaks["user_id"] == user_id
        ].copy()

    leak_records = []

    if not user_leaks.empty:
        sort_cols = []

        if "severity" in user_leaks.columns:
            severity_order = {
                "high": 0,
                "medium": 1,
                "low": 2,
            }
            user_leaks["_severity_order"] = (
                user_leaks["severity"]
                .map(severity_order)
                .fillna(3)
            )
            sort_cols.append("_severity_order")

        if "leak_amount" in user_leaks.columns:
            sort_cols.append("leak_amount")

        if sort_cols:
            user_leaks = user_leaks.sort_values(
                sort_cols,
                ascending=[True] * len(sort_cols),
            )

        for _, leak in user_leaks.head(3).iterrows():
            leak_records.append(
                {
                    "icon": _severity_icon(
                        str(leak.get("severity", "low"))
                    ),
                    "leak_type": str(
                        leak.get(
                            "leak_type",
                            "behavioral pattern",
                        )
                    ),
                    "category": str(
                        leak.get(
                            "category",
                            "multiple",
                        )
                    ),
                    "amount": float(
                        leak.get(
                            "leak_amount",
                            0.0,
                        )
                    ),
                    "percentage": float(
                        leak.get(
                            "leak_percentage",
                            0.0,
                        )
                    ),
                    "severity": str(
                        leak.get(
                            "severity",
                            "low",
                        )
                    ),
                    "evidence": str(
                        leak.get(
                            "evidence",
                            "",
                        )
                    ),
                    "description": str(
                        leak.get(
                            "description",
                            "",
                        )
                    ),
                }
            )

    user_transactions = df[
        df["user_id"] == user_id
    ].copy()

    if user_transactions.empty:
        return empty

    category_spending = (
        user_transactions
        .groupby("category")["amount"]
        .sum()
        .sort_values(ascending=False)
    )

    top_category = str(
        category_spending.index[0]
    )
    top_category_amount = float(
        category_spending.iloc[0]
    )

    total_spend = float(
        user["total_spend"]
    )

    top_category_percentage = (
        top_category_amount
        / total_spend
        * 100
        if total_spend > 0
        else 0.0
    )

    recommendations = []

    if spending_ratio >= 0.80:
        recommendations.append(
            "Set a monthly discretionary spending limit "
            "to prevent expenses from approaching your income."
        )
    elif spending_ratio >= 0.60:
        recommendations.append(
            "Track your discretionary expenses closely "
            "and review them before the end of each month."
        )
    else:
        recommendations.append(
            "Maintain your current spending discipline "
            "and continue monitoring your monthly ratio."
        )

    if not user_leaks.empty and "leak_type" in user_leaks.columns:

        micro = user_leaks[
            user_leaks["leak_type"]
            == "micro_spending"
        ]

        if not micro.empty:
            micro_amount = float(
                micro["leak_amount"].sum()
            )
            recommendations.append(
                f"Review small frequent purchases. "
                f"₹{micro_amount:,.2f} was associated with "
                f"micro-spending patterns."
            )

        late = user_leaks[
            user_leaks["leak_type"]
            == "late_night"
        ]

        if not late.empty:
            recommendations.append(
                "Consider setting a late-night spending "
                "limit because your transaction timing shows "
                "a recurring night-time pattern."
            )

        spikes = user_leaks[
            user_leaks["leak_type"]
            == "category_spike"
        ]

        if not spikes.empty:
            category = str(
                spikes.iloc[0]["category"]
            )
            recommendations.append(
                f"Review recent spending in {category}. "
                "A significant increase was detected compared "
                "with the previous month."
            )

        concentration = user_leaks[
            user_leaks["leak_type"]
            == "category_concentration"
        ]

        if not concentration.empty:
            concentration = concentration.sort_values(
                "leak_percentage",
                ascending=False,
            )
            category = str(
                concentration.iloc[0]["category"]
            )
            recommendations.append(
                f"Consider setting a separate budget for "
                f"{category} because it represents a large "
                "share of your spending."
            )

    recommendations = recommendations[:4]

    if (
        spending_ratio >= 0.80
        and not user_leaks.empty
    ):
        overall_insight = (
            "Your spending profile shows elevated financial "
            "pressure combined with behavioral spending signals. "
            "Focus on controlling the largest recurring patterns "
            "rather than attempting to reduce every expense."
        )
    elif not user_leaks.empty:
        overall_insight = (
            "Your overall spending is not necessarily excessive, "
            "but several behavioral patterns could gradually "
            "increase your expenses if they continue."
        )
    else:
        overall_insight = (
            "Your current spending behavior appears relatively "
            "stable. Continue monitoring your spending patterns "
            "to maintain financial control."
        )

    return {
        "user_id": int(user_id),
        "profile": profile,
        "monthly_income": monthly_income,
        "monthly_spend": monthly_spend,
        "spending_ratio": spending_ratio,
        "financial_status": financial_status,
        "status_description": status_description,
        "leaks": leak_records,
        "top_category": top_category,
        "top_category_amount": top_category_amount,
        "top_category_percentage": top_category_percentage,
        "recommendations": recommendations,
        "overall_insight": overall_insight,
    }


def generate_all_insights(
    df: pd.DataFrame,
    leaks: pd.DataFrame,
    clusters: pd.DataFrame,
):
    """Generate structured insights for every user."""
    summary = prepare_user_summary(df)

    reports = {}

    if summary.empty:
        return reports

    for user_id in summary["user_id"].astype(int):
        reports[user_id] = generate_user_insight(
            df=df,
            leaks=leaks,
            clusters=clusters,
            user_id=user_id,
        )

    return reports


def report_to_text(report: dict) -> str:
    """Convert one structured report to the original text style."""
    lines = []

    lines.append("=" * 60)
    lines.append(
        f"USER {report['user_id']} — "
        "PERSONALIZED EXPENSE REPORT"
    )
    lines.append("=" * 60)

    lines.append(
        f"\n👤 Spending Profile: {report['profile']}"
    )
    lines.append(
        f"💰 Monthly income: "
        f"₹{report['monthly_income']:,.2f}"
    )
    lines.append(
        f"💸 Estimated monthly spending: "
        f"₹{report['monthly_spend']:,.2f}"
    )
    lines.append(
        f"📊 Spending-to-income ratio: "
        f"{report['spending_ratio'] * 100:.1f}%"
    )

    lines.append(
        f"\n{'🔴' if report['financial_status'] == 'HIGH SPENDING' else '🟡' if report['financial_status'] == 'MODERATE SPENDING' else '🟢'} "
        f"Financial Status: {report['financial_status']}"
    )
    lines.append(
        report["status_description"]
    )

    if report["leaks"]:
        lines.append(
            "\n⚠️ BEHAVIORAL PATTERNS DETECTED:"
        )

        for leak in report["leaks"]:
            lines.append(
                f"\n{leak['icon']} "
                f"{leak['leak_type'].replace('_', ' ').title()}"
            )

            if leak["category"] != "multiple":
                lines.append(
                    f"   Category: {leak['category']}"
                )

            lines.append(
                f"   Amount involved: "
                f"₹{leak['amount']:,.2f}"
            )
            lines.append(
                f"   Impact: "
                f"{leak['percentage']:.1f}%"
            )
            lines.append(
                f"   Evidence: {leak['evidence']}"
            )
    else:
        lines.append(
            "\n✅ No significant behavioral leaks detected."
        )

    lines.append(
        "\n📌 TOP SPENDING CATEGORY"
    )
    lines.append(
        f"{str(report['top_category']).title()}: "
        f"₹{report['top_category_amount']:,.2f} "
        f"({report['top_category_percentage']:.1f}% "
        "of total spending)"
    )

    lines.append(
        "\n💡 PERSONALIZED RECOMMENDATIONS"
    )

    for recommendation in report["recommendations"]:
        lines.append(
            f"• {recommendation}"
        )

    lines.append(
        "\n🧠 OVERALL INSIGHT"
    )
    lines.append(
        report["overall_insight"]
    )

    return "\n".join(lines)


def save_all_insights(
    df: pd.DataFrame,
    leaks: pd.DataFrame,
    clusters: pd.DataFrame,
    output_file: str = "insights.txt",
):
    """Generate and save all user reports."""
    reports = generate_all_insights(
        df=df,
        leaks=leaks,
        clusters=clusters,
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:
        for user_id, report in reports.items():
            file.write(
                report_to_text(report)
            )
            file.write("\n\n")

    return reports


if __name__ == "__main__":
    df = pd.read_csv(
        "processed_transactions.csv"
    )

    leaks_path = Path(
        "leak_summary.csv"
    )
    clusters_path = Path(
        "user_clusters.csv"
    )

    leaks = (
        pd.read_csv(leaks_path)
        if leaks_path.exists()
        else pd.DataFrame()
    )

    clusters = (
        pd.read_csv(clusters_path)
        if clusters_path.exists()
        else pd.DataFrame()
    )

    reports = save_all_insights(
        df=df,
        leaks=leaks,
        clusters=clusters,
    )

    print(
        "\n=============================================="
    )
    print(
        "       SMART INSIGHT ENGINE"
    )
    print(
        "=============================================="
    )
    print(
        f"Users processed: {len(reports):,}"
    )
    print(
        "Saved as: insights.txt"
    )
    print(
        "=============================================="
    )
