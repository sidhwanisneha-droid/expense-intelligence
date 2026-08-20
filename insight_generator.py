import pandas as pd

# Load data
df = pd.read_csv("processed_transactions.csv")
leaks = pd.read_csv("leak_summary.csv")

final_insights = []

users = df["user_id"].unique()

for user in users:
    user_text = f"\n🔹 USER {user} INSIGHTS:\n"

    user_leaks = leaks[leaks["user_id"] == user]

    # -------- MICRO LEAK -------- #
    micro_total = 0
    micro = user_leaks[user_leaks["type"] == "micro_leak"]
    if not micro.empty:
        micro_total = round(micro["sum"].values[0], 2)
        user_text += f"- You are spending ₹{micro_total} on frequent small purchases. These micro-expenses are adding up significantly and impacting your savings.\n"

    # -------- LATE NIGHT -------- #
    late_total = 0
    late = user_leaks[user_leaks["type"] == "late_night_leak"]
    if not late.empty:
        late_total = round(late["sum"].values[0], 2)
        user_text += f"- Your spending peaks late at night (10PM–1AM), contributing around ₹{late_total}. This pattern is linked to impulse behavior and reduced spending control.\n"

    # -------- TOP CATEGORY -------- #
    category_counts = df[df["user_id"] == user]["category"].value_counts()
    top_category = category_counts.idxmax()
    count = category_counts.max()

    user_text += f"- Your most frequent spending category is '{top_category}' ({count} transactions), which may require better tracking and limits.\n"

    # -------- SAVINGS ESTIMATION -------- #
    estimated_saving = round((micro_total * 0.2) + (late_total * 0.1), 2)
    # -------- USER TYPE DETECTION -------- #
    user_type = "Balanced Spender"

    if late_total > micro_total:
        user_type = "Impulse Spender"
    elif micro_total > 7000:
        user_type = "Micro-Spender"

    # -------- FINAL SUMMARY -------- #
    user_text += f"\n💡 Overall Insight ({user_type}):\n"
    user_text += "You show a pattern of high-frequency and impulse-driven spending. "

    user_text += "If this behavior continues, a significant portion of your income may go toward non-essential expenses.\n"

    user_text += f"By reducing small frequent purchases and controlling late-night spending, you can potentially save around ₹{estimated_saving} per month.\n"

    final_insights.append(user_text)

# -------- SAVE FILE -------- #
with open("insights.txt", "w", encoding="utf-8") as f:
    for insight in final_insights:
        f.write(insight + "\n")

print("✅ Smart Insights Generated Successfully!")