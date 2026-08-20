import pandas as pd

# Load processed data
df = pd.read_csv("processed_transactions.csv")

# -------- MICRO LEAK -------- #
micro_leaks = df[df["amount"] < 200]
micro_summary = micro_leaks.groupby("user_id")["amount"].agg(["count", "sum"]).reset_index()
micro_summary["type"] = "micro_leak"

# -------- LATE NIGHT LEAK -------- #
late_night = df[df["is_late_night"] == 1]
late_summary = late_night.groupby("user_id")["amount"].agg(["count", "sum"]).reset_index()
late_summary["type"] = "late_night_leak"

# -------- CATEGORY OVERUSE -------- #
category_summary = df.groupby(["user_id", "category"]).size().reset_index(name="count")
high_freq = category_summary[category_summary["count"] > 20]
high_freq["type"] = "category_overuse"

# -------- COMBINE -------- #
leaks = pd.concat([micro_summary, late_summary, high_freq], ignore_index=True)

# -------- SEVERITY -------- #
leaks["severity"] = "low"
leaks.loc[leaks["sum"] > 3000, "severity"] = "high"
leaks.loc[(leaks["sum"] > 1000) & (leaks["sum"] <= 3000), "severity"] = "medium"

# Save
leaks.to_csv("leak_summary.csv", index=False)

print("✅ Leak Detection Completed!")