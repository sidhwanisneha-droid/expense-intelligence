import pandas as pd

# Load data
df = pd.read_csv("transactions.csv")

# Convert to datetime
df["date_time"] = pd.to_datetime(df["date_time"])

# -------- TIME FEATURES -------- #
df["hour"] = df["date_time"].dt.hour
df["day_of_week"] = df["date_time"].dt.dayofweek
df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
df["week_of_month"] = df["date_time"].dt.day // 7 + 1

# -------- SPENDING FEATURES -------- #
def spend_bucket(amount):
    if amount < 200:
        return "low"
    elif amount < 700:
        return "medium"
    else:
        return "high"

df["spend_bucket"] = df["amount"].apply(spend_bucket)

# -------- BEHAVIORAL FEATURE -------- #
df = df.sort_values(by=["user_id", "date_time"])

df["time_since_last_txn"] = df.groupby("user_id")["date_time"].diff().dt.total_seconds() / 3600
df["time_since_last_txn"] = df["time_since_last_txn"].fillna(0)

# -------- LATE NIGHT FLAG -------- #
df["is_late_night"] = df["hour"].apply(lambda x: 1 if x >= 22 or x <= 2 else 0)

# Save
df.to_csv("processed_transactions.csv", index=False)

print("✅ Preprocessing Done!")