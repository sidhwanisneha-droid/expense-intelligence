
import pandas as pd

# Load data
df = pd.read_csv("processed_transactions.csv")

# ✅ IMPORTANT FIX: convert to datetime
df["date_time"] = pd.to_datetime(df["date_time"])

# -------- USER INPUT -------- #
user_id = int(input("Enter user ID: "))
category = input("Enter category to reduce (food/shopping/etc): ").lower()
reduction_percent = float(input("Enter reduction % (e.g., 30): "))

# -------- FILTER DATA -------- #
user_data = df[df["user_id"] == user_id]
category_data = user_data[user_data["category"] == category]

# -------- SAFETY CHECK -------- #
if category_data.empty:
    print("⚠️ No data found for this category.")
    exit()

# -------- CALCULATIONS -------- #
total_spend = category_data["amount"].sum()

# Calculate time range
days = (user_data["date_time"].max() - user_data["date_time"].min()).days
months = max(days / 30, 1)

# Monthly spend
monthly_spend = total_spend / months

# Savings
monthly_saving = (monthly_spend * reduction_percent) / 100
yearly_saving = monthly_saving * 12

# -------- OUTPUT -------- #
print("\n📊 Simulation Result:")

print(f"User {user_id} currently spends approx ₹{round(monthly_spend,2)} per month on {category}.")

print(f"\nIf you reduce {category} spending by {reduction_percent}%:")
print(f"👉 Estimated Monthly Savings: ₹{round(monthly_saving,2)}")
print(f"👉 Estimated Yearly Savings: ₹{round(yearly_saving,2)}")

# -------- SMART INSIGHT -------- #
print("\n💡 Insight:")

if monthly_saving > 5000:
    print("⚠️ This is a high-impact category. Reducing it can significantly improve your financial health.")
else:
    print("Reducing this category gradually can help improve your savings without affecting your lifestyle too much.")