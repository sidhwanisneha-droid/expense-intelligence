import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta


# ============================================================
# EXPENSE INTELLIGENCE
# REALISTIC SYNTHETIC DATA GENERATOR
# ============================================================

random.seed(42)
np.random.seed(42)


# ============================================================
# SETTINGS
# ============================================================

NUMBER_OF_USERS = 1000
NUMBER_OF_DAYS = 180

START_DATE = datetime(2026, 1, 1)


# ============================================================
# CATEGORIES
# ============================================================

categories = [
    "food",
    "shopping",
    "transport",
    "entertainment",
    "utilities",
    "health"
]


descriptions = {
    "food": [
        "Swiggy",
        "Zomato",
        "Restaurant",
        "Cafe",
        "Grocery"
    ],

    "shopping": [
        "Amazon",
        "Flipkart",
        "Myntra",
        "Shopping Mall",
        "Online Store"
    ],

    "transport": [
        "Uber",
        "Ola",
        "Metro",
        "Auto",
        "Fuel"
    ],

    "entertainment": [
        "Netflix",
        "Movie",
        "Spotify",
        "Games",
        "Event"
    ],

    "utilities": [
        "Electricity",
        "Internet",
        "Mobile Bill",
        "Water Bill"
    ],

    "health": [
        "Pharmacy",
        "Doctor",
        "Hospital",
        "Health Store"
    ]
}


# ============================================================
# GENERATE USERS
# ============================================================

data = []


for user_id in range(1, NUMBER_OF_USERS + 1):

    # --------------------------------------------------------
    # USER INCOME
    # --------------------------------------------------------

    monthly_income = round(
        np.random.lognormal(
            mean=np.log(50000),
            sigma=0.55
        ),
        2
    )

    # Keep income within realistic limits
    monthly_income = max(
        20000,
        min(monthly_income, 300000)
    )


    # --------------------------------------------------------
    # USER SPENDING BEHAVIOR
    # --------------------------------------------------------

    # Average proportion of income spent
    spending_ratio = np.random.normal(
        0.60,
        0.12
    )

    spending_ratio = max(
        0.35,
        min(spending_ratio, 0.95)
    )


    # Transaction frequency
    transactions_per_day = np.random.normal(
        2.5,
        0.7
    )

    transactions_per_day = max(
        1,
        min(transactions_per_day, 5)
    )


    # Late-night behavior
    late_night_probability = np.random.beta(
        2,
        8
    )


    # Weekend spending tendency
    weekend_multiplier = np.random.normal(
        1.10,
        0.12
    )

    weekend_multiplier = max(
        0.8,
        min(weekend_multiplier, 1.5)
    )


    # Impulse tendency
    impulse_tendency = np.random.beta(
        2,
        6
    )


    # --------------------------------------------------------
    # CATEGORY PREFERENCES
    # --------------------------------------------------------

    category_weights = np.random.dirichlet(
        np.ones(len(categories)) * 3
    )


    # --------------------------------------------------------
    # DAILY SPENDING BUDGET
    # --------------------------------------------------------

    daily_budget = (
        monthly_income *
        spending_ratio /
        30
    )


    # --------------------------------------------------------
    # GENERATE TRANSACTIONS
    # --------------------------------------------------------

    current_date = START_DATE

    for day in range(NUMBER_OF_DAYS):

        # Number of transactions for this day
        number_of_transactions = max(
            1,
            np.random.poisson(transactions_per_day)
        )

        for _ in range(number_of_transactions):

            # -----------------------------------------------
            # CATEGORY
            # -----------------------------------------------

            category = random.choices(
                categories,
                weights=category_weights,
                k=1
            )[0]


            # -----------------------------------------------
            # TIME
            # -----------------------------------------------

            if random.random() < late_night_probability:

                hour = random.choice([
                    22,
                    23,
                    0,
                    1,
                    2
                ])

            else:

                hour = random.randint(
                    7,
                    21
                )


            minute = random.randint(
                0,
                59
            )


            # -----------------------------------------------
            # TRANSACTION AMOUNT
            # -----------------------------------------------

            # Base amount related to daily spending budget
            base_amount = daily_budget / number_of_transactions

            # Log-normal variation creates realistic
            # small + medium + occasional large transactions
            amount = np.random.lognormal(
                mean=np.log(
                    max(base_amount * 0.45, 50)
                ),
                sigma=0.65
            )


            # -----------------------------------------------
            # IMPULSE EFFECT
            # -----------------------------------------------

            if random.random() < impulse_tendency:

                amount *= np.random.uniform(
                    1.2,
                    2.5
                )


            # -----------------------------------------------
            # WEEKEND EFFECT
            # -----------------------------------------------

            transaction_date = current_date

            if transaction_date.weekday() >= 5:

                amount *= weekend_multiplier


            # -----------------------------------------------
            # CATEGORY ADJUSTMENTS
            # -----------------------------------------------

            if category == "utilities":

                amount *= np.random.uniform(
                    1.2,
                    2.0
                )

            elif category == "health":

                amount *= np.random.uniform(
                    1.0,
                    2.5
                )


            # -----------------------------------------------
            # FINAL AMOUNT
            # -----------------------------------------------

            amount = round(
                max(amount, 20),
                2
            )


            # -----------------------------------------------
            # SAVE TRANSACTION
            # -----------------------------------------------

            data.append({

                "user_id": user_id,

                "date_time": transaction_date.replace(
                    hour=hour,
                    minute=minute
                ),

                "amount": amount,

                "description": random.choice(
                    descriptions[category]
                ),

                "category": category,

                "monthly_income": monthly_income

            })


        current_date += timedelta(
            days=1
        )


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(data)


# ============================================================
# SORT DATA
# ============================================================

df = df.sort_values(
    by=[
        "user_id",
        "date_time"
    ]
).reset_index(
    drop=True
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    "transactions.csv",
    index=False
)


# ============================================================
# INFORMATION
# ============================================================

print("\n==============================================")
print("✅ EXPENSE INTELLIGENCE DATASET CREATED")
print("==============================================")

print(
    "Number of users:",
    df["user_id"].nunique()
)

print(
    "Number of transactions:",
    len(df)
)

print(
    "Date range:",
    df["date_time"].min(),
    "to",
    df["date_time"].max()
)

print(
    "Average transaction:",
    round(df["amount"].mean(), 2)
)

print(
    "Average monthly income:",
    round(
        df.groupby("user_id")["monthly_income"]
        .first()
        .mean(),
        2
    )
)

print("\nCategory distribution:")

print(
    df["category"]
    .value_counts()
)

print("\n==============================================")