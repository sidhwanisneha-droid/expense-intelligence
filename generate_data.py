import pandas as pd
import random
from datetime import datetime, timedelta

categories = ["food", "shopping", "transport", "entertainment"]
descriptions = {
    "food": ["Swiggy", "Zomato", "Cafe"],
    "shopping": ["Amazon", "Flipkart"],
    "transport": ["Uber", "Auto"],
    "entertainment": ["Netflix", "Movie"]
}

data = []
start_date = datetime(2026, 1, 1)

for user in range(1, 6):
    current_date = start_date
    
    for _ in range(120):
        num_txn = random.randint(1, 5)
        
        for _ in range(num_txn):
            category = random.choice(categories)
            hour = random.randint(8, 23)

            # Late-night bias
            if category == "food" and random.random() < 0.4:
                hour = random.choice([22, 23, 0])
            
            amount = round(random.uniform(50, 1000), 2)

            data.append({
                "user_id": user,
                "date_time": current_date.replace(hour=hour),
                "amount": amount,
                "description": random.choice(descriptions[category]),
                "category": category
            })
        
        current_date += timedelta(days=1)

df = pd.DataFrame(data)
df.to_csv("transactions.csv", index=False)

print("✅ Dataset created!")