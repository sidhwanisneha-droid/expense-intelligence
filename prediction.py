import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Load data
df = pd.read_csv("processed_transactions.csv")

# -------- ADD NOISE -------- #
df["amount"] = df["amount"] + np.random.normal(0, 50, size=len(df))

# -------- EXTRA FEATURE -------- #
df["is_high_amount"] = df["amount"].apply(lambda x: 1 if x > 700 else 0)

# -------- IMPROVED LABEL -------- #
df["overspend"] = (
    ((df["is_late_night"] == 1) & (df["amount"] > 500)) |
    (np.random.rand(len(df)) < 0.1)
).astype(int)

# -------- FEATURES -------- #
features = ["amount", "is_weekend", "time_since_last_txn", "is_high_amount"]
X = df[features]
y = df["overspend"]

# -------- SPLIT -------- #
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------- MODEL -------- #
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -------- EVALUATION -------- #
accuracy = model.score(X_test, y_test)
print(f"\n✅ Model Accuracy: {round(accuracy*100,2)}%")

y_pred = model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------- FEATURE IMPORTANCE -------- #
print("\nFeature Importance:")
importance = model.feature_importances_

for i, col in enumerate(features):
    print(f"{col}: {round(importance[i],3)}")

# -------- SAMPLE -------- #
sample = X.iloc[0:1]
prediction = model.predict(sample)

print("\nSample Prediction:",
      "⚠️ Overspending Risk" if prediction[0] == 1 else "✅ Safe Spending")