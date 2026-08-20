import pandas as pd
from sklearn.cluster import KMeans

# Load data
df = pd.read_csv("processed_transactions.csv")

# -------- FEATURE AGGREGATION -------- #
user_features = df.groupby("user_id").agg({
    "amount": ["sum", "mean"],
    "is_late_night": "sum",
    "time_since_last_txn": "mean"
})

# Flatten columns
user_features.columns = ["total_spend", "avg_spend", "late_night_count", "avg_gap"]

# -------- KMEANS -------- #
kmeans = KMeans(n_clusters=3, random_state=42)
user_features["cluster"] = kmeans.fit_predict(user_features)

# -------- PRINT CLUSTER CENTERS -------- #
print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# -------- MAP CLUSTERS TO LABELS -------- #
cluster_map = {
    0: "Balanced Spender",
    1: "Impulse Spender",
    2: "Micro Spender"
}

user_features["user_type"] = user_features["cluster"].map(cluster_map)

# Save
user_features.to_csv("user_clusters.csv")

print("\n✅ Clustering Completed!\n")
print(user_features)