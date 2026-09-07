import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "spending_income_ratio",
    "avg_spend",
    "transaction_count",
    "late_night_ratio",
    "avg_gap",
]


def build_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create user-level behavioral features for clustering."""
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["user_id", *FEATURES, "monthly_income", "total_spend", "monthly_spend"]
        )

    data = df.copy()

    if "amount" not in data.columns or "user_id" not in data.columns:
        return pd.DataFrame()

    if "is_late_night" not in data.columns:
        if "date_time" in data.columns:
            dt = pd.to_datetime(data["date_time"], errors="coerce")
            data["is_late_night"] = (
                (dt.dt.hour >= 22) | (dt.dt.hour <= 2)
            ).astype(int)
        else:
            data["is_late_night"] = 0

    if "time_since_last_txn" not in data.columns:
        if "date_time" in data.columns:
            dt = pd.to_datetime(data["date_time"], errors="coerce")
            data["time_since_last_txn"] = (
                dt.sort_values()
                .groupby(data.loc[dt.sort_values().index, "user_id"])
                .diff()
                .dt.total_seconds()
                / 3600
            )
            # The construction above can be awkward after sorting; use a
            # safer per-user calculation below.
            data["time_since_last_txn"] = 0.0
            temp = data.copy()
            temp["_dt"] = pd.to_datetime(temp["date_time"], errors="coerce")
            temp = temp.sort_values(["user_id", "_dt"])
            temp["time_since_last_txn"] = (
                temp.groupby("user_id")["_dt"]
                .diff()
                .dt.total_seconds()
                .div(3600)
                .fillna(0)
            )
            data = temp.drop(columns=["_dt"])
        else:
            data["time_since_last_txn"] = 0.0

    if "monthly_income" not in data.columns:
        data["monthly_income"] = 0.0

    user_features = (
        data.groupby("user_id")
        .agg(
            total_spend=("amount", "sum"),
            avg_spend=("amount", "mean"),
            transaction_count=("amount", "count"),
            late_night_ratio=("is_late_night", "mean"),
            avg_gap=("time_since_last_txn", "mean"),
            monthly_income=("monthly_income", "first"),
        )
        .reset_index()
    )

    user_features["monthly_spend"] = np.where(
        user_features["transaction_count"] > 0,
        user_features["total_spend"] / 6,
        0.0,
    )

    user_features["spending_income_ratio"] = np.where(
        user_features["monthly_income"] > 0,
        user_features["monthly_spend"]
        / user_features["monthly_income"],
        0.0,
    )

    return user_features


def label_clusters(cluster_profile: pd.DataFrame) -> dict:
    """Automatically assign the existing project profile labels."""
    labels = {}

    if cluster_profile.empty:
        return labels

    for cluster in cluster_profile.index:
        row = cluster_profile.loc[cluster]

        if (
            row["transaction_count"]
            == cluster_profile["transaction_count"].max()
        ):
            label = "Frequent Spender"
        elif (
            row["late_night_ratio"]
            == cluster_profile["late_night_ratio"].max()
        ):
            label = "Late-Night Spender"
        elif (
            row["avg_spend"]
            == cluster_profile["avg_spend"].max()
        ):
            label = "High-Value Spender"
        else:
            label = "Balanced Spender"

        labels[cluster] = label

    return labels


def cluster_users(
    df: pd.DataFrame,
    n_clusters: int = 4,
):
    """
    Fit K-Means on user behavior and return user profiles,
    cluster profiles, scaler, and fitted model.
    """
    users = build_user_features(df)

    if users.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            None,
            None,
        )

    usable = users.copy()

    # K-Means needs at least n_clusters users.
    k = min(
        max(1, n_clusters),
        len(usable),
    )

    scaler = StandardScaler()

    X = usable[FEATURES].fillna(0.0)

    X_scaled = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20,
    )

    usable["cluster"] = model.fit_predict(X_scaled)

    cluster_profile = (
        usable.groupby("cluster")[FEATURES]
        .mean()
        .round(3)
    )

    cluster_profile["users"] = (
        usable.groupby("cluster")
        .size()
    )

    labels = label_clusters(
        cluster_profile
    )

    usable["user_type"] = (
        usable["cluster"].map(labels)
    )

    return (
        usable,
        cluster_profile,
        scaler,
        model,
    )


def get_user_profile(
    df: pd.DataFrame,
    user_id: int,
) -> str:
    """Return the cluster/profile label for a single user."""
    users, _, _, _ = cluster_users(df)

    if users.empty:
        return "New User"

    result = users[
        users["user_id"] == user_id
    ]

    if result.empty:
        return "New User"

    return str(
        result.iloc[0]["user_type"]
    )


def save_clusters(
    df: pd.DataFrame,
    user_output: str = "user_clusters.csv",
    profile_output: str = "cluster_profiles.csv",
):
    """Cluster users and save both output files."""
    users, profile, _, _ = cluster_users(df)

    users.to_csv(
        user_output,
        index=False,
    )

    profile.to_csv(
        profile_output,
    )

    return users, profile


if __name__ == "__main__":
    df = pd.read_csv(
        "processed_transactions.csv"
    )

    users, profile = save_clusters(df)

    print(
        "\n=============================================="
    )
    print(
        "       USER BEHAVIOR CLUSTERING"
    )
    print(
        "=============================================="
    )

    print(
        f"Transactions loaded: {len(df):,}"
    )
    print(
        f"Users: {df['user_id'].nunique():,}"
    )

    if not profile.empty:
        print(
            "\n========== CLUSTER PROFILES =========="
        )
        print(profile)

        print(
            "\n========== CLUSTER LABELS =========="
        )

        for cluster, label in (
            users.groupby("cluster")["user_type"]
            .first()
            .items()
        ):
            count = int(
                (users["cluster"] == cluster).sum()
            )
            print(
                f"Cluster {cluster}: {label} "
                f"({count} users)"
            )

    print(
        "\nSaved:"
    )
    print(
        user_output := "user_clusters.csv"
    )
    print(
        "cluster_profiles.csv"
    )
    print(
        "=============================================="
    )
