import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def segment_users(
    ratings: pd.DataFrame, clusters: int = 3, random_state: int = 42
) -> pd.DataFrame:
    if clusters < 1:
        raise ValueError("clusters must be at least 1")
    summary = ratings.groupby("UserID").agg(
        rating_count=("Rating", "size"),
        average_rating=("Rating", "mean"),
    )
    if summary.empty:
        return summary.assign(segment=pd.Series(dtype="int64"))
    cluster_count = min(clusters, len(summary))
    features = StandardScaler().fit_transform(summary)
    labels = KMeans(n_clusters=cluster_count, random_state=random_state, n_init=10).fit_predict(
        features
    )
    summary["segment"] = labels
    return summary
