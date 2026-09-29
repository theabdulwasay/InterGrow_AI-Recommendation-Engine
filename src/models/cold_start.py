import pandas as pd


def popularity_scores(
    ratings: pd.DataFrame, movie_ids: list[int], exclude: set[int]
) -> dict[int, float]:
    if ratings.empty:
        return {}
    statistics = ratings.groupby("MovieID")["Rating"].agg(["mean", "count"])
    global_mean = float(ratings["Rating"].mean())
    counts = statistics["count"]
    means = statistics["mean"]
    bayesian = (counts * means + 10 * global_mean) / (counts + 10)
    candidates = [(movie_id, float(bayesian.get(movie_id, 0))) for movie_id in movie_ids]
    eligible = [(movie_id, score) for movie_id, score in candidates if movie_id not in exclude]
    maximum = max((score for _, score in eligible), default=1.0)
    return {
        movie_id: score / maximum
        for movie_id, score in eligible
        if score > 0
    }
