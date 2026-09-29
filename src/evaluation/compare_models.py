from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k
from src.recommender.engine import RecommendationEngine


def compare_models(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    users: pd.DataFrame,
    k: int = 10,
) -> dict[str, dict[str, float]]:
    if k < 1:
        raise ValueError("k must be at least 1")
    train_parts: list[pd.DataFrame] = []
    test_rows: list[pd.Series] = []
    for _, group in ratings.groupby("UserID", sort=True):
        positive = group.loc[group["Rating"] >= 4.0]
        if len(group) < 2 or positive.empty:
            train_parts.append(group)
            continue
        held_out = positive.sample(n=1, random_state=42)
        test_rows.append(held_out.iloc[0])
        train_parts.append(group.drop(index=held_out.index))
    if not test_rows:
        raise ValueError("No users have enough ratings for a held-out evaluation")
    training = pd.concat(train_parts, ignore_index=True)

    with TemporaryDirectory(prefix="movie-recommender-eval-") as temporary:
        data_dir = Path(temporary)
        raw_dir = data_dir / "raw"
        raw_dir.mkdir()
        movies.to_csv(raw_dir / "movies.csv", index=False)
        training.to_csv(raw_dir / "ratings.csv", index=False)
        users.to_csv(raw_dir / "users.csv", index=False)
        engine = RecommendationEngine(data_dir=data_dir)

        totals = {
            name: {"precision": 0.0, "recall": 0.0, "ndcg": 0.0}
            for name in ("content", "collaborative", "hybrid")
        }
        for held_out in test_rows:
            user_id = int(held_out["UserID"])
            user_training = training.loc[training["UserID"] == user_id]
            rated = list(
                zip(
                    user_training["MovieID"].astype(int).tolist(),
                    user_training["Rating"].astype(float).tolist(),
                )
            )
            seen = {movie_id for movie_id, _ in rated}
            rankings = {
                "content": engine.content.score_for_user(rated, seen),
                "collaborative": engine.collaborative.score_for_user(user_id, seen),
                "hybrid": engine.hybrid.recommend(user_id, rated, seen),
            }
            relevant = {int(held_out["MovieID"])}
            for name, scores in rankings.items():
                ranked_ids = [
                    movie_id
                    for movie_id, _ in sorted(
                        scores.items(), key=lambda item: (-item[1], item[0])
                    )[:k]
                ]
                totals[name]["precision"] += precision_at_k(relevant, ranked_ids, k)
                totals[name]["recall"] += recall_at_k(relevant, ranked_ids, k)
                totals[name]["ndcg"] += ndcg_at_k(relevant, ranked_ids, k)

    count = len(test_rows)
    return {
        name: {metric: value / count for metric, value in values.items()}
        for name, values in totals.items()
    }
