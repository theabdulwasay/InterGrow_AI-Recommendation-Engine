import pandas as pd

from src.evaluation.compare_models import compare_models


def test_compare_models_returns_held_out_ranking_metrics():
    movies = pd.DataFrame(
        {
            "MovieID": [1, 2, 3, 4, 5],
            "Title": ["Space One", "Space Two", "Funny Three", "Drama Four", "Action Five"],
            "Genres": ["Sci-Fi", "Sci-Fi", "Comedy", "Drama", "Action"],
        }
    )
    ratings = pd.DataFrame(
        {
            "UserID": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
            "MovieID": [1, 2, 3, 1, 3, 4, 2, 4, 5, 3, 5, 1, 4, 1, 2],
            "Rating": [5, 4, 2, 4, 5, 3, 5, 4, 3, 4, 5, 2, 5, 4, 3],
        }
    )
    users = pd.DataFrame(
        {
            "UserID": [1, 2, 3, 4, 5],
            "Age": [20, 21, 22, 23, 24],
            "Gender": ["F", "M", "F", "M", "F"],
            "Occupation": ["student"] * 5,
            "ZipCode": ["1", "2", "3", "4", "5"],
        }
    )

    results = compare_models(movies, ratings, users, k=2)

    assert set(results) == {"content", "collaborative", "hybrid"}
    assert all(set(metrics) == {"precision", "recall", "ndcg"} for metrics in results.values())
    assert all(0.0 <= score <= 1.0 for metrics in results.values() for score in metrics.values())
