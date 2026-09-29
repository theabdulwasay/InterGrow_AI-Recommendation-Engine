import pandas as pd

from src.recommender.engine import RecommendationEngine


def test_engine_recommends_unseen_movies_and_serves_cold_start(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame(
        {
            "MovieID": [1, 2, 3, 4],
            "Title": ["Space Journey", "Space Mission", "Funny Friends", "Ocean Story"],
            "Genres": ["Sci-Fi|Adventure", "Sci-Fi|Adventure", "Comedy", "Drama"],
        }
    ).to_csv(raw / "movies.csv", index=False)
    pd.DataFrame(
        {
            "UserID": [1, 1, 2, 2, 3, 3],
            "MovieID": [1, 2, 1, 3, 3, 4],
            "Rating": [5, 4, 4, 5, 4, 3],
            "Timestamp": [0] * 6,
        }
    ).to_csv(raw / "ratings.csv", index=False)
    pd.DataFrame(
        {
            "UserID": [1, 2, 3],
            "Age": [25, 30, 35],
            "Gender": ["M", "F", "M"],
            "Occupation": ["student", "artist", "teacher"],
            "ZipCode": ["1", "2", "3"],
        }
    ).to_csv(raw / "users.csv", index=False)

    engine = RecommendationEngine(tmp_path)
    personalized = engine.recommend(1, 3)
    cold_start = engine.recommend(999, 3)

    assert personalized
    assert all(movie["movie_id"] not in {1, 2} for movie in personalized)
    assert cold_start
    assert len(engine.similar(1, 2)) > 0
    assert engine.profile(1)["rating_count"] == 2
