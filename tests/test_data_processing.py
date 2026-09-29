import pandas as pd

from src.data_processing.cleaner import clean_movies, clean_ratings
from src.data_processing.matrix_builder import build_user_item_matrix


def test_clean_movies_drops_missing_and_duplicate_ids():
    movies = pd.DataFrame(
        {
            "MovieID": [1, 1, None],
            "Title": ["First", "Duplicate", "Missing"],
            "Genres": ["Drama", "Comedy", ""],
        }
    )
    result = clean_movies(movies)
    assert result[["MovieID", "Title"]].to_dict("records") == [
        {"MovieID": 1, "Title": "First"}
    ]


def test_clean_ratings_and_build_sparse_matrix():
    ratings = pd.DataFrame(
        {
            "UserID": [2, 1, 1, 0],
            "MovieID": [5, 6, 5, 5],
            "Rating": [4, 2, 3, 5],
        }
    )
    cleaned = clean_ratings(ratings)
    matrix, users, items = build_user_item_matrix(cleaned)
    assert users == [1, 2]
    assert items == [5, 6]
    assert matrix.toarray().tolist() == [[3.0, 2.0], [4.0, 0.0]]
