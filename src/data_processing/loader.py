from pathlib import Path

import pandas as pd


def load_movies(path: Path) -> pd.DataFrame:
    movies = pd.read_csv(path)
    required = {"MovieID", "Title", "Genres"}
    missing = required.difference(movies.columns)
    if missing:
        raise ValueError(f"Movies file is missing columns: {', '.join(sorted(missing))}")
    return movies


def load_ratings(path: Path) -> pd.DataFrame:
    ratings = pd.read_csv(path)
    required = {"UserID", "MovieID", "Rating"}
    missing = required.difference(ratings.columns)
    if missing:
        raise ValueError(
            f"Ratings file is missing columns: {', '.join(sorted(missing))}"
        )
    return ratings


def load_users(path: Path) -> pd.DataFrame:
    users = pd.read_csv(path)
    if "UserID" not in users.columns:
        raise ValueError("Users file is missing the UserID column")
    return users
