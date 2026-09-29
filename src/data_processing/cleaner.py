import pandas as pd


def clean_movies(movies: pd.DataFrame) -> pd.DataFrame:
    cleaned = movies.dropna(subset=["MovieID", "Title"]).copy()
    cleaned["MovieID"] = pd.to_numeric(cleaned["MovieID"], errors="coerce")
    cleaned = cleaned.dropna(subset=["MovieID"])
    cleaned["MovieID"] = cleaned["MovieID"].astype(int)
    cleaned["Title"] = cleaned["Title"].astype(str).str.strip()
    cleaned["Genres"] = cleaned.get("Genres", "").fillna("").astype(str)
    return cleaned.drop_duplicates("MovieID").reset_index(drop=True)


def clean_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    cleaned = ratings.dropna(subset=["UserID", "MovieID", "Rating"]).copy()
    for column in ("UserID", "MovieID", "Rating"):
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.dropna(subset=["UserID", "MovieID", "Rating"])
    cleaned = cleaned[
        cleaned["Rating"].between(0.5, 5.0)
        & (cleaned["UserID"] > 0)
        & (cleaned["MovieID"] > 0)
    ]
    cleaned["UserID"] = cleaned["UserID"].astype(int)
    cleaned["MovieID"] = cleaned["MovieID"].astype(int)
    return cleaned.drop_duplicates(["UserID", "MovieID"], keep="last").reset_index(
        drop=True
    )
