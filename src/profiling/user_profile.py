import pandas as pd


def build_user_profile(
    user_id: int, users: pd.DataFrame, ratings: pd.DataFrame, movies: pd.DataFrame
) -> dict[str, object]:
    user_row = users.loc[users["UserID"] == user_id]
    user_ratings = ratings.loc[ratings["UserID"] == user_id]
    if user_row.empty and user_ratings.empty:
        raise KeyError(f"User {user_id} was not found")

    profile: dict[str, object] = {
        "user_id": user_id,
        "rating_count": int(len(user_ratings)),
        "average_rating": (
            round(float(user_ratings["Rating"].mean()), 2)
            if not user_ratings.empty
            else None
        ),
        "activity_level": (
            "high"
            if len(user_ratings) >= 100
            else "medium"
            if len(user_ratings) >= 20
            else "low"
        ),
        "favorite_genres": [],
    }
    if not user_row.empty:
        user = user_row.iloc[0]
        profile.update(
            {
                "age": int(user["Age"]) if pd.notna(user["Age"]) else None,
                "gender": str(user["Gender"]) if pd.notna(user["Gender"]) else None,
                "occupation": (
                    str(user["Occupation"])
                    if pd.notna(user["Occupation"])
                    else None
                ),
                "zipcode": (
                    str(user["ZipCode"]) if pd.notna(user["ZipCode"]) else None
                ),
            }
        )

    rated_movies = user_ratings.merge(movies[["MovieID", "Genres"]], on="MovieID")
    positive = rated_movies.loc[rated_movies["Rating"] >= 4.0, "Genres"]
    genre_counts: dict[str, int] = {}
    for genre_list in positive.fillna(""):
        for genre in str(genre_list).split("|"):
            if genre and genre != "(no genres listed)":
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
    profile["favorite_genres"] = [
        genre for genre, _ in sorted(genre_counts.items(), key=lambda item: (-item[1], item[0]))[:5]
    ]
    return profile
