import pandas as pd


def rank_recommendations(
    scores: dict[int, float], movies: pd.DataFrame, limit: int
) -> list[dict[str, object]]:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    metadata = movies.set_index("MovieID")
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    result: list[dict[str, object]] = []
    for movie_id, score in ordered:
        if movie_id not in metadata.index:
            continue
        movie = metadata.loc[movie_id]
        result.append(
            {
                "movie_id": int(movie_id),
                "title": str(movie["Title"]),
                "genres": str(movie["Genres"]),
                "score": round(float(score), 4),
            }
        )
        if len(result) == limit:
            break
    return result
