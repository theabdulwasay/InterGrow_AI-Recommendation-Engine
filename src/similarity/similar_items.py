from src.models.content_based import ContentBasedRecommender


def find_similar_items(
    model: ContentBasedRecommender, movie_id: int, limit: int = 10
) -> list[tuple[int, float]]:
    return model.similar_items(movie_id, limit)
