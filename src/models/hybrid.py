from src.models.collaborative_user import UserCollaborativeRecommender
from src.models.content_based import ContentBasedRecommender


def _normalize(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    low = min(scores.values())
    high = max(scores.values())
    if high == low:
        return {movie_id: 1.0 for movie_id in scores}
    return {movie_id: (score - low) / (high - low) for movie_id, score in scores.items()}


class HybridRecommender:
    def __init__(
        self,
        content: ContentBasedRecommender,
        collaborative: UserCollaborativeRecommender,
        content_weight: float = 0.5,
        collaborative_weight: float = 0.5,
    ) -> None:
        if content_weight < 0 or collaborative_weight < 0:
            raise ValueError("Hybrid weights must be non-negative")
        if content_weight + collaborative_weight == 0:
            raise ValueError("At least one hybrid weight must be positive")
        total = content_weight + collaborative_weight
        self.content_weight = content_weight / total
        self.collaborative_weight = collaborative_weight / total
        self.content = content
        self.collaborative = collaborative

    def recommend(
        self,
        user_id: int,
        ratings: list[tuple[int, float]],
        seen: set[int],
    ) -> dict[int, float]:
        content_scores = _normalize(self.content.score_for_user(ratings, seen))
        collaborative_scores = _normalize(
            self.collaborative.score_for_user(user_id, seen)
        )
        candidates = set(content_scores) | set(collaborative_scores)
        return {
            movie_id: self.content_weight * content_scores.get(
                movie_id, 0.0
            )
            + self.collaborative_weight * collaborative_scores.get(movie_id, 0.0)
            for movie_id in candidates
        }
