from typing import Protocol

from src.models.collaborative_utils import normalize_scores


class CollaborativeModel(Protocol):
    def score_for_user(self, user_id: int, exclude: set[int]) -> dict[int, float]: ...


class CollaborativeEnsemble:
    def __init__(self, models: list[CollaborativeModel]) -> None:
        if not models:
            raise ValueError("At least one collaborative model is required")
        self.models = models

    def score_for_user(self, user_id: int, exclude: set[int]) -> dict[int, float]:
        totals: dict[int, float] = {}
        counts: dict[int, int] = {}
        for model in self.models:
            for item_id, score in normalize_scores(
                model.score_for_user(user_id, exclude)
            ).items():
                totals[item_id] = totals.get(item_id, 0.0) + score
                counts[item_id] = counts.get(item_id, 0) + 1
        return {
            item_id: totals[item_id] / counts[item_id]
            for item_id in totals
        }
