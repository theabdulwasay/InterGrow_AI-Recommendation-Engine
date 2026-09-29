import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from src.models.collaborative_utils import center_ratings


class ItemCollaborativeRecommender:
    def __init__(
        self, matrix: csr_matrix, user_ids: list[int], item_ids: list[int]
    ) -> None:
        self.item_ids = item_ids
        self.user_positions = {user_id: index for index, user_id in enumerate(user_ids)}
        self.centered, _ = center_ratings(matrix)
        self.similarity = cosine_similarity(self.centered.T, dense_output=False).tocsr()

    def score_for_user(self, user_id: int, exclude: set[int]) -> dict[int, float]:
        user_position = self.user_positions.get(user_id)
        if user_position is None:
            return {}
        scores = (
            self.centered.getrow(user_position) @ self.similarity
        ).toarray().ravel()
        positive_scores = {
            item_id: float(scores[index])
            for index, item_id in enumerate(self.item_ids)
            if item_id not in exclude and scores[index] > 0
        }
        return positive_scores
