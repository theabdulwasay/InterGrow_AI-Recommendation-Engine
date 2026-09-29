import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from src.models.collaborative_utils import center_ratings


class UserCollaborativeRecommender:
    def __init__(
        self, matrix: csr_matrix, user_ids: list[int], item_ids: list[int]
    ) -> None:
        self.matrix = matrix
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.user_positions = {user_id: i for i, user_id in enumerate(user_ids)}
        self.item_positions = {item_id: i for i, item_id in enumerate(item_ids)}
        self.centered, self.means = center_ratings(matrix)
        self.similarity = cosine_similarity(self.centered, dense_output=True)

    def score_for_user(self, user_id: int, exclude: set[int]) -> dict[int, float]:
        user_position = self.user_positions.get(user_id)
        if user_position is None:
            return {}
        similarities = self.similarity[user_position].copy()
        similarities[user_position] = 0
        scores: dict[int, float] = {}
        for item_position, item_id in enumerate(self.item_ids):
            if item_id in exclude:
                continue
            candidate_users = self.matrix.getcol(item_position).tocoo()
            if candidate_users.nnz == 0:
                continue
            neighbor_positions = candidate_users.row
            weights = similarities[neighbor_positions]
            denominator = float(np.abs(weights).sum())
            if denominator == 0:
                continue
            centered_ratings = candidate_users.data - self.means[neighbor_positions]
            prediction = self.means[user_position] + float(
                np.dot(weights, centered_ratings) / denominator
            )
            scores[item_id] = max(0.0, min(1.0, (prediction - 0.5) / 4.5))
        return scores
