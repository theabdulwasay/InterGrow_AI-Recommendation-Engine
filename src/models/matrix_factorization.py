import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from src.models.collaborative_utils import center_ratings


class MatrixFactorizationRecommender:
    def __init__(
        self,
        matrix: csr_matrix,
        user_ids: list[int],
        item_ids: list[int],
        factors: int = 50,
        random_state: int = 42,
    ) -> None:
        if factors < 1:
            raise ValueError("factors must be at least 1")
        self.item_ids = item_ids
        self.user_positions = {user_id: index for index, user_id in enumerate(user_ids)}
        centered, self.means = center_ratings(matrix)
        component_count = min(factors, min(matrix.shape) - 1)
        if component_count <= 0:
            self.predictions = np.empty(matrix.shape, dtype=float)
        else:
            model = TruncatedSVD(n_components=component_count, random_state=random_state)
            user_factors = model.fit_transform(centered)
            self.predictions = user_factors @ model.components_

    def score_for_user(self, user_id: int, exclude: set[int]) -> dict[int, float]:
        user_position = self.user_positions.get(user_id)
        if user_position is None:
            return {}
        estimates = self.predictions[user_position] + self.means[user_position]
        return {
            item_id: float(max(0.0, min(1.0, (estimates[index] - 0.5) / 4.5)))
            for index, item_id in enumerate(self.item_ids)
            if item_id not in exclude
        }
