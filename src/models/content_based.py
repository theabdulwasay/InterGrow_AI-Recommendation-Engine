import numpy as np
from scipy.sparse import csr_matrix


class ContentBasedRecommender:
    def __init__(self, movie_ids: list[int], similarity: csr_matrix) -> None:
        self.movie_ids = movie_ids
        self.positions = {movie_id: index for index, movie_id in enumerate(movie_ids)}
        self.similarity = similarity

    def similar_items(self, movie_id: int, limit: int = 10) -> list[tuple[int, float]]:
        position = self.positions.get(movie_id)
        if position is None:
            return []
        scores = self.similarity.getrow(position).toarray().ravel()
        order = np.argsort(scores)[::-1]
        return [
            (self.movie_ids[index], float(scores[index]))
            for index in order
            if index != position and scores[index] > 0
        ][:limit]

    def score_for_user(
        self, ratings: list[tuple[int, float]], exclude: set[int]
    ) -> dict[int, float]:
        profile = np.zeros(len(self.movie_ids), dtype=float)
        for movie_id, rating in ratings:
            position = self.positions.get(movie_id)
            if position is not None:
                profile += (rating - 3.0) * self.similarity.getrow(position).toarray()[0]
        if not np.any(profile):
            return {}
        maximum = float(np.max(profile))
        if maximum <= 0:
            return {}
        return {
            movie_id: float(profile[index] / maximum)
            for index, movie_id in enumerate(self.movie_ids)
            if movie_id not in exclude and profile[index] > 0
        }
