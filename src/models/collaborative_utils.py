import numpy as np
from scipy.sparse import csr_matrix


def center_ratings(matrix: csr_matrix) -> tuple[csr_matrix, np.ndarray]:
    counts = np.asarray((matrix != 0).sum(axis=1)).ravel()
    means = np.asarray(matrix.sum(axis=1)).ravel() / np.maximum(counts, 1)
    centered = matrix.tolil(copy=True)
    for user_index in range(centered.shape[0]):
        if centered.rows[user_index]:
            centered.data[user_index] = [
                value - means[user_index] for value in centered.data[user_index]
            ]
    return centered.tocsr(), means


def normalize_scores(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    low = min(scores.values())
    high = max(scores.values())
    if high == low:
        return {item_id: 1.0 for item_id in scores}
    return {item_id: (score - low) / (high - low) for item_id, score in scores.items()}
