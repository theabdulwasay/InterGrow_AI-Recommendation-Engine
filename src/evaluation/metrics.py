import math


def precision_at_k(relevant: set[int], ranked: list[int], k: int) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    return sum(item in relevant for item in ranked[:k]) / k


def recall_at_k(relevant: set[int], ranked: list[int], k: int) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    if not relevant:
        return 0.0
    return sum(item in relevant for item in ranked[:k]) / len(relevant)


def ndcg_at_k(relevant: set[int], ranked: list[int], k: int) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    dcg = sum(
        1 / math.log2(rank + 2)
        for rank, item in enumerate(ranked[:k])
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / ideal_dcg if ideal_dcg else 0.0
