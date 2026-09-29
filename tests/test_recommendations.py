import pandas as pd
from scipy.sparse import csr_matrix

from src.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k
from src.models.content_based import ContentBasedRecommender
from src.models.hybrid import HybridRecommender
from src.models.collaborative_user import UserCollaborativeRecommender
from src.profiling.user_profile import build_user_profile
from src.recommender.ranking import rank_recommendations


def test_content_similarity_and_hybrid_exclude_seen_movies():
    content = ContentBasedRecommender([10, 20, 30], csr_matrix([[1, 0.8, 0], [0.8, 1, 0], [0, 0, 1]]))
    collaborative = UserCollaborativeRecommender(
        csr_matrix([[5, 4, 0], [5, 0, 4], [0, 4, 5]]),
        [1, 2, 3],
        [10, 20, 30],
    )
    hybrid = HybridRecommender(content, collaborative)
    assert content.similar_items(10, 1) == [(20, 0.8)]
    assert 10 not in hybrid.recommend(1, [(10, 5.0), (20, 4.0)], {10, 20})


def test_profile_and_ranking_are_deterministic():
    users = pd.DataFrame(
        {"UserID": [1], "Age": [25], "Gender": ["M"], "Occupation": ["student"], "ZipCode": ["12345"]}
    )
    ratings = pd.DataFrame({"UserID": [1, 1], "MovieID": [10, 20], "Rating": [5.0, 4.0]})
    movies = pd.DataFrame(
        {"MovieID": [10, 20], "Title": ["A", "B"], "Genres": ["Drama|Comedy", "Drama"]}
    )
    profile = build_user_profile(1, users, ratings, movies)
    assert profile["favorite_genres"] == ["Drama", "Comedy"]
    assert profile["average_rating"] == 4.5
    assert rank_recommendations({20: 0.8, 10: 0.8}, movies, 1)[0]["movie_id"] == 10


def test_ranking_metrics():
    relevant = {2, 4}
    ranked = [2, 3, 4]
    assert precision_at_k(relevant, ranked, 2) == 0.5
    assert recall_at_k(relevant, ranked, 2) == 0.5
    assert 0 < ndcg_at_k(relevant, ranked, 3) < 1
