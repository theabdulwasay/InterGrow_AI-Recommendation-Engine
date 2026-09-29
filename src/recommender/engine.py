from config import (
    COLLABORATIVE_WEIGHT,
    CONTENT_WEIGHT,
    DATA_DIR,
    DEFAULT_RECOMMENDATION_COUNT,
)
from src.data_processing.cleaner import clean_movies, clean_ratings
from src.data_processing.feature_engineering import (
    build_content_features,
    build_content_similarity,
)
from src.data_processing.loader import load_movies, load_ratings, load_users
from src.data_processing.matrix_builder import build_user_item_matrix
from src.models.cold_start import popularity_scores
from src.models.collaborative_user import UserCollaborativeRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.hybrid import HybridRecommender
from src.profiling.user_profile import build_user_profile
from src.recommender.ranking import rank_recommendations


class RecommendationEngine:
    def __init__(self, data_dir=DATA_DIR) -> None:
        self.data_dir = data_dir
        movies_path = data_dir / "raw" / "movies.csv"
        ratings_path = data_dir / "raw" / "ratings.csv"
        users_path = data_dir / "raw" / "users.csv"
        missing = [path.name for path in (movies_path, ratings_path, users_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Dataset files are missing ({', '.join(missing)}). "
                "Run `python scripts/download_dataset.py` first."
            )
        self.movies = clean_movies(load_movies(movies_path))
        self.ratings = clean_ratings(load_ratings(ratings_path))
        self.users = load_users(users_path)
        if self.movies.empty or self.ratings.empty:
            raise ValueError("The movie and rating datasets must not be empty")
        _, features, movie_ids = build_content_features(self.movies)
        similarity = build_content_similarity(features)
        self.content = ContentBasedRecommender(movie_ids, similarity)
        matrix, user_ids, item_ids = build_user_item_matrix(self.ratings)
        self.collaborative = UserCollaborativeRecommender(matrix, user_ids, item_ids)
        self.hybrid = HybridRecommender(
            self.content, self.collaborative, CONTENT_WEIGHT, COLLABORATIVE_WEIGHT
        )
    def recommend(self, user_id: int, limit: int = DEFAULT_RECOMMENDATION_COUNT):
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        user_ratings_frame = self.ratings.loc[self.ratings["UserID"] == user_id]
        seen = set(user_ratings_frame["MovieID"].astype(int).tolist())
        if user_ratings_frame.empty:
            scores = popularity_scores(self.ratings, self.content.movie_ids, seen)
        else:
            ratings = list(
                zip(
                    user_ratings_frame["MovieID"].astype(int).tolist(),
                    user_ratings_frame["Rating"].astype(float).tolist(),
                )
            )
            scores = self.hybrid.recommend(user_id, ratings, seen)
            if not scores:
                scores = popularity_scores(self.ratings, self.content.movie_ids, seen)
        return rank_recommendations(scores, self.movies, limit)

    def similar(self, movie_id: int, limit: int = 10):
        if movie_id not in self.content.positions:
            raise KeyError(f"Movie {movie_id} was not found")
        return rank_recommendations(
            dict(self.content.similar_items(movie_id, limit)), self.movies, limit
        )

    def profile(self, user_id: int):
        return build_user_profile(user_id, self.users, self.ratings, self.movies)
