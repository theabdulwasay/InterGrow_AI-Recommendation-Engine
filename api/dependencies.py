from functools import lru_cache

from config import HISTORY_DB_PATH
from src.history.database import initialize_database
from src.recommender.engine import RecommendationEngine


@lru_cache(maxsize=1)
def get_engine() -> RecommendationEngine:
    return RecommendationEngine()


def initialize_service() -> None:
    initialize_database(HISTORY_DB_PATH)
