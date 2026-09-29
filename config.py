from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
HISTORY_DB_PATH = DATA_DIR / "history" / "recommendation_history.db"
MODEL_DIR = ROOT_DIR / "models_saved"

CONTENT_WEIGHT = 0.5
COLLABORATIVE_WEIGHT = 0.5
DEFAULT_RECOMMENDATION_COUNT = 10
MIN_RATINGS_FOR_PERSONALIZATION = 2
