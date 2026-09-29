from config import DATA_DIR
from src.data_processing.cleaner import clean_ratings
from src.data_processing.loader import load_ratings
from src.evaluation.metrics import mae, rmse


def main() -> None:
    ratings = clean_ratings(load_ratings(DATA_DIR / "raw" / "ratings.csv"))
    global_mean = float(ratings["Rating"].mean())
    actual = ratings["Rating"].astype(float).tolist()
    predicted = [global_mean] * len(actual)
    baseline_mae = mae(actual, predicted)
    baseline_rmse = rmse(actual, predicted)
    print("Global-mean rating baseline (reference only; not a held-out model benchmark)")
    print(f"Ratings: {len(ratings):,}")
    print(f"MAE: {baseline_mae:.4f}")
    print(f"RMSE: {baseline_rmse:.4f}")
    print("For ranking metrics, use src.evaluation.metrics with a held-out per-user set.")


if __name__ == "__main__":
    main()
