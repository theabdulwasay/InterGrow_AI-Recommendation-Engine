from config import DATA_DIR
from src.data_processing.cleaner import clean_ratings
from src.data_processing.loader import load_ratings


def main() -> None:
    ratings = clean_ratings(load_ratings(DATA_DIR / "raw" / "ratings.csv"))
    global_mean = float(ratings["Rating"].mean())
    baseline_mae = float((ratings["Rating"] - global_mean).abs().mean())
    baseline_rmse = float(((ratings["Rating"] - global_mean) ** 2).mean() ** 0.5)
    print("Global-mean rating baseline (reference only; not a held-out model benchmark)")
    print(f"Ratings: {len(ratings):,}")
    print(f"MAE: {baseline_mae:.4f}")
    print(f"RMSE: {baseline_rmse:.4f}")
    print("For ranking metrics, use src.evaluation.metrics with a held-out per-user set.")


if __name__ == "__main__":
    main()
