from config import DATA_DIR
from src.data_processing.cleaner import clean_movies, clean_ratings
from src.data_processing.loader import load_movies, load_ratings, load_users
from src.evaluation.compare_models import compare_models


def main() -> None:
    movies = clean_movies(load_movies(DATA_DIR / "raw" / "movies.csv"))
    ratings = clean_ratings(load_ratings(DATA_DIR / "raw" / "ratings.csv"))
    users = load_users(DATA_DIR / "raw" / "users.csv")
    results = compare_models(movies, ratings, users)
    print(f"{'Model':<16} {'Precision@10':>14} {'Recall@10':>12} {'NDCG@10':>10}")
    for model, scores in results.items():
        print(
            f"{model:<16} {scores['precision']:>14.4f} "
            f"{scores['recall']:>12.4f} {scores['ndcg']:>10.4f}"
        )


if __name__ == "__main__":
    main()
