from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

from config import RAW_DATA_DIR


DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
GENRES = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def download_and_prepare(destination: Path = RAW_DATA_DIR) -> tuple[Path, Path, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MovieLens 100K from {DATASET_URL} ...")
    with urlopen(DATASET_URL, timeout=60) as response:
        archive_bytes = response.read()

    with ZipFile(BytesIO(archive_bytes)) as archive:
        movies = pd.read_csv(
            archive.open("ml-100k/u.item"),
            sep="|",
            header=None,
            encoding="latin-1",
            names=["MovieID", "Title", "ReleaseDate", "VideoReleaseDate", "IMDbURL", *GENRES],
        )
        movies["Genres"] = movies[GENRES].apply(
            lambda row: "|".join(genre for genre in GENRES if row[genre] == 1), axis=1
        )
        movies = movies[["MovieID", "Title", "Genres"]]

        ratings = pd.read_csv(
            archive.open("ml-100k/u.data"),
            sep="\t",
            header=None,
            names=["UserID", "MovieID", "Rating", "Timestamp"],
        )
        users = pd.read_csv(
            archive.open("ml-100k/u.user"),
            sep="|",
            header=None,
            names=["UserID", "Age", "Gender", "Occupation", "ZipCode"],
        )

    paths = (
        destination / "movies.csv",
        destination / "ratings.csv",
        destination / "users.csv",
    )
    for frame, path in zip((movies, ratings, users), paths):
        frame.to_csv(path, index=False)
    print(
        f"Saved {len(movies):,} movies, {len(ratings):,} ratings, "
        f"and {len(users):,} users to {destination}"
    )
    return paths


if __name__ == "__main__":
    download_and_prepare()
