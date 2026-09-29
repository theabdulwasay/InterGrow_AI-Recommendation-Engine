from io import BytesIO
from pathlib import Path
import os
import ssl
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd
import truststore

from config import RAW_DATA_DIR


DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
GENRES = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
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
    try:
        tls_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ca_bundle = os.getenv("SSL_CERT_FILE") or os.getenv("REQUESTS_CA_BUNDLE")
        if ca_bundle:
            tls_context.load_verify_locations(cafile=ca_bundle)
        with urlopen(DATASET_URL, timeout=60, context=tls_context) as response:
            archive_bytes = response.read()
    except URLError as error:
        raise RuntimeError(
            "MovieLens download failed. Check internet access and trusted TLS certificates."
        ) from error

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
            dtype={"ZipCode": str},
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
