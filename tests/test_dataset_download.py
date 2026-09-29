from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

import scripts.download_dataset as downloader


def test_download_and_prepare_converts_movielens_files(tmp_path, monkeypatch):
    archive_buffer = BytesIO()
    movie_fields = ["1", "Toy Story (1995)", "01-Jan-1995", "", "https://example.test"]
    movie_fields.extend(
        "1" if genre == "Comedy" else "0" for genre in downloader.GENRES
    )
    with ZipFile(archive_buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("ml-100k/u.item", "|".join(movie_fields) + "\n")
        archive.writestr("ml-100k/u.data", "1\t1\t5\t874965758\n")
        archive.writestr("ml-100k/u.user", "1|24|M|artist|12345\n")
    archive_buffer.seek(0)
    monkeypatch.setattr(downloader, "urlopen", lambda *_args, **_kwargs: archive_buffer)

    movie_path, ratings_path, users_path = downloader.download_and_prepare(tmp_path)

    movies = pd.read_csv(movie_path)
    ratings = pd.read_csv(ratings_path)
    users = pd.read_csv(users_path)
    assert movies.loc[0, "Genres"] == "Comedy"
    assert ratings.loc[0, ["UserID", "MovieID", "Rating"]].tolist() == [1, 1, 5]
    assert users.loc[0, ["UserID", "Occupation"]].tolist() == [1, "artist"]
