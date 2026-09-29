import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


def build_content_features(
    movies: pd.DataFrame,
) -> tuple[TfidfVectorizer, csr_matrix, list[int]]:
    ordered = movies.sort_values("MovieID").reset_index(drop=True)
    text = (ordered["Title"].fillna("") + " " + ordered["Genres"].fillna("")).tolist()
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    features = vectorizer.fit_transform(text)
    return vectorizer, features, ordered["MovieID"].astype(int).tolist()


def build_content_similarity(features: csr_matrix) -> csr_matrix:
    return cosine_similarity(features, dense_output=False).tocsr()
