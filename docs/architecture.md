# Architecture

```text
MovieLens 100K archive
        |
 scripts/download_dataset.py
        v
 data/raw/{movies,ratings,users}.csv
        |
 RecommendationEngine (loaded on first request and cached)
   |                    |
 TF-IDF + cosine     sparse user-item ratings
   |                 /      |       \
 content scores    user CF  item CF  SVD
                     \       |       /
                  collaborative scores
          \            /
        weighted content/CF hybrid
                 |
 FastAPI routes ---- SQLite history
        ^
        |
 Streamlit UI (HTTP client)
```

The UI communicates with the API rather than loading models independently. At process lifetime, the API caches a single engine instance, which builds cleaned data, TF-IDF features, item similarities, user-user similarities, item-user similarities, and truncated-SVD factors in memory. The collaborative ranker averages normalized user-based CF, item-based CF, and SVD predictions; the hybrid then combines it with the content ranker. A Bayesian popularity ranking serves users with no known ratings. History stores each requested ranked list with a UTC timestamp.
