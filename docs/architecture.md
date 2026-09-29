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
   |                    |
 content scores      user-neighbor scores
          \            /
        weighted hybrid ranking
                 |
 FastAPI routes ---- SQLite history
        ^
        |
 Streamlit UI (HTTP client)
```

The UI communicates with the API rather than loading models independently. At process lifetime, the API caches a single engine instance, which builds cleaned data, TF-IDF features, item similarities, and user-user similarities in memory. A score-only Bayesian popularity ranking serves users with no known ratings. History stores each requested ranked list with a UTC timestamp.
