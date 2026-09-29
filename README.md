# AI Movie Recommendation Engine

A runnable, local-first movie recommender using the MovieLens 100K dataset. It combines TF-IDF content similarity with user-based collaborative filtering, falls back to a Bayesian popularity ranking for new users, exposes a FastAPI service, and includes a Streamlit interface and SQLite recommendation history.

## Features

- **Personalized recommendations:** normalized hybrid content and collaborative scores; rated movies are filtered out.
- **Cold start:** new users receive a popularity-based list.
- **Similar movies:** TF-IDF cosine similarity over titles and genres.
- **User profiles:** rating activity, average rating, and genres of highly rated movies.
- **Recommendation history:** API recommendation requests are saved per user in SQLite.
- **Evaluation helpers:** precision@k, recall@k, and NDCG@k, plus a simple global-mean rating baseline.
- **Tests:** preprocessing, recommendation components, history, API, and a synthetic end-to-end engine check.

The engine trains its in-memory representations when first requested; no model artifact needs to be downloaded or checked into source control. The collaborative component is user-based; item similarity is content-based.

## Setup (Windows PowerShell)

From the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\download_dataset.py
```

The dataset script downloads the official MovieLens 100K archive from GroupLens and writes `movies.csv`, `ratings.csv`, and `users.csv` into `data\raw\`. Dataset files, trained state, and the history database are intentionally excluded from version control.

## Run

Start the API in one terminal:

```powershell
python run.py
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation. Start the Streamlit UI in a second terminal:

```powershell
streamlit run app\streamlit_app.py
```

The UI uses `http://127.0.0.1:8000` by default. Set `RECOMMENDER_API_URL` in the environment to use another API address. The `.env.example` file documents this setting; the application does not automatically load `.env` files.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Service and dataset readiness |
| `GET` | `/recommend/{user_id}?limit=10` | Personalized or cold-start recommendations; logs the result |
| `GET` | `/similar/{movie_id}?limit=10` | Find similar movies |
| `GET` | `/profile/{user_id}` | User rating profile |
| `GET` | `/history/{user_id}?limit=20` | Recent recommendation requests |
| `DELETE` | `/history/{user_id}` | Clear that user's history |

The recommendation endpoints accept limits from 1 to 100. Until the dataset has been downloaded, recommendation endpoints return HTTP 503 with setup instructions.

## Evaluate and test

```powershell
python scripts\evaluate_models.py
pytest
```

The evaluation script reports a global-mean baseline on all available ratings; it is explicitly **not** a held-out model comparison. Ranking metrics are provided for evaluation code that supplies per-user held-out relevance sets. No claim of measured recommendation quality is made by this baseline.

## Configuration and layout

Adjust hybrid weights and local data paths in [config.py](./config.py). The engine normalizes non-negative hybrid weights to sum to one. Main implementation areas:

- `src/data_processing/` — validation, cleaning, text features, and sparse user-item matrix.
- `src/models/` — content similarity, user-based collaborative filtering, popularity fallback, and hybrid scoring.
- `src/recommender/` — orchestration and result ranking.
- `src/profiling/`, `src/history/`, `src/evaluation/` — profiles, SQLite history, and metrics.
- `api/` — FastAPI routes and response schemas.
- `app/` — Streamlit UI.
- `scripts/` — dataset download and baseline evaluation.

## Dataset and privacy

MovieLens 100K is provided by GroupLens Research. Review the [GroupLens dataset page](https://grouplens.org/datasets/movielens/100k/) and its terms before use. User demographic fields are sourced from the public dataset; the project does not collect credentials or send data to a hosted service. History is stored locally in `data/history/recommendation_history.db`.
