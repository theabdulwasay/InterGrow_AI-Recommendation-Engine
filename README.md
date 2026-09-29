# AI Movie Recommendation Engine

A runnable, local-first movie recommender using the MovieLens 100K dataset. It combines TF-IDF content similarity with an ensemble of user-based CF, item-based CF, and truncated-SVD matrix factorization, falls back to a Bayesian popularity ranking for new users, exposes a FastAPI service, and includes a Streamlit interface and SQLite recommendation history.

## Features

- **Personalized recommendations:** normalized hybrid content and collaborative scores; rated movies are filtered out.
- **Cold start:** new users receive a popularity-based list.
- **Similar movies:** TF-IDF cosine similarity over titles and genres.
- **User profiles:** rating activity, average rating, and genres of highly rated movies.
- **Recommendation history:** API recommendation requests are saved per user in SQLite.
- **Evaluation:** MAE, RMSE, precision@k, recall@k, and NDCG@k; scripts include a global-mean rating baseline and a held-out content/CF/hybrid ranking comparison.
- **Tests:** preprocessing, recommendation components, history, API, and a synthetic end-to-end engine check.

The engine builds its in-memory representations when first requested; no model artifact needs to be downloaded or checked into source control. The collaborative ensemble averages the normalized scores from the available user-based, item-based, and SVD models. Content and collaborative predictions are combined using the configurable hybrid weights.

## Setup (Windows PowerShell)

From the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m scripts.download_dataset
```

The dataset script downloads the official MovieLens 100K archive from GroupLens and writes `movies.csv`, `ratings.csv`, and `users.csv` into `data\raw\`. Dataset files, trained state, and the history database are intentionally excluded from version control.

On Windows, the downloader uses the operating system's trusted certificate store. If your network uses a custom TLS inspection certificate that is not installed in the Windows trust store, set `SSL_CERT_FILE` to the PEM bundle issued by your IT administrator before running the downloader. TLS verification remains enabled.

## Run

Start the API in one terminal:

```powershell
python run.py
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation. Start the Streamlit UI in a second terminal:

```powershell
streamlit run app\streamlit_app.py
```

The API root (`http://127.0.0.1:8000/`) redirects to interactive docs. The UI uses `http://127.0.0.1:8000` by default. Set `RECOMMENDER_API_URL` in the environment to use another API address. The `.env.example` file documents this setting; the application does not automatically load `.env` files.

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
python -m scripts.evaluate_models
python -m scripts.compare_models
pytest
```

The evaluation script reports a global-mean baseline on all available ratings. The comparison script holds out one positive rating for users with enough history and compares content-only, collaborative-only, and hybrid rankings with Precision@10, Recall@10, and NDCG@10. Its results depend on the dataset and should be generated locally; no benchmark values are hard-coded here.

## Configuration and layout

Adjust hybrid weights and local data paths in [config.py](./config.py). The engine normalizes non-negative hybrid weights to sum to one. Main implementation areas:

- `src/data_processing/` — validation, cleaning, text features, and sparse user-item matrix.
- `src/models/` — content similarity, user/item collaborative filtering, SVD, popularity fallback, and hybrid scoring.
- `src/recommender/` — orchestration and result ranking.
- `src/profiling/`, `src/history/`, `src/evaluation/` — profiles, SQLite history, and metrics.
- `api/` — FastAPI routes and response schemas.
- `app/` — Streamlit UI.
- `scripts/` — dataset download, baseline evaluation, and held-out model comparison.

## Dataset and privacy

MovieLens 100K is provided by GroupLens Research. Review the [GroupLens dataset page](https://grouplens.org/datasets/movielens/100k/) and its terms before use. User demographic fields are sourced from the public dataset; the project does not collect credentials or send data to a hosted service. History is stored locally in `data/history/recommendation_history.db`.
<img width="1204" height="991" alt="image" src="https://github.com/user-attachments/assets/4621def2-733d-4f82-9479-a19f57041a7e" />
