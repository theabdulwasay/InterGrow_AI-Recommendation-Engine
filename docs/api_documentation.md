# API documentation

Start the server with `python run.py`. Interactive OpenAPI documentation is available at `/docs`; the OpenAPI schema is at `/openapi.json`.

## Responses

- `GET /health` returns `{"status": "ok", "dataset_ready": true}` once the MovieLens CSVs can be loaded. Before setup, dataset readiness is false.
- `GET /recommend/{user_id}?limit=10` returns `user_id` and a `recommendations` array of `{movie_id, title, genres, score}`. Requested results are logged to SQLite. Unknown users get the popularity-based cold-start list.
- `GET /similar/{movie_id}?limit=10` returns `movie_id` and a `similar_items` array with the same item fields.
- `GET /profile/{user_id}` returns rating count, average rating, activity level, favorite genres, and the available MovieLens demographic fields.
- `GET /history/{user_id}?limit=20` returns recent logged recommendation lists.
- `DELETE /history/{user_id}` deletes only the selected user's history and returns the number of deleted entries.

`limit` is validated to the inclusive range 1–100. Recommendation routes return 503 if dataset files are missing; unknown profile or similar-item IDs return 404.
