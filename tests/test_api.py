from fastapi.testclient import TestClient

import api.routes.recommend as recommend_route
from api.main import app


class FakeEngine:
    def recommend(self, user_id: int, limit: int):
        return [
            {"movie_id": 9, "title": "Test Movie", "genres": "Drama", "score": 0.75}
        ][:limit]


def test_recommend_endpoint_returns_expected_shape(monkeypatch, tmp_path):
    monkeypatch.setattr(recommend_route, "get_engine", lambda: FakeEngine())
    monkeypatch.setattr(recommend_route, "HISTORY_DB_PATH", tmp_path / "history.db")
    response = TestClient(app).get("/recommend/42?limit=1")
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == 42
    assert body["recommendations"][0]["movie_id"] == 9


def test_recommend_endpoint_reports_missing_dataset(monkeypatch):
    def missing_engine():
        raise FileNotFoundError("Run the dataset setup script")

    monkeypatch.setattr(recommend_route, "get_engine", missing_engine)
    response = TestClient(app).get("/recommend/42")
    assert response.status_code == 503
    assert "dataset setup" in response.json()["detail"]
