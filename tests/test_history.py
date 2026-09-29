from src.history.history_manager import clear_history, get_history, save_history


def test_history_round_trip_and_clear(tmp_path):
    database = tmp_path / "history.db"
    recommendation = [{"movie_id": 1, "title": "Example", "genres": "Drama", "score": 0.8}]
    save_history(database, 42, recommendation)
    entries = get_history(database, 42)
    assert len(entries) == 1
    assert entries[0]["recommendations"] == recommendation
    assert clear_history(database, 42) == 1
    assert get_history(database, 42) == []
