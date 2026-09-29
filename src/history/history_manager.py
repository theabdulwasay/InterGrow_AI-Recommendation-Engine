import json
from datetime import datetime, timezone
from pathlib import Path

from src.history.database import connect


def save_history(path: Path, user_id: int, recommendations: list[dict]) -> None:
    created_at = datetime.now(timezone.utc).isoformat()
    with connect(path) as connection:
        connection.execute(
            "INSERT INTO recommendation_history (user_id, created_at, recommendations) "
            "VALUES (?, ?, ?)",
            (user_id, created_at, json.dumps(recommendations)),
        )


def get_history(path: Path, user_id: int, limit: int = 20) -> list[dict[str, object]]:
    if limit < 1 or limit > 100:
        raise ValueError("limit must be between 1 and 100")
    with connect(path) as connection:
        rows = connection.execute(
            "SELECT id, user_id, created_at, recommendations "
            "FROM recommendation_history WHERE user_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [
        {
            "id": int(row["id"]),
            "user_id": int(row["user_id"]),
            "created_at": row["created_at"],
            "recommendations": json.loads(row["recommendations"]),
        }
        for row in rows
    ]


def clear_history(path: Path, user_id: int) -> int:
    with connect(path) as connection:
        cursor = connection.execute(
            "DELETE FROM recommendation_history WHERE user_id = ?", (user_id,)
        )
        return int(cursor.rowcount)
