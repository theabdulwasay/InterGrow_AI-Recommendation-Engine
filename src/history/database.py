import sqlite3
from contextlib import closing
from pathlib import Path


def initialize_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS recommendation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    recommendations TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_history_user_created "
                "ON recommendation_history(user_id, created_at DESC)"
            )


def connect(path: Path) -> sqlite3.Connection:
    initialize_database(path)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection
