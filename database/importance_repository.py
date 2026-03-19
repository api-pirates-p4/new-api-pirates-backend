"""
database/importance_repository.py
RESPONSIBILITY: One thing only — read and write feature_importance_log rows.
"""

from database.db import get_connection


def log_feature_importances(importances: dict) -> None:
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO feature_importance_log (feature_name, importance)
            VALUES (:feature_name, :importance)
            """,
            [{'feature_name': k, 'importance': v} for k, v in importances.items()]
        )
        conn.commit()


def get_latest_feature_importances() -> list:
    with get_connection() as conn:
        latest = conn.execute(
            "SELECT MAX(logged_at) AS ts FROM feature_importance_log"
        ).fetchone()['ts']

        if latest is None:
            return []

        rows = conn.execute(
            """
            SELECT feature_name, importance
            FROM   feature_importance_log
            WHERE  logged_at = ?
            ORDER  BY importance DESC
            """,
            (latest,)
        ).fetchall()

    return [dict(r) for r in rows]