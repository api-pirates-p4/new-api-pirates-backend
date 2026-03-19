"""
database/schema.py
RESPONSIBILITY: One thing only — create all PVO tables if they don't exist.
"""

from database.db import get_connection


def create_tables() -> None:
    ddl_statements = [
        """
        CREATE TABLE IF NOT EXISTS prescreener_submissions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            submitted_at  TEXT    NOT NULL DEFAULT (datetime('now')),
            need_type     TEXT    NOT NULL,
            location      TEXT    NOT NULL,
            vet_status    TEXT    NOT NULL,
            employment    TEXT    NOT NULL,
            housing_risk  INTEGER NOT NULL DEFAULT 0,
            household_sz  INTEGER NOT NULL DEFAULT 1,
            has_va_care   INTEGER NOT NULL DEFAULT 0,
            pvo_direct    REAL    NOT NULL,
            refer_out     REAL    NOT NULL,
            confidence    TEXT    NOT NULL,
            top_factors   TEXT    NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS feature_importance_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at     TEXT    NOT NULL DEFAULT (datetime('now')),
            feature_name  TEXT    NOT NULL,
            importance    REAL    NOT NULL
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_submissions_submitted_at
            ON prescreener_submissions(submitted_at)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_submissions_need_type
            ON prescreener_submissions(need_type)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_submissions_location
            ON prescreener_submissions(location)
        """,
    ]

    with get_connection() as conn:
        for stmt in ddl_statements:
            conn.execute(stmt)
        conn.commit()