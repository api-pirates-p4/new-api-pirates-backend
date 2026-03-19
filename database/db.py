"""
database/db.py
RESPONSIBILITY: One thing only — initialize and provide a connection to the SQLite3 database.
Does NOT know about veterans, predictions, or business logic.
"""

import sqlite3
import os

DB_PATH = os.environ.get('PVO_DB_PATH', 'instance/pvo.db')


def get_connection() -> sqlite3.Connection:
    """
    Returns a sqlite3 connection with row_factory set so rows behave like dicts.
    Caller is responsible for closing the connection (use as context manager).
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn