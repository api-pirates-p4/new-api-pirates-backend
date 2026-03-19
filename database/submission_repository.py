"""
database/submission_repository.py
RESPONSIBILITY: One thing only — read and write prescreener_submissions rows.
"""

import json
import sqlite3
from database.db import get_connection


class SubmissionNotFound(Exception):
    pass

class SubmissionWriteError(Exception):
    pass


def save_submission(answers: dict, prediction: dict) -> int:
    try:
        with get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO prescreener_submissions
                    (need_type, location, vet_status, employment,
                     housing_risk, household_sz, has_va_care,
                     pvo_direct, refer_out, confidence, top_factors)
                VALUES
                    (:need_type, :location, :vet_status, :employment,
                     :housing_risk, :household_sz, :has_va_care,
                     :pvo_direct, :refer_out, :confidence, :top_factors)
                """,
                {
                    'need_type':    answers['need_type'],
                    'location':     answers['location'],
                    'vet_status':   answers['vet_status'],
                    'employment':   answers['employment'],
                    'housing_risk': int(answers.get('housing_risk', 0)),
                    'household_sz': int(answers.get('household_sz', 1)),
                    'has_va_care':  int(answers.get('has_va_care', 0)),
                    'pvo_direct':   prediction['pvo_direct'],
                    'refer_out':    prediction['refer_out'],
                    'confidence':   prediction['confidence'],
                    'top_factors':  json.dumps(prediction['top_factors']),
                }
            )
            conn.commit()
            return cursor.lastrowid
    except Exception as exc:
        raise SubmissionWriteError(f"Failed to save submission: {exc}") from exc


def get_submission_by_id(submission_id: int) -> dict:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM prescreener_submissions WHERE id = ?",
            (submission_id,)
        ).fetchone()

    if row is None:
        raise SubmissionNotFound(f"No submission with id={submission_id}")

    return _deserialize_row(row)


def get_recent_submissions(limit: int = 50) -> list:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM prescreener_submissions
            ORDER BY submitted_at DESC
            LIMIT ?
            """,
            (limit,)
        ).fetchall()

    return [_deserialize_row(r) for r in rows]


def get_submission_stats() -> dict:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*)                             AS total,
                ROUND(AVG(pvo_direct) * 100, 1)      AS avg_pvo_direct_pct,
                SUM(CASE WHEN pvo_direct >= 0.5
                         THEN 1 ELSE 0 END)          AS likely_direct_count,
                SUM(CASE WHEN refer_out  >= 0.5
                         THEN 1 ELSE 0 END)          AS likely_refer_count
            FROM prescreener_submissions
            """
        ).fetchone()

        need_rows = conn.execute(
            """
            SELECT need_type, COUNT(*) AS cnt
            FROM   prescreener_submissions
            GROUP  BY need_type
            ORDER  BY cnt DESC
            LIMIT  5
            """
        ).fetchall()

        loc_rows = conn.execute(
            """
            SELECT location, COUNT(*) AS cnt
            FROM   prescreener_submissions
            GROUP  BY location
            ORDER  BY cnt DESC
            LIMIT  5
            """
        ).fetchall()

    return {
        'total':               row['total'],
        'avg_pvo_direct_pct':  row['avg_pvo_direct_pct'],
        'likely_direct_count': row['likely_direct_count'],
        'likely_refer_count':  row['likely_refer_count'],
        'top_need_types':      [dict(r) for r in need_rows],
        'top_locations':       [dict(r) for r in loc_rows],
    }


def _deserialize_row(row: sqlite3.Row) -> dict:
    d = dict(row)
    d['top_factors'] = json.loads(d['top_factors'])
    return d