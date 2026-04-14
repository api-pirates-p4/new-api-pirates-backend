import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "instance", "volunteers.db")

class VolunteerModel:
    def __init__(self, db_path=DEFAULT_DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()
    # ... rest unchanged
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS volunteers (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name   TEXT NOT NULL,
                last_name    TEXT NOT NULL,
                phone        TEXT,
                email        TEXT NOT NULL,
                city_zip     TEXT,
                roles        TEXT,
                skills       TEXT,
                availability TEXT,
                best_time    TEXT,
                organization TEXT,
                created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def create(self, data):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO volunteers
            (first_name, last_name, phone, email, city_zip, roles, skills, availability, best_time, organization)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get("first_name"),
            data.get("last_name"),
            data.get("phone"),
            data.get("email"),
            data.get("city_zip"),
            data.get("roles"),
            data.get("skills"),
            data.get("availability"),
            data.get("best_time"),
            data.get("organization"),
        ))
        conn.commit()
        conn.close()

    def read(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM volunteers ORDER BY created_at DESC"
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def delete(self, volunteer_id):
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM volunteers WHERE id = ?", (volunteer_id,))
        conn.commit()
        conn.close()

from flask import Blueprint, request, jsonify

volunteer_api = Blueprint('volunteer_api', __name__)
volunteer_model = VolunteerModel()

@volunteer_api.route('/api/volunteers', methods=['POST'])
def submit_volunteer():
    data = request.get_json()
    if not data or not data.get('first_name') or not data.get('email'):
        return jsonify({'error': 'first_name and email are required'}), 400
    volunteer_model.create(data)
    return jsonify({'message': 'Volunteer application submitted!'}), 201

@volunteer_api.route('/api/volunteers', methods=['GET'])
def get_volunteers():
    return jsonify(volunteer_model.read())