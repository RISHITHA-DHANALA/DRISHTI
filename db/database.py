"""
db/database.py
---------------
Lightweight SQLite persistence layer for the prototype.

Tables:
    users         - login accounts (health workers / doctors)
    patients      - registered patients
    screenings    - one row per fundus image screening event
    doctor_notes  - doctor's suggestions / smart-referral confirmation,
                    one row per screening (added — does not affect any
                    existing table or data)

All functions open/close their own connection (simple & safe for a
Streamlit single-process prototype; swap for a connection pool /
proper ORM before any production use).
"""

import sqlite3
import hashlib
import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH, DEMO_USERS  # noqa: E402


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _hash_password(password: str) -> str:
    """Simple SHA-256 hash. Fine for a local prototype; use bcrypt/argon2 in production."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def init_db():
    """Create tables if they don't exist and seed demo users. Never drops data."""
    conn = _connect()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('Health Worker', 'Doctor')),
            display_name TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_code TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            diabetes_duration INTEGER NOT NULL,
            previous_screening TEXT NOT NULL,
            registered_by TEXT,
            registered_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS screenings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            image_path TEXT,
            dr_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            risk_level TEXT NOT NULL,
            referral TEXT NOT NULL,
            followup_status TEXT NOT NULL DEFAULT 'Pending',
            heatmap_path TEXT,
            screened_by TEXT,
            screened_at TEXT NOT NULL,
            FOREIGN KEY(patient_id) REFERENCES patients(id)
        )
    """)

    # --- NEW: doctor suggestions / smart-referral confirmation ----------------
    # One row per screening. Added with CREATE TABLE IF NOT EXISTS so it is
    # 100% safe to run against a database that already has patients/screenings.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS doctor_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            screening_id INTEGER UNIQUE NOT NULL,
            doctor_name TEXT,
            recommendation TEXT,
            referral_override TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(screening_id) REFERENCES screenings(id)
        )
    """)

    conn.commit()

    # Seed demo login accounts if not already present.
    for username, password, role, display_name in DEMO_USERS:
        cur.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cur.fetchone() is None:
            cur.execute(
                "INSERT INTO users (username, password_hash, role, display_name) VALUES (?,?,?,?)",
                (username, _hash_password(password), role, display_name),
            )
    conn.commit()
    conn.close()


def authenticate(username: str, password: str):
    """Return the user row dict if credentials match, else None."""
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if row and row["password_hash"] == _hash_password(password):
        return dict(row)
    return None


def add_patient(patient_code, name, age, diabetes_duration, previous_screening, registered_by):
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO patients
           (patient_code, name, age, diabetes_duration, previous_screening, registered_by, registered_at)
           VALUES (?,?,?,?,?,?,?)""",
        (patient_code, name, age, diabetes_duration, previous_screening,
         registered_by, datetime.datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def get_patients():
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM patients ORDER BY registered_at DESC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_patient_by_id(patient_id):
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM patients WHERE id = ?", (patient_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def add_screening(patient_id, image_path, dr_class, confidence, risk_level,
                   referral, heatmap_path, screened_by):
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO screenings
           (patient_id, image_path, dr_class, confidence, risk_level, referral,
            followup_status, heatmap_path, screened_by, screened_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (patient_id, image_path, dr_class, confidence, risk_level, referral,
         "Pending", heatmap_path, screened_by,
         datetime.datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def get_screenings():
    """
    Return all screenings joined with patient info AND (if present) the
    doctor's suggestion / smart-referral override, newest first.
    A screening with no doctor note yet simply gets NULLs for those columns
    (LEFT JOIN), so this is fully backward compatible.
    """
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT s.*,
               p.patient_code, p.name AS patient_name, p.age, p.diabetes_duration,
               dn.doctor_name        AS reviewing_doctor,
               dn.recommendation     AS doctor_recommendation,
               dn.referral_override  AS doctor_referral_override,
               dn.updated_at         AS doctor_note_updated_at
        FROM screenings s
        JOIN patients p ON s.patient_id = p.id
        LEFT JOIN doctor_notes dn ON dn.screening_id = s.id
        ORDER BY s.screened_at DESC
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_screening_by_id(screening_id):
    """Single screening (with patient + doctor note info), used by the review panel."""
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT s.*,
               p.patient_code, p.name AS patient_name, p.age, p.diabetes_duration,
               dn.doctor_name        AS reviewing_doctor,
               dn.recommendation     AS doctor_recommendation,
               dn.referral_override  AS doctor_referral_override,
               dn.updated_at         AS doctor_note_updated_at
        FROM screenings s
        JOIN patients p ON s.patient_id = p.id
        LEFT JOIN doctor_notes dn ON dn.screening_id = s.id
        WHERE s.id = ?
    """, (screening_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def update_followup_status(screening_id, status):
    conn = _connect()
    cur = conn.cursor()
    cur.execute("UPDATE screenings SET followup_status = ? WHERE id = ?", (status, screening_id))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# NEW: Doctor suggestions / Smart Referral confirmation
# ---------------------------------------------------------------------------
def add_doctor_recommendation(screening_id, doctor_name, recommendation, referral_override=None):
    """
    Upserts the doctor's note for a screening — one screening has at most one
    note; saving again updates it in place rather than duplicating rows.
    """
    conn = _connect()
    cur = conn.cursor()
    now = datetime.datetime.now().isoformat(timespec="seconds")
    cur.execute("SELECT id FROM doctor_notes WHERE screening_id = ?", (screening_id,))
    existing = cur.fetchone()
    if existing:
        cur.execute(
            """UPDATE doctor_notes
               SET doctor_name = ?, recommendation = ?, referral_override = ?, updated_at = ?
               WHERE screening_id = ?""",
            (doctor_name, recommendation, referral_override, now, screening_id),
        )
    else:
        cur.execute(
            """INSERT INTO doctor_notes
               (screening_id, doctor_name, recommendation, referral_override, created_at, updated_at)
               VALUES (?,?,?,?,?,?)""",
            (screening_id, doctor_name, recommendation, referral_override, now, now),
        )
    conn.commit()
    conn.close()


def get_doctor_recommendation(screening_id):
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM doctor_notes WHERE screening_id = ?", (screening_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None
