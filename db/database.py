"""
db/database.py
--------------
SQLite persistence layer for DRISHTI-XAI.
"""

import sqlite3
import hashlib
import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DB_PATH, DEMO_USERS


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _column_exists(cur, table, column):
    cur.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cur.fetchall())


def init_db():
    conn = _connect()
    cur = conn.cursor()

    # ---------------------------------------------------------
    # USERS
    # ---------------------------------------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('Health Worker', 'Doctor')),
            display_name TEXT NOT NULL
        )
    """)

    # ---------------------------------------------------------
    # PATIENTS
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # SCREENINGS
    # ---------------------------------------------------------
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

            doctor_decision TEXT,
            specialist_exam TEXT,
            blood_sugar_advice TEXT,
            bp_cholesterol_advice TEXT,
            diet_lifestyle_advice TEXT,
            warning_signs_advice TEXT,
            followup_advice TEXT,
            additional_doctor_advice TEXT,
            reviewed_by TEXT,
            reviewed_at TEXT,

            FOREIGN KEY(patient_id) REFERENCES patients(id)
        )
    """)

    # ---------------------------------------------------------
    # SAFE MIGRATION FOR EXISTING DATABASES
    # ---------------------------------------------------------
    new_columns = {
        "doctor_decision": "TEXT",
        "specialist_exam": "TEXT",
        "blood_sugar_advice": "TEXT",
        "bp_cholesterol_advice": "TEXT",
        "diet_lifestyle_advice": "TEXT",
        "warning_signs_advice": "TEXT",
        "followup_advice": "TEXT",
        "additional_doctor_advice": "TEXT",
        "reviewed_by": "TEXT",
        "reviewed_at": "TEXT",
    }

    for column, datatype in new_columns.items():
        if not _column_exists(cur, "screenings", column):
            cur.execute(
                f"ALTER TABLE screenings ADD COLUMN {column} {datatype}"
            )

    # ---------------------------------------------------------
    # DEMO USERS
    # ---------------------------------------------------------
    for username, password, role, display_name in DEMO_USERS:
        cur.execute(
            "SELECT id FROM users WHERE username = ?",
            (username,)
        )

        if cur.fetchone() is None:
            cur.execute(
                """
                INSERT INTO users
                (username, password_hash, role, display_name)
                VALUES (?, ?, ?, ?)
                """,
                (
                    username,
                    _hash_password(password),
                    role,
                    display_name,
                ),
            )

    conn.commit()
    conn.close()


# =============================================================
# USERS
# =============================================================

def authenticate(username: str, password: str):
    conn = _connect()
    cur = conn.cursor()

    cur.execute(
        "SELECT * FROM users WHERE username = ?",
        (username,)
    )

    row = cur.fetchone()
    conn.close()

    if row and row["password_hash"] == _hash_password(password):
        return dict(row)

    return None


def add_doctor(username, password, display_name):
    conn = _connect()
    cur = conn.cursor()

    try:
        cur.execute(
            """
            INSERT INTO users
            (username, password_hash, role, display_name)
            VALUES (?, ?, 'Doctor', ?)
            """,
            (
                username,
                _hash_password(password),
                display_name,
            ),
        )

        conn.commit()
        doctor_id = cur.lastrowid
        conn.close()

        return doctor_id

    except sqlite3.IntegrityError:
        conn.close()
        return None


def get_doctors():
    conn = _connect()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, username, display_name
        FROM users
        WHERE role = 'Doctor'
        ORDER BY display_name
    """)

    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    return rows


# =============================================================
# PATIENTS
# =============================================================

def add_patient(
    patient_code,
    name,
    age,
    diabetes_duration,
    previous_screening,
    registered_by
):
    conn = _connect()
    cur = conn.cursor()

    try:
        cur.execute(
            """
            INSERT INTO patients
            (
                patient_code,
                name,
                age,
                diabetes_duration,
                previous_screening,
                registered_by,
                registered_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patient_code,
                name,
                age,
                diabetes_duration,
                previous_screening,
                registered_by,
                datetime.datetime.now().isoformat(
                    timespec="seconds"
                ),
            ),
        )

        conn.commit()
        new_id = cur.lastrowid
        conn.close()

        return new_id

    except sqlite3.IntegrityError:
        conn.close()
        return None


def get_patients():
    conn = _connect()
    cur = conn.cursor()

    cur.execute("""
        SELECT *
        FROM patients
        ORDER BY registered_at DESC
    """)

    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    return rows


def get_patient_by_id(patient_id):
    conn = _connect()
    cur = conn.cursor()

    cur.execute(
        "SELECT * FROM patients WHERE id = ?",
        (patient_id,)
    )

    row = cur.fetchone()
    conn.close()

    return dict(row) if row else None


# =============================================================
# SCREENINGS
# =============================================================

def add_screening(
    patient_id,
    image_path,
    dr_class,
    confidence,
    risk_level,
    referral,
    heatmap_path,
    screened_by
):
    conn = _connect()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO screenings
        (
            patient_id,
            image_path,
            dr_class,
            confidence,
            risk_level,
            referral,
            followup_status,
            heatmap_path,
            screened_by,
            screened_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            patient_id,
            image_path,
            dr_class,
            confidence,
            risk_level,
            referral,
            "Pending",
            heatmap_path,
            screened_by,
            datetime.datetime.now().isoformat(
                timespec="seconds"
            ),
        ),
    )

    conn.commit()
    new_id = cur.lastrowid
    conn.close()

    return new_id


def get_screenings():
    conn = _connect()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            s.*,
            p.patient_code,
            p.name AS patient_name,
            p.age,
            p.diabetes_duration,
            p.previous_screening
        FROM screenings s
        JOIN patients p
        ON s.patient_id = p.id
        ORDER BY s.screened_at DESC
    """)

    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    return rows


def get_screening_by_id(screening_id):
    conn = _connect()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            s.*,
            p.patient_code,
            p.name AS patient_name,
            p.age,
            p.diabetes_duration,
            p.previous_screening,
            p.registered_by,
            p.registered_at
        FROM screenings s
        JOIN patients p
        ON s.patient_id = p.id
        WHERE s.id = ?
    """, (screening_id,))

    row = cur.fetchone()
    conn.close()

    return dict(row) if row else None


def update_followup_status(screening_id, status):
    conn = _connect()
    cur = conn.cursor()

    cur.execute(
        """
        UPDATE screenings
        SET followup_status = ?
        WHERE id = ?
        """,
        (status, screening_id)
    )

    conn.commit()
    conn.close()


# =============================================================
# DOCTOR REVIEW
# =============================================================

def save_doctor_review(
    screening_id,
    doctor_decision,
    specialist_exam,
    blood_sugar_advice,
    bp_cholesterol_advice,
    diet_lifestyle_advice,
    warning_signs_advice,
    followup_advice,
    additional_doctor_advice,
    reviewed_by
):
    conn = _connect()
    cur = conn.cursor()

    cur.execute(
        """
        UPDATE screenings
        SET
            doctor_decision = ?,
            specialist_exam = ?,
            blood_sugar_advice = ?,
            bp_cholesterol_advice = ?,
            diet_lifestyle_advice = ?,
            warning_signs_advice = ?,
            followup_advice = ?,
            additional_doctor_advice = ?,
            reviewed_by = ?,
            reviewed_at = ?
        WHERE id = ?
        """,
        (
            doctor_decision,
            specialist_exam,
            blood_sugar_advice,
            bp_cholesterol_advice,
            diet_lifestyle_advice,
            warning_signs_advice,
            followup_advice,
            additional_doctor_advice,
            reviewed_by,
            datetime.datetime.now().isoformat(
                timespec="seconds"
            ),
            screening_id,
        )
    )

    conn.commit()
    conn.close()
