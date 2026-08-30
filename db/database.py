"""
db/database.py
DRISHTI-XAI — SQLite persistence layer.

This module is a drop-in replacement. It preserves every existing
patient/screening record already stored on disk and only ever ADDS
tables/columns (never drops or recreates anything).

Public API (kept 100% compatible with existing app.py imports):
    init_db()
    authenticate(username, password)
    add_patient(...)
    get_patients()
    get_patient_by_id(patient_id)
    add_screening(...)
    get_screenings(patient_id=None)
    update_followup_status(screening_id, status)

New API added for Doctor Review / registration / printing:
    register_doctor(username, password, full_name)
    add_doctor_review(...)
    get_doctor_reviews()
    get_doctor_review_by_screening(screening_id)
    get_pending_doctor_reviews()
    get_complete_patient_record(patient_id)
"""

import os
import sqlite3
import hashlib
import binascii
import json
from contextlib import contextmanager
from datetime import datetime

# ---------------------------------------------------------------------------
# DB location — reuse DATA_DIR from config.py so the prototype's existing
# database file (if any) is found and reused rather than recreated.
# ---------------------------------------------------------------------------
try:
    from config import DATA_DIR
except Exception:
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "drishti.db")


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------
@contextmanager
def get_connection():
    """Yields a sqlite3 connection with Row access, always closed safely."""
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _row_to_dict(row):
    return dict(row) if row is not None else None


def _rows_to_list(rows):
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Password hashing (PBKDF2-HMAC-SHA256, per-user random salt). No plaintext
# passwords are ever stored.
# ---------------------------------------------------------------------------
def _hash_password(password: str, salt: bytes = None) -> str:
    if salt is None:
        salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return f"{binascii.hexlify(salt).decode()}${binascii.hexlify(dk).decode()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt_hex, _ = stored_hash.split("$")
        salt = binascii.unhexlify(salt_hex)
        return _hash_password(password, salt) == stored_hash
    except Exception:
        return False


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Schema creation + safe migration
# ---------------------------------------------------------------------------
def _table_columns(conn, table_name):
    cur = conn.execute(f"PRAGMA table_info({table_name})")
    return {row["name"] for row in cur.fetchall()}


def _ensure_column(conn, table_name, column_name, column_def):
    """Adds a column only if it does not already exist. Never destructive."""
    cols = _table_columns(conn, table_name)
    if column_name not in cols:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_def}")


def init_db():
    """Creates tables if missing and migrates schema safely. Never drops data."""
    with get_connection() as conn:
        # --- users -----------------------------------------------------
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                full_name TEXT,
                created_at TEXT
            )
        """)

        # --- patients ----------------------------------------------------
        conn.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                age INTEGER,
                diabetes_duration TEXT,
                previous_screening TEXT,
                registered_by TEXT,
                registered_at TEXT
            )
        """)
        # Safe migrations for new optional fields
        _ensure_column(conn, "patients", "gender", "gender TEXT")
        _ensure_column(conn, "patients", "phone", "phone TEXT")

        # --- screenings ----------------------------------------------------
        conn.execute("""
            CREATE TABLE IF NOT EXISTS screenings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                image_path TEXT,
                dr_class TEXT,
                confidence REAL,
                probabilities TEXT,
                gradcam_path TEXT,
                risk_level TEXT,
                risk_score REAL,
                risk_rationale TEXT,
                referral TEXT,
                screened_by TEXT,
                screened_at TEXT,
                followup_status TEXT DEFAULT 'Pending',
                FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
            )
        """)
        # Safe migration: eye field added after initial release
        _ensure_column(conn, "screenings", "eye", "eye TEXT DEFAULT 'Right Eye'")
        _ensure_column(conn, "screenings", "followup_status", "followup_status TEXT DEFAULT 'Pending'")

        # --- doctor_reviews ----------------------------------------------------
        conn.execute("""
            CREATE TABLE IF NOT EXISTS doctor_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                screening_id INTEGER NOT NULL,
                doctor_id TEXT,
                doctor_name TEXT,
                review_status TEXT DEFAULT 'Pending',
                specialist_recommendation TEXT,
                additional_advice TEXT,
                reviewed_at TEXT,
                FOREIGN KEY (screening_id) REFERENCES screenings (id)
            )
        """)

        # --- seed demo users (only if they do not already exist) ------------
        demo_users = [
            ("healthworker1", "worker123", "health_worker", "Priya (Health Worker)"),
            ("doctor1", "doctor123", "doctor", "Dr. Anand Kumar"),
        ]
        for username, password, role, full_name in demo_users:
            existing = conn.execute(
                "SELECT id FROM users WHERE username = ?", (username,)
            ).fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO users (username, password_hash, role, full_name, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (username, _hash_password(password), role, full_name, _now()),
                )


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
def authenticate(username, password):
    """Returns a user dict (without password_hash) on success, else None."""
    if not username or not password:
        return None
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username.strip(),)
        ).fetchone()
    if row is None:
        return None
    if not _verify_password(password, row["password_hash"]):
        return None
    user = _row_to_dict(row)
    user.pop("password_hash", None)
    return user


def register_doctor(username, password, full_name):
    """Registers a new doctor account. Returns (success: bool, message: str)."""
    username = (username or "").strip()
    if not username or not password or not full_name:
        return False, "All fields are required."
    with get_connection() as conn:
        existing = conn.execute(
            "SELECT id FROM users WHERE username = ?", (username,)
        ).fetchone()
        if existing is not None:
            return False, f"Username '{username}' already exists."
        conn.execute(
            "INSERT INTO users (username, password_hash, role, full_name, created_at) "
            "VALUES (?, ?, 'doctor', ?, ?)",
            (username, _hash_password(password), full_name, _now()),
        )
    return True, "Doctor account created successfully."


# ---------------------------------------------------------------------------
# Patients
# ---------------------------------------------------------------------------
def add_patient(patient_id, name, age, diabetes_duration="", previous_screening="No",
                 registered_by="", gender="", phone=""):
    """Adds a new patient. Returns (success: bool, message: str)."""
    patient_id = (patient_id or "").strip()
    name = (name or "").strip()
    if not patient_id or not name:
        return False, "Patient ID and Name are required."
    with get_connection() as conn:
        existing = conn.execute(
            "SELECT id FROM patients WHERE patient_id = ?", (patient_id,)
        ).fetchone()
        if existing is not None:
            return False, f"Patient ID '{patient_id}' already exists."
        conn.execute(
            """INSERT INTO patients
               (patient_id, name, age, gender, diabetes_duration, previous_screening,
                phone, registered_by, registered_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (patient_id, name, age, gender, diabetes_duration, previous_screening,
             phone, registered_by, _now()),
        )
    return True, f"Patient '{name}' registered successfully."


def get_patients():
    """Returns every registered patient, most recently registered first."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM patients ORDER BY id DESC"
        ).fetchall()
    return _rows_to_list(rows)


def get_patient_by_id(patient_id):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM patients WHERE patient_id = ?", (patient_id,)
        ).fetchone()
    return _row_to_dict(row)


# ---------------------------------------------------------------------------
# Screenings
# ---------------------------------------------------------------------------
def add_screening(patient_id, image_path="", dr_class="", confidence=0.0,
                   probabilities=None, gradcam_path="", risk_level="", risk_score=0.0,
                   risk_rationale="", referral="", screened_by="", eye="Right Eye"):
    """Adds a new screening record. Returns the new screening's id."""
    probs_json = json.dumps(probabilities) if probabilities is not None else "{}"
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO screenings
               (patient_id, eye, image_path, dr_class, confidence, probabilities,
                gradcam_path, risk_level, risk_score, risk_rationale, referral,
                screened_by, screened_at, followup_status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'Pending')""",
            (patient_id, eye, image_path, dr_class, confidence, probs_json,
             gradcam_path, risk_level, risk_score, risk_rationale, referral,
             screened_by, _now()),
        )
        return cur.lastrowid


def _deserialize_screening(row):
    d = _row_to_dict(row)
    if d is None:
        return None
    try:
        d["probabilities"] = json.loads(d.get("probabilities") or "{}")
    except Exception:
        d["probabilities"] = {}
    return d


def get_screenings(patient_id=None):
    """Returns screenings (optionally filtered by patient_id), newest first."""
    with get_connection() as conn:
        if patient_id:
            rows = conn.execute(
                "SELECT * FROM screenings WHERE patient_id = ? ORDER BY id DESC",
                (patient_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM screenings ORDER BY id DESC"
            ).fetchall()
    return [_deserialize_screening(r) for r in rows]


def get_screening_by_id(screening_id):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM screenings WHERE id = ?", (screening_id,)
        ).fetchone()
    return _deserialize_screening(row)


def update_followup_status(screening_id, status):
    """Updates the follow-up status of a screening. Returns True on success."""
    with get_connection() as conn:
        cur = conn.execute(
            "UPDATE screenings SET followup_status = ? WHERE id = ?",
            (status, screening_id),
        )
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Doctor reviews
# ---------------------------------------------------------------------------
def add_doctor_review(screening_id, doctor_id, doctor_name, review_status,
                       specialist_recommendation="", additional_advice=""):
    """
    Upserts the doctor review for a screening: one screening -> one review row.
    Returns the review's id.
    """
    with get_connection() as conn:
        existing = conn.execute(
            "SELECT id FROM doctor_reviews WHERE screening_id = ?", (screening_id,)
        ).fetchone()
        if existing is not None:
            conn.execute(
                """UPDATE doctor_reviews
                   SET doctor_id = ?, doctor_name = ?, review_status = ?,
                       specialist_recommendation = ?, additional_advice = ?, reviewed_at = ?
                   WHERE screening_id = ?""",
                (doctor_id, doctor_name, review_status, specialist_recommendation,
                 additional_advice, _now(), screening_id),
            )
            return existing["id"]
        cur = conn.execute(
            """INSERT INTO doctor_reviews
               (screening_id, doctor_id, doctor_name, review_status,
                specialist_recommendation, additional_advice, reviewed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (screening_id, doctor_id, doctor_name, review_status,
             specialist_recommendation, additional_advice, _now()),
        )
        return cur.lastrowid


def get_doctor_reviews():
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM doctor_reviews ORDER BY id DESC"
        ).fetchall()
    return _rows_to_list(rows)


def get_doctor_review_by_screening(screening_id):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM doctor_reviews WHERE screening_id = ?", (screening_id,)
        ).fetchone()
    return _row_to_dict(row)


def get_pending_doctor_reviews():
    """
    Returns screenings that have no review yet or are still 'Pending',
    each enriched with basic patient info, newest first.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT s.*, p.name AS patient_name, p.age AS patient_age,
                   dr.review_status AS existing_review_status
            FROM screenings s
            LEFT JOIN patients p ON p.patient_id = s.patient_id
            LEFT JOIN doctor_reviews dr ON dr.screening_id = s.id
            WHERE dr.id IS NULL OR dr.review_status = 'Pending'
            ORDER BY s.id DESC
            """
        ).fetchall()
    results = []
    for r in rows:
        d = _deserialize_screening(r)
        d["patient_name"] = r["patient_name"]
        d["patient_age"] = r["patient_age"]
        results.append(d)
    return results


# ---------------------------------------------------------------------------
# Combined record (used by the print / doctor review pages)
# ---------------------------------------------------------------------------
def get_complete_patient_record(patient_id):
    """Returns a dict with patient info + all screenings (each with its review)."""
    patient = get_patient_by_id(patient_id)
    if patient is None:
        return None
    screenings = get_screenings(patient_id)
    for s in screenings:
        s["doctor_review"] = get_doctor_review_by_screening(s["id"])
    return {"patient": patient, "screenings": screenings}
