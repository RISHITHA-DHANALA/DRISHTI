"""
config.py
---------
Central place for constants used across DRISHTI-XAI.
Keeping these in one file makes it easy to tune thresholds
or swap in a real trained model later without touching UI code.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
SAMPLE_IMAGES_DIR = os.path.join(ASSETS_DIR, "sample_images")
DB_PATH = os.path.join(DATA_DIR, "drishti_xai.db")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SAMPLE_IMAGES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# App metadata
# ---------------------------------------------------------------------------
APP_NAME = "DRISHTI-XAI"
APP_TAGLINE = "Explainable AI Diabetic Retinopathy Screening for Rural India"
APP_VERSION = "0.1.0-prototype"
IS_PROTOTYPE = True  # Never set False without real clinical validation.

# ---------------------------------------------------------------------------
# DR severity classes (standard ICDR-style grading, simplified to 4 classes)
# ---------------------------------------------------------------------------
DR_CLASSES = ["No DR", "Mild DR", "Moderate DR", "Severe DR"]

DR_CLASS_COLORS = {
    "No DR": "#2E7D32",        # green
    "Mild DR": "#F9A825",      # amber
    "Moderate DR": "#EF6C00",  # orange
    "Severe DR": "#C62828",    # red
}

# ---------------------------------------------------------------------------
# Risk levels & badge colors
# ---------------------------------------------------------------------------
RISK_LEVELS = ["Low", "Medium", "High", "Critical"]

RISK_COLORS = {
    "Low": "#2E7D32",
    "Medium": "#F9A825",
    "High": "#EF6C00",
    "Critical": "#B71C1C",
}

# ---------------------------------------------------------------------------
# Referral actions
# ---------------------------------------------------------------------------
REFERRAL_ACTIONS = {
    "Low": "Routine annual screening",
    "Medium": "Follow-up screening in 6 months",
    "High": "Refer to ophthalmologist within 4 weeks",
    "Critical": "Urgent specialist referral (within 1 week)",
}

# ---------------------------------------------------------------------------
# Follow-up statuses
# ---------------------------------------------------------------------------
FOLLOWUP_STATUSES = ["Pending", "Referred", "Completed"]

# ---------------------------------------------------------------------------
# Image quality thresholds (tunable)
# ---------------------------------------------------------------------------
BLUR_LAPLACIAN_THRESHOLD = 80.0     # below this variance => likely blurry
MIN_BRIGHTNESS = 40                 # 0-255 mean brightness lower bound
MAX_BRIGHTNESS = 220                # 0-255 mean brightness upper bound
MIN_RESOLUTION = (150, 150)         # minimum (width, height) accepted

# ---------------------------------------------------------------------------
# Demo credentials (prototype only — replace with real auth before any
# real-world deployment; never hardcode credentials in production).
# ---------------------------------------------------------------------------
DEMO_USERS = [
    # username, password, role, display_name
    ("healthworker1", "worker123", "Health Worker", "Asha Devi"),
    ("doctor1", "doctor123", "Doctor", "Dr. R. Krishnan"),
]

SUPPORTED_LANGUAGES = {
    "English": "en",
    "తెలుగు (Telugu)": "te",
    "हिन्दी (Hindi)": "hi",
}
