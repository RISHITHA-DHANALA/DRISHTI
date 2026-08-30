"""
app.py
DRISHTI-XAI — Explainable AI Diabetic Retinopathy Screening for Rural India

Run with:  streamlit run app.py
"""

import json
import base64
import io
from datetime import datetime

import streamlit as st
import pandas as pd

from config import (
    APP_NAME,
    APP_TAGLINE,
    IS_PROTOTYPE,
    DR_CLASSES,
    SUPPORTED_LANGUAGES,
    FOLLOWUP_STATUSES,
    SAMPLE_IMAGES_DIR,
    DATA_DIR,
)

from i18n.translations import t

from db.database import (
    init_db,
    authenticate,
    add_patient,
    get_patients,
    get_patient_by_id,
    add_screening,
    get_screenings,
    update_followup_status,
    register_doctor,
    add_doctor_review,
    get_doctor_reviews,
    get_doctor_review_by_screening,
    get_pending_doctor_reviews,
    get_complete_patient_record,
    get_screening_by_id,
)

from core.image_quality import check_image_quality
from core.model import DRModel
from core.gradcam import generate_explanation
from core.risk_engine import assess_risk

from utils.demo_data import generate_all_samples, seed_demo_patients

from utils.helpers import (
    pil_to_bgr,
    bgr_to_rgb,
    save_uploaded_image,
    risk_badge,
    dr_class_badge,
)

from PIL import Image
import numpy as np
import streamlit.components.v1 as components


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title=APP_NAME,
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# THEME / CSS
# =============================================================================
def inject_css():
    """
    Injects the DRISHTI-XAI healthcare theme.
    Uses Streamlit's own CSS variables (--text-color, --background-color,
    --secondary-background-color) so text stays legible in BOTH light and
    dark Streamlit themes, plus a prefers-color-scheme fallback.
    """
    st.markdown(
        """
        <style>
        :root {
            --dxai-navy: #0B1F3A;
            --dxai-teal: #008C95;
            --dxai-teal-light: #E8F6F7;
            --dxai-bg-light: #F5F9FA;
            --dxai-text-secondary: #607D8B;
            --dxai-success: #16803C;
            --dxai-warning: #D97706;
            --dxai-danger: #DC2626;
        }

        /* ---------- Global text always follows Streamlit's own theme ---------- */
        html, body, [class*="css"] {
            color: var(--text-color);
        }

        #MainMenu, footer {visibility: hidden;}

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        /* ---------- Animated gradient header ---------- */
        .dxai-header {
            background: linear-gradient(120deg, var(--dxai-navy), var(--dxai-teal), var(--dxai-navy));
            background-size: 200% 200%;
            animation: dxai-gradient 12s ease infinite;
            border-radius: 16px;
            padding: 28px 32px;
            margin-bottom: 20px;
            box-shadow: 0 8px 24px rgba(11,31,58,0.25);
        }
        @keyframes dxai-gradient {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
        .dxai-header h1, .dxai-header p {
            color: #FFFFFF !important;
            margin: 0;
        }
        .dxai-header h1 { font-size: 1.9rem; font-weight: 700; }
        .dxai-header p { font-size: 0.95rem; opacity: 0.9; margin-top: 4px; }

        /* ---------- Cards ---------- */
        .dxai-card {
            background: var(--secondary-background-color);
            border: 1px solid rgba(128,128,128,0.18);
            border-radius: 14px;
            padding: 20px;
            margin-bottom: 16px;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
            animation: dxai-fade-in 0.5s ease;
        }
        .dxai-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 18px rgba(0,140,149,0.18);
        }
        @keyframes dxai-fade-in {
            from {opacity: 0; transform: translateY(6px);}
            to {opacity: 1; transform: translateY(0);}
        }
        .dxai-card h3, .dxai-card h4, .dxai-card p, .dxai-card span, .dxai-card li {
            color: var(--text-color) !important;
        }

        /* ---------- KPI cards ---------- */
        .dxai-kpi {
            background: var(--secondary-background-color);
            border-radius: 14px;
            padding: 18px 20px;
            border-left: 5px solid var(--dxai-teal);
            animation: dxai-fade-in 0.5s ease;
        }
        .dxai-kpi .dxai-kpi-label {
            color: var(--dxai-text-secondary) !important;
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .dxai-kpi .dxai-kpi-value {
            color: var(--text-color) !important;
            font-size: 2rem;
            font-weight: 700;
            margin-top: 4px;
        }

        /* ---------- Badges ---------- */
        .dxai-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }
        .badge-low { background: rgba(22,128,60,0.15); color: var(--dxai-success); }
        .badge-moderate { background: rgba(217,119,6,0.15); color: var(--dxai-warning); }
        .badge-high, .badge-critical { background: rgba(220,38,38,0.15); color: var(--dxai-danger); }
        .badge-info { background: rgba(0,140,149,0.15); color: var(--dxai-teal); }

        /* ---------- Workflow pills ---------- */
        .dxai-flow { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; margin: 10px 0 4px 0; }
        .dxai-flow-step {
            background: var(--dxai-teal-light);
            color: var(--dxai-navy) !important;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.82rem;
            font-weight: 600;
        }
        .dxai-flow-arrow { color: var(--dxai-teal); font-weight: 700; }

        /* ---------- Login page ---------- */
        .dxai-login-wrap {
            display: flex;
            justify-content: center;
            padding-top: 10px;
        }
        .dxai-login-card {
            background: var(--secondary-background-color);
            border-radius: 20px;
            padding: 36px 40px;
            max-width: 480px;
            width: 100%;
            box-shadow: 0 12px 32px rgba(11,31,58,0.18);
            position: relative;
            overflow: hidden;
        }
        .dxai-eye-icon {
            font-size: 3rem;
            text-align: center;
            display: block;
            animation: dxai-pulse 2.6s ease-in-out infinite;
        }
        @keyframes dxai-pulse {
            0%, 100% { text-shadow: 0 0 0px rgba(0,140,149,0.0); transform: scale(1); }
            50% { text-shadow: 0 0 22px rgba(0,140,149,0.55); transform: scale(1.05); }
        }
        .dxai-floating-circle {
            position: absolute;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(0,140,149,0.16), transparent 70%);
            animation: dxai-float 9s ease-in-out infinite;
        }
        .dxai-fc1 { width: 140px; height: 140px; top: -40px; left: -40px; animation-delay: 0s; }
        .dxai-fc2 { width: 90px; height: 90px; bottom: -20px; right: -20px; animation-delay: 2.5s; }
        @keyframes dxai-float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-14px); }
        }
        .dxai-login-title { text-align: center; color: var(--dxai-navy) !important; font-weight: 700; font-size: 1.55rem; margin: 6px 0 2px 0; }
        .dxai-login-tagline { text-align: center; color: var(--dxai-teal) !important; font-size: 0.92rem; font-weight: 600; margin-bottom: 10px; }
        .dxai-login-desc { text-align: center; color: var(--dxai-text-secondary) !important; font-size: 0.86rem; margin-bottom: 14px; }

        .dxai-disclaimer {
            background: rgba(217,119,6,0.12);
            border: 1px solid rgba(217,119,6,0.3);
            border-radius: 10px;
            padding: 10px 14px;
            font-size: 0.78rem;
            color: var(--dxai-warning) !important;
            margin-top: 14px;
        }
        .dxai-demo-box {
            background: var(--dxai-teal-light);
            border-radius: 10px;
            padding: 10px 14px;
            font-size: 0.8rem;
            color: var(--dxai-navy) !important;
            margin-top: 10px;
        }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background: var(--dxai-navy);
        }
        section[data-testid="stSidebar"] * {
            color: #FFFFFF !important;
        }
        section[data-testid="stSidebar"] .stButton button {
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.18);
            color: #FFFFFF !important;
            border-radius: 10px;
        }
        section[data-testid="stSidebar"] .stButton button:hover {
            background: var(--dxai-teal);
            border-color: var(--dxai-teal);
        }

        /* ---------- Buttons ---------- */
        .stButton > button {
            border-radius: 10px;
            font-weight: 600;
            border: none;
            background: var(--dxai-teal);
            color: #FFFFFF;
            transition: all 0.15s ease;
        }
        .stButton > button:hover {
            background: var(--dxai-navy);
            transform: translateY(-1px);
        }

        /* ---------- Explainability box ---------- */
        .dxai-xai-box {
            background: var(--dxai-teal-light);
            border-left: 4px solid var(--dxai-teal);
            border-radius: 10px;
            padding: 14px 16px;
            font-size: 0.88rem;
            color: var(--dxai-navy) !important;
        }

        .dxai-section-title {
            color: var(--dxai-navy) !important;
            font-weight: 700;
            font-size: 1.25rem;
            margin: 6px 0 14px 0;
            border-bottom: 2px solid var(--dxai-teal-light);
            padding-bottom: 6px;
        }

        @media (prefers-color-scheme: dark) {
            .dxai-login-title, .dxai-flow-step, .dxai-xai-box, .dxai-section-title,
            .dxai-demo-box { color: #FFFFFF !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# SESSION STATE
# =============================================================================
def init_session_state():
    defaults = {
        "logged_in": False,
        "user": None,
        "language": SUPPORTED_LANGUAGES[0] if SUPPORTED_LANGUAGES else "en",
        "page": "Dashboard",
        "selected_patient_id": None,
        "selected_screening_id": None,
        "uploaded_image": None,
        "quality_result": None,
        "prediction_result": None,
        "gradcam_result": None,
        "risk_result": None,
        "screening_eye": "Right Eye",
        "db_initialized": False,
        "doctor_review_draft": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if not st.session_state.db_initialized:
        init_db()
        try:
            generate_all_samples()
        except Exception:
            pass
        try:
            seed_demo_patients()
        except Exception:
            pass
        st.session_state.db_initialized = True


# =============================================================================
# COMPATIBILITY HELPERS
# (core/*.py signatures may vary slightly — these wrappers normalize output
#  so the rest of the app never touches undefined keys.)
# =============================================================================
def run_quality_check(image_bgr):
    try:
        result = check_image_quality(image_bgr)
    except Exception as e:
        result = {"passed": True, "blur_score": None, "brightness": None,
                   "resolution": None, "issues": [f"Quality check unavailable: {e}"]}
    if not isinstance(result, dict):
        result = {"passed": bool(result), "blur_score": None, "brightness": None,
                   "resolution": None, "issues": []}
    result.setdefault("passed", True)
    result.setdefault("blur_score", None)
    result.setdefault("brightness", None)
    result.setdefault("resolution", None)
    result.setdefault("issues", [])
    return result


@st.cache_resource(show_spinner=False)
def load_model():
    return DRModel()


def run_prediction(model, image_bgr):
    raw = model.predict(image_bgr)
    dr_class, confidence, probabilities = None, 0.0, {}
    if isinstance(raw, dict):
        dr_class = raw.get("dr_class") or raw.get("class") or raw.get("label")
        confidence = raw.get("confidence") or raw.get("score") or 0.0
        probabilities = raw.get("probabilities") or raw.get("probs") or {}
    elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
        dr_class, confidence = raw[0], raw[1]
        probabilities = raw[2] if len(raw) > 2 else {}
    if dr_class is None and DR_CLASSES:
        dr_class = DR_CLASSES[0]
    return {"dr_class": dr_class, "confidence": float(confidence or 0.0),
            "probabilities": probabilities, "raw": raw}


def run_gradcam(model, image_bgr, prediction):
    try:
        try:
            result = generate_explanation(model, image_bgr, prediction.get("dr_class"))
        except TypeError:
            result = generate_explanation(model, image_bgr)
    except Exception as e:
        return {"image": None, "explanation_text": f"Grad-CAM unavailable: {e}"}

    heatmap_image = None
    explanation_text = ""
    if isinstance(result, dict):
        heatmap_image = result.get("overlay") or result.get("heatmap") or result.get("image")
        explanation_text = result.get("explanation") or result.get("text") or ""
    elif isinstance(result, np.ndarray):
        heatmap_image = result
    else:
        heatmap_image = result
    return {"image": heatmap_image, "explanation_text": explanation_text}


def run_risk_assessment(dr_class, confidence, patient=None):
    try:
        try:
            result = assess_risk(dr_class, confidence, patient)
        except TypeError:
            result = assess_risk(dr_class, confidence)
    except Exception as e:
        result = {"risk_level": "Unknown", "risk_score": 0.0,
                   "rationale": f"Risk engine unavailable: {e}", "referral": "Manual review advised"}
    if not isinstance(result, dict):
        result = {"risk_level": str(result), "risk_score": 0.0, "rationale": "", "referral": ""}
    result.setdefault("risk_level", "Unknown")
    result.setdefault("risk_score", 0.0)
    result.setdefault("rationale", result.get("reason", ""))
    result.setdefault("referral", result.get("recommendation", ""))
    return result


def to_display_image(img):
    """Normalizes model/gradcam outputs (np.ndarray BGR/RGB or PIL) for st.image."""
    if img is None:
        return None
    if isinstance(img, np.ndarray):
        try:
            return bgr_to_rgb(img)
        except Exception:
            return img
    return img


# =============================================================================
# SMALL UI HELPERS
# =============================================================================
def render_header():
    st.markdown(
        f"""
        <div class="dxai-header">
            <h1>👁️ {APP_NAME}</h1>
            <p>{APP_TAGLINE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi(label, value, icon=""):
    st.markdown(
        f"""
        <div class="dxai-kpi">
            <div class="dxai-kpi-label">{icon} {label}</div>
            <div class="dxai-kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_title(text):
    st.markdown(f'<div class="dxai-section-title">{text}</div>', unsafe_allow_html=True)


def render_flow(steps):
    parts = []
    for i, step in enumerate(steps):
        parts.append(f'<span class="dxai-flow-step">{step}</span>')
        if i < len(steps) - 1:
            parts.append('<span class="dxai-flow-arrow">➜</span>')
    st.markdown(f'<div class="dxai-flow">{"".join(parts)}</div>', unsafe_allow_html=True)


def safe_badge(value, kind="risk"):
    """Falls back to a generic badge if utils.helpers badge functions error out."""
    try:
        if kind == "risk":
            return risk_badge(value)
        return dr_class_badge(value)
    except Exception:
        css_class = "badge-info"
        v = str(value).lower()
        if "low" in v:
            css_class = "badge-low"
        elif "moderate" in v or "medium" in v:
            css_class = "badge-moderate"
        elif "high" in v or "critical" in v or "severe" in v:
            css_class = "badge-high"
        return f'<span class="dxai-badge {css_class}">{value}</span>'


# =============================================================================
# LOGIN PAGE
# =============================================================================
def login_page():
    st.markdown('<div class="dxai-login-wrap">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="dxai-login-card">
            <div class="dxai-floating-circle dxai-fc1"></div>
            <div class="dxai-floating-circle dxai-fc2"></div>
            <span class="dxai-eye-icon">👁️</span>
            <div class="dxai-login-title">DRISHTI-XAI</div>
            <div class="dxai-login-tagline">Explainable AI Diabetic Retinopathy Screening for Rural India</div>
            <div class="dxai-login-desc">
                AI-assisted retinal screening workflow connecting PHC health workers with eye specialists.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    center = st.columns([1, 1.3, 1])[1]
    with center:
        render_flow(["PHC", "AI Screening", "Specialist"])

        lang = st.selectbox(
            "🌐 Language / भाषा",
            options=SUPPORTED_LANGUAGES if SUPPORTED_LANGUAGES else ["en"],
            index=0,
            key="login_language",
        )
        st.session_state.language = lang

        st.markdown("#### 🔒 Secure Login")
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            user = authenticate(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.user = user
                st.success(f"Welcome, {user.get('full_name') or user.get('username')}!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

        st.markdown(
            """
            <div class="dxai-demo-box">
                <b>Demo accounts</b><br>
                Health Worker: <code>healthworker1 / worker123</code><br>
                Doctor: <code>doctor1 / doctor123</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="dxai-disclaimer">
                ⚠️ <b>PROTOTYPE</b> — AI-assisted screening aid only.
                Not a certified medical diagnostic device.
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("👩‍⚕️ New doctor? Register here"):
            with st.form("doctor_register_form"):
                d_name = st.text_input("Doctor Name")
                d_username = st.text_input("Choose Username")
                d_pass = st.text_input("Choose Password", type="password")
                d_pass2 = st.text_input("Confirm Password", type="password")
                reg_submitted = st.form_submit_button("Register as Doctor")
            if reg_submitted:
                if not d_name or not d_username or not d_pass:
                    st.error("Please fill in all fields.")
                elif d_pass != d_pass2:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = register_doctor(d_username, d_pass, d_name)
                    if ok:
                        st.success(msg + " You can now log in above.")
                    else:
                        st.error(msg)


# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar():
    user = st.session_state.user
    role = user.get("role", "health_worker")

    with st.sidebar:
        st.markdown("### 👁️ DRISHTI-XAI")
        st.caption("Rural Retina Screening")
        st.markdown(f"**Welcome, {user.get('full_name') or user.get('username')}**")
        role_label = "Doctor" if role == "doctor" else "Health Worker"
        st.caption(f"Role: {role_label}")

        st.selectbox(
            "🌐 Language",
            options=SUPPORTED_LANGUAGES if SUPPORTED_LANGUAGES else ["en"],
            key="language",
        )

        st.markdown("---")
        st.markdown("#### Navigation")

        if role == "doctor":
            nav_items = [
                ("🏠 Dashboard", "Dashboard"),
                ("👥 Patient Records", "Patient Records"),
                ("🩺 Doctor Review", "Doctor Review"),
                ("📋 Follow-up", "Follow-up"),
                ("📊 Analytics", "Analytics"),
            ]
        else:
            nav_items = [
                ("🏠 Dashboard", "Dashboard"),
                ("👤 Register Patient", "Register Patient"),
                ("🔬 New Screening", "New Screening"),
                ("📋 Patient Records", "Patient Records"),
            ]

        for label, page_key in nav_items:
            if st.button(label, key=f"nav_{page_key}", use_container_width=True):
                st.session_state.page = page_key
                st.rerun()

        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            for key in ["logged_in", "user", "selected_patient_id", "selected_screening_id",
                        "uploaded_image", "quality_result", "prediction_result",
                        "gradcam_result", "risk_result"]:
                st.session_state[key] = False if key == "logged_in" else None
            st.rerun()


# =============================================================================
# DASHBOARD
# =============================================================================
def page_dashboard():
    render_header()
    render_section_title("Dashboard")

    patients = get_patients()
    screenings = get_screenings()

    total_patients = len(patients)
    total_screenings = len(screenings)
    high_risk = sum(1 for s in screenings if str(s.get("risk_level", "")).lower() in ("high", "critical"))
    pending_followup = sum(1 for s in screenings if s.get("followup_status", "Pending") == "Pending")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi("Total Patients", total_patients, "👥")
    with c2:
        render_kpi("Screenings", total_screenings, "🔬")
    with c3:
        render_kpi("High Risk", high_risk, "⚠️")
    with c4:
        render_kpi("Pending Follow-up", pending_followup, "📅")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="dxai-card">', unsafe_allow_html=True)
        st.markdown("**DR Severity Distribution**")
        if screenings:
            df = pd.DataFrame(screenings)
            counts = df["dr_class"].fillna("Unknown").value_counts()
            st.bar_chart(counts)
        else:
            st.caption("No screenings yet.")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="dxai-card">', unsafe_allow_html=True)
        st.markdown("**Risk Distribution**")
        if screenings:
            df = pd.DataFrame(screenings)
            counts = df["risk_level"].fillna("Unknown").value_counts()
            st.bar_chart(counts)
        else:
            st.caption("No screenings yet.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="dxai-card">', unsafe_allow_html=True)
    st.markdown("**Follow-up Status**")
    if screenings:
        df = pd.DataFrame(screenings)
        counts = df["followup_status"].fillna("Pending").value_counts()
        st.bar_chart(counts)
    else:
        st.caption("No follow-up data yet.")
    st.markdown("</div>", unsafe_allow_html=True)

    render_section_title("Recent Screenings")
    if screenings:
        df = pd.DataFrame(screenings)[["patient_id", "eye", "dr_class", "risk_level",
                                        "followup_status", "screened_at"]].head(10)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No screenings recorded yet.")

    render_section_title("Priority Cases")
    priority = [s for s in screenings if str(s.get("risk_level", "")).lower() in ("high", "critical")][:8]
    if priority:
        for s in priority:
            patient = get_patient_by_id(s["patient_id"])
            pname = patient["name"] if patient else s["patient_id"]
            st.markdown(
                f"""<div class="dxai-card">
                    <b>{pname}</b> ({s['patient_id']}) — {s.get('eye','')}<br>
                    {safe_badge(s.get('risk_level'), 'risk')} &nbsp; {safe_badge(s.get('dr_class'), 'dr')}
                    <br><span style="color:var(--dxai-text-secondary)">Screened: {s.get('screened_at','')}</span>
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No high/critical risk cases at the moment.")


# =============================================================================
# REGISTER PATIENT
# =============================================================================
def page_register_patient():
    render_header()
    render_section_title("Register Patient")

    with st.form("register_patient_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            patient_id = st.text_input("Patient ID *")
            name = st.text_input("Patient Name *")
            age = st.number_input("Age", min_value=0, max_value=120, value=45)
            gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        with col2:
            diabetes_duration = st.text_input("Diabetes Duration (years)", value="")
            previous_screening = st.selectbox("Previous Screening", ["No", "Yes"])
            phone = st.text_input("Phone / Contact (optional)", value="")
            st.text_input("Registration Date", value=datetime.now().strftime("%Y-%m-%d"), disabled=True)

        registered_by = st.session_state.user.get("full_name") or st.session_state.user.get("username")
        st.caption(f"Registered by: {registered_by}")

        submitted = st.form_submit_button("✅ Register Patient", use_container_width=True)

    if submitted:
        ok, msg = add_patient(
            patient_id=patient_id,
            name=name,
            age=int(age),
            diabetes_duration=diabetes_duration,
            previous_screening=previous_screening,
            registered_by=registered_by,
            gender=gender,
            phone=phone,
        )
        if ok:
            st.success(f"✓ {msg}")
        else:
            st.error(msg)

    render_section_title("Registered Patients")
    patients = get_patients()
    if patients:
        df = pd.DataFrame(patients)
        cols = [c for c in ["patient_id", "name", "age", "gender", "registered_at"] if c in df.columns]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)
    else:
        st.info("No patients registered yet.")


# =============================================================================
# NEW SCREENING
# =============================================================================
def page_new_screening():
    render_header()
    render_section_title("New Screening")
    render_flow(["Select Patient", "Upload Image", "Quality Check", "AI Prediction", "Grad-CAM", "Risk & Referral"])

    patients = get_patients()
    if not patients:
        st.warning("No patients registered yet. Please register a patient first.")
        return

    patient_options = {f"{p['name']} ({p['patient_id']})": p["patient_id"] for p in patients}
    selected_label = st.selectbox("👤 Select Patient", list(patient_options.keys()))
    patient_id = patient_options[selected_label]
    patient = get_patient_by_id(patient_id)

    eye = st.radio("👁️ Eye", ["Left Eye", "Right Eye"], horizontal=True, key="screening_eye")

    st.markdown("#### 📤 Fundus Image")
    use_sample = st.checkbox("Use a demo/sample image instead of uploading")
    image_pil = None

    if use_sample:
        import os
        sample_files = []
        try:
            sample_files = [f for f in os.listdir(SAMPLE_IMAGES_DIR)
                             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        except Exception:
            pass
        if sample_files:
            chosen = st.selectbox("Choose a sample image", sample_files)
            import os as _os
            image_pil = Image.open(_os.path.join(SAMPLE_IMAGES_DIR, chosen)).convert("RGB")
        else:
            st.info("No sample images found.")
    else:
        uploaded_file = st.file_uploader("Upload fundus image (JPG/JPEG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image_pil = Image.open(uploaded_file).convert("RGB")

    if image_pil is not None:
        st.session_state.uploaded_image = image_pil
        st.image(image_pil, caption="Preview", width=340)

        if st.button("🔍 Run AI Screening", use_container_width=True):
            with st.spinner("Checking image quality..."):
                image_bgr = pil_to_bgr(image_pil)
                quality = run_quality_check(image_bgr)
                st.session_state.quality_result = quality

            if not quality.get("passed", True):
                st.error("❌ Image quality check failed. Please retake or choose another image.")
            else:
                with st.spinner("Running AI prediction..."):
                    model = load_model()
                    prediction = run_prediction(model, image_bgr)
                    st.session_state.prediction_result = prediction

                with st.spinner("Generating Grad-CAM explanation..."):
                    gradcam = run_gradcam(model, image_bgr, prediction)
                    st.session_state.gradcam_result = gradcam

                with st.spinner("Assessing risk..."):
                    risk = run_risk_assessment(prediction["dr_class"], prediction["confidence"], patient)
                    st.session_state.risk_result = risk

    # ---- Results ----
    if st.session_state.quality_result is not None:
        render_section_title("Image Quality")
        q = st.session_state.quality_result
        qc1, qc2, qc3, qc4 = st.columns(4)
        qc1.metric("Blur Score", q.get("blur_score") if q.get("blur_score") is not None else "N/A")
        qc2.metric("Brightness", q.get("brightness") if q.get("brightness") is not None else "N/A")
        qc3.metric("Resolution", q.get("resolution") if q.get("resolution") is not None else "N/A")
        qc4.metric("Quality", "✅ Pass" if q.get("passed") else "❌ Fail")
        if q.get("issues"):
            st.warning(" • ".join(str(i) for i in q["issues"]))

    if st.session_state.prediction_result is not None:
        render_section_title("AI Prediction")
        pred = st.session_state.prediction_result
        pc1, pc2 = st.columns([1, 1])
        with pc1:
            st.markdown(safe_badge(pred["dr_class"], "dr"), unsafe_allow_html=True)
            st.metric("Confidence", f"{pred['confidence']*100:.1f}%" if pred["confidence"] <= 1 else f"{pred['confidence']:.1f}%")
        with pc2:
            if pred.get("probabilities"):
                probs_df = pd.DataFrame(
                    {"Class": list(pred["probabilities"].keys()),
                     "Probability": list(pred["probabilities"].values())}
                ).set_index("Class")
                st.bar_chart(probs_df)

    if st.session_state.gradcam_result is not None:
        render_section_title("Explainable AI — Grad-CAM")
        gc = st.session_state.gradcam_result
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state.uploaded_image, caption="Original", use_container_width=True)
        with col2:
            display_img = to_display_image(gc.get("image"))
            if display_img is not None:
                st.image(display_img, caption="Grad-CAM Heatmap", use_container_width=True)
            else:
                st.info("Grad-CAM visualization not available.")
        st.markdown(
            f"""<div class="dxai-xai-box">
            💡 <b>What does this mean?</b> Highlighted regions indicate areas that contributed
            to the AI screening result. {gc.get('explanation_text','')}<br>
            <i>This is an explanation aid, not proof of diagnosis.</i>
            </div>""",
            unsafe_allow_html=True,
        )

    if st.session_state.risk_result is not None:
        render_section_title("Risk Assessment & Referral")
        risk = st.session_state.risk_result
        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown(f"**Risk Level:** {safe_badge(risk.get('risk_level'), 'risk')}", unsafe_allow_html=True)
            if risk.get("risk_score"):
                st.metric("Risk Score", risk.get("risk_score"))
        with rc2:
            st.markdown(f"**Referral:** {risk.get('referral', 'N/A')}")
        if risk.get("rationale"):
            st.info(f"**Rationale:** {risk['rationale']}")

        st.markdown("---")
        if st.button("💾 Save Screening", use_container_width=True):
            pred = st.session_state.prediction_result
            gc = st.session_state.gradcam_result
            image_path = save_uploaded_image(st.session_state.uploaded_image, patient_id)
            screening_id = add_screening(
                patient_id=patient_id,
                image_path=image_path or "",
                dr_class=pred["dr_class"],
                confidence=pred["confidence"],
                probabilities=pred.get("probabilities"),
                gradcam_path="",
                risk_level=risk.get("risk_level"),
                risk_score=risk.get("risk_score"),
                risk_rationale=risk.get("rationale"),
                referral=risk.get("referral"),
                screened_by=st.session_state.user.get("full_name") or st.session_state.user.get("username"),
                eye=eye,
            )
            st.success(f"✓ Screening saved successfully (ID: {screening_id}).")
            for key in ["uploaded_image", "quality_result", "prediction_result",
                        "gradcam_result", "risk_result"]:
                st.session_state[key] = None
            st.rerun()


# =============================================================================
# PATIENT RECORDS
# =============================================================================
def page_patient_records():
    render_header()
    render_section_title("Patient Records")

    patients = get_patients()
    if not patients:
        st.info("No patients registered yet.")
        return

    search_col1, search_col2 = st.columns(2)
    with search_col1:
        search_id = st.text_input("🔍 Search by Patient ID")
    with search_col2:
        search_name = st.text_input("🔍 Search by Name")

    filtered = patients
    if search_id:
        filtered = [p for p in filtered if search_id.lower() in str(p["patient_id"]).lower()]
    if search_name:
        filtered = [p for p in filtered if search_name.lower() in str(p["name"]).lower()]

    df = pd.DataFrame(filtered)
    display_cols = [c for c in ["patient_id", "name", "age", "gender", "registered_at"] if c in df.columns]
    st.dataframe(df[display_cols] if not df.empty else df, use_container_width=True, hide_index=True)

    patient_labels = {f"{p['name']} ({p['patient_id']})": p["patient_id"] for p in filtered}
    if patient_labels:
        chosen_label = st.selectbox("View patient details", list(patient_labels.keys()))
        chosen_id = patient_labels[chosen_label]
        render_patient_detail(chosen_id)


def render_patient_detail(patient_id):
    record = get_complete_patient_record(patient_id)
    if not record:
        st.error("Patient record not found.")
        return
    patient = record["patient"]
    screenings = record["screenings"]

    st.markdown('<div class="dxai-card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Name:** {patient['name']}")
    c1.markdown(f"**Patient ID:** {patient['patient_id']}")
    c2.markdown(f"**Age:** {patient.get('age','N/A')}")
    c2.markdown(f"**Gender:** {patient.get('gender','N/A')}")
    c3.markdown(f"**Diabetes Duration:** {patient.get('diabetes_duration','N/A')}")
    c3.markdown(f"**Registered:** {patient.get('registered_at','N/A')}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("**Screening History**")
    if not screenings:
        st.caption("No screenings recorded for this patient yet.")
        return

    for s in screenings:
        review = s.get("doctor_review")
        review_status = review["review_status"] if review else "Not yet reviewed"
        st.markdown(
            f"""<div class="dxai-card">
                <b>{s.get('eye','')}</b> — {safe_badge(s.get('dr_class'), 'dr')}
                {safe_badge(s.get('risk_level'), 'risk')}<br>
                <span style="color:var(--dxai-text-secondary)">Screened: {s.get('screened_at','')} by {s.get('screened_by','')}</span><br>
                Referral: {s.get('referral','N/A')}<br>
                Follow-up: <b>{s.get('followup_status','Pending')}</b> &nbsp;|&nbsp; Doctor Review: <b>{review_status}</b>
            </div>""",
            unsafe_allow_html=True,
        )
        if st.session_state.user.get("role") == "doctor":
            if st.button(f"🖨️ Print Record — {s.get('eye','')} ({s.get('screened_at','')})", key=f"print_{s['id']}"):
                render_print_view(patient, s, review)


# =============================================================================
# DOCTOR REVIEW
# =============================================================================
RECOMMENDATION_LIBRARY = {
    "Specialist Eye Examination": "Get a dilated retinal examination and OCT scan at an eye hospital within 2–4 weeks.",
    "Control Blood Sugar": "Target HbA1c below 7%. Take prescribed medicines/insulin regularly and never skip doses.",
    "Blood Pressure and Cholesterol": "Keep BP under 130/80 mmHg and check lipid profile every 6 months. Both can worsen retinopathy.",
    "Diet & Lifestyle": "High-fibre, low-sugar diet, 30 minutes walking daily, no smoking, limited alcohol.",
    "Watch for Warning Signs": "Sudden blurred vision, floaters, dark spots or eye pain — visit hospital immediately.",
    "Follow-up Screening": "Repeat retinal screening every 3 months until the specialist advises otherwise.",
}


def page_doctor_review():
    if st.session_state.user.get("role") != "doctor":
        st.error("Access restricted to doctors.")
        return

    render_header()
    render_section_title("Doctor Review")

    pending = get_pending_doctor_reviews()
    all_screenings = get_screenings()

    tab1, tab2 = st.tabs(["🕓 Pending Review", "📁 All Screenings"])

    with tab1:
        if not pending:
            st.success("No screenings pending review. 🎉")
        else:
            options = {f"{s.get('patient_name','?')} ({s['patient_id']}) — {s.get('eye','')} — {s['screened_at']}": s["id"]
                       for s in pending}
            chosen = st.selectbox("Select a screening to review", list(options.keys()))
            render_review_form(options[chosen])

    with tab2:
        if all_screenings:
            options = {f"{s['patient_id']} — {s.get('eye','')} — {s['screened_at']}": s["id"] for s in all_screenings}
            chosen2 = st.selectbox("Select any screening", list(options.keys()), key="all_screening_select")
            render_review_form(options[chosen2])
        else:
            st.info("No screenings recorded yet.")


def render_review_form(screening_id):
    screening = get_screening_by_id(screening_id)
    if not screening:
        st.error("Screening not found.")
        return
    patient = get_patient_by_id(screening["patient_id"])
    existing_review = get_doctor_review_by_screening(screening_id)

    st.markdown('<div class="dxai-card">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Patient:** {patient['name'] if patient else screening['patient_id']} ({screening['patient_id']})")
        st.markdown(f"**Eye:** {screening.get('eye','')}")
        st.markdown(f"**Screened:** {screening.get('screened_at','')} by {screening.get('screened_by','')}")
    with c2:
        st.markdown(f"**AI Prediction:** {safe_badge(screening.get('dr_class'), 'dr')}", unsafe_allow_html=True)
        conf = screening.get("confidence") or 0
        st.markdown(f"**Confidence:** {conf*100:.1f}%" if conf <= 1 else f"**Confidence:** {conf:.1f}%")
        st.markdown(f"**Risk Level:** {safe_badge(screening.get('risk_level'), 'risk')}", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if screening.get("probabilities"):
        probs_df = pd.DataFrame(
            {"Class": list(screening["probabilities"].keys()),
             "Probability": list(screening["probabilities"].values())}
        ).set_index("Class")
        st.bar_chart(probs_df)

    image_path = screening.get("image_path")
    if image_path:
        try:
            st.image(image_path, caption="Fundus Image", width=340)
        except Exception:
            st.caption("Stored image could not be loaded.")

    st.markdown(
        f"""<div class="dxai-xai-box">
        💡 Grad-CAM highlighted regions indicate areas that contributed to the AI result.
        Referral suggested: <b>{screening.get('referral','N/A')}</b><br>
        Rationale: {screening.get('risk_rationale','N/A')}
        </div>""",
        unsafe_allow_html=True,
    )

    render_section_title("Doctor Review")
    status_options = ["Pending", "Reviewed", "Confirmed", "Needs Further Examination"]
    default_status = existing_review["review_status"] if existing_review else "Pending"
    review_status = st.selectbox("Doctor Review Status", status_options,
                                  index=status_options.index(default_status) if default_status in status_options else 0,
                                  key=f"status_{screening_id}")

    st.markdown("**Specialist Recommendations** — select all that apply")
    selected_recs = []
    for title, text in RECOMMENDATION_LIBRARY.items():
        checked = st.checkbox(f"{title}", key=f"rec_{screening_id}_{title}")
        st.caption(text)
        if checked:
            selected_recs.append(f"{title}: {text}")

    default_free_text = existing_review["specialist_recommendation"] if existing_review else ""
    free_text_rec = st.text_area("Additional / custom specialist recommendation", value=default_free_text or "",
                                  key=f"freerec_{screening_id}")

    default_advice = existing_review["additional_advice"] if existing_review else ""
    additional_advice = st.text_area("Additional Advice from Doctor", value=default_advice or "", height=140,
                                      key=f"advice_{screening_id}")

    if st.button("💾 Save Doctor Review", key=f"save_review_{screening_id}", use_container_width=True):
        combined_recommendation = "\n".join(selected_recs)
        if free_text_rec:
            combined_recommendation = (combined_recommendation + "\n" + free_text_rec).strip()
        doctor = st.session_state.user
        add_doctor_review(
            screening_id=screening_id,
            doctor_id=doctor.get("username"),
            doctor_name=doctor.get("full_name") or doctor.get("username"),
            review_status=review_status,
            specialist_recommendation=combined_recommendation,
            additional_advice=additional_advice,
        )
        st.success("✓ Doctor review saved successfully.")
        st.rerun()

    if existing_review:
        st.markdown("---")
        if st.button("🖨️ Print Patient Record", key=f"print_review_{screening_id}"):
            render_print_view(patient, screening, get_doctor_review_by_screening(screening_id))


# =============================================================================
# PRINT VIEW
# =============================================================================
def build_print_html(patient, screening, review):
    conf = screening.get("confidence") or 0
    conf_display = f"{conf*100:.1f}%" if conf <= 1 else f"{conf:.1f}%"
    review_html = "<p><i>No doctor review recorded yet.</i></p>"
    if review:
        review_html = f"""
        <p><b>Doctor:</b> {review.get('doctor_name','')}</p>
        <p><b>Status:</b> {review.get('review_status','')}</p>
        <p><b>Specialist Recommendation:</b><br>{(review.get('specialist_recommendation') or '').replace(chr(10), '<br>')}</p>
        <p><b>Additional Advice:</b><br>{(review.get('additional_advice') or '').replace(chr(10), '<br>')}</p>
        <p><b>Reviewed At:</b> {review.get('reviewed_at','')}</p>
        """
    html = f"""
    <html>
    <head><meta charset="utf-8"><title>DRISHTI-XAI Patient Record</title>
    <style>
        body {{ font-family: Arial, sans-serif; color: #0B1F3A; padding: 24px; }}
        h1 {{ color: #0B1F3A; }}
        h2 {{ color: #008C95; border-bottom: 1px solid #E8F6F7; padding-bottom: 4px; }}
        .disclaimer {{ background:#FFF3E0; border:1px solid #D97706; padding:10px; font-size:12px; margin-top:20px;}}
    </style>
    </head>
    <body>
        <h1>👁️ DRISHTI-XAI</h1>
        <p>Explainable AI Diabetic Retinopathy Screening</p>

        <h2>Patient Information</h2>
        <p><b>Patient ID:</b> {patient.get('patient_id','')}</p>
        <p><b>Name:</b> {patient.get('name','')}</p>
        <p><b>Age:</b> {patient.get('age','')}</p>
        <p><b>Diabetes Duration:</b> {patient.get('diabetes_duration','')}</p>
        <p><b>Previous Screening:</b> {patient.get('previous_screening','')}</p>

        <h2>Screening Information</h2>
        <p><b>Date:</b> {screening.get('screened_at','')}</p>
        <p><b>Eye:</b> {screening.get('eye','')}</p>
        <p><b>DR Class:</b> {screening.get('dr_class','')}</p>
        <p><b>AI Confidence:</b> {conf_display}</p>
        <p><b>Risk Level:</b> {screening.get('risk_level','')}</p>
        <p><b>Referral Recommendation:</b> {screening.get('referral','')}</p>

        <h2>Explainable AI</h2>
        <p>Grad-CAM highlighted regions indicate areas that contributed to the AI result.
        This is an explanation aid, not proof of diagnosis.</p>

        <h2>Doctor Review</h2>
        {review_html}

        <h2>Follow-up Status</h2>
        <p>{screening.get('followup_status','Pending')}</p>

        <div class="disclaimer">
            PROTOTYPE — AI-assisted screening aid only. Not a certified medical diagnostic
            device or an official medical certificate.
        </div>
    </body>
    </html>
    """
    return html


def render_print_view(patient, screening, review):
    html = build_print_html(patient, screening, review)
    components.html(
        html + """<script>window.print();</script>""",
        height=200,
        scrolling=True,
    )
    b64 = base64.b64encode(html.encode("utf-8")).decode()
    st.markdown(
        f'<a download="patient_record_{patient.get("patient_id","")}.html" '
        f'href="data:text/html;base64,{b64}">⬇️ Download Patient Record (HTML)</a>',
        unsafe_allow_html=True,
    )


# =============================================================================
# FOLLOW-UP
# =============================================================================
def page_followup():
    render_header()
    render_section_title("Follow-up Management")

    screenings = get_screenings()
    is_doctor = st.session_state.user.get("role") == "doctor"

    statuses = FOLLOWUP_STATUSES if FOLLOWUP_STATUSES else \
        ["Pending", "Contacted", "Appointment Scheduled", "Completed", "Not Reachable"]

    if screenings:
        df = pd.DataFrame(screenings)
        counts = df["followup_status"].fillna("Pending").value_counts()
        st.bar_chart(counts)

    for s in screenings:
        patient = get_patient_by_id(s["patient_id"])
        pname = patient["name"] if patient else s["patient_id"]
        st.markdown('<div class="dxai-card">', unsafe_allow_html=True)
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"**{pname}** ({s['patient_id']}) — {s.get('eye','')}")
            st.markdown(f"{safe_badge(s.get('risk_level'), 'risk')}", unsafe_allow_html=True)
            st.caption(f"Screened: {s.get('screened_at','')}")
        with c2:
            if is_doctor:
                current = s.get("followup_status", "Pending")
                new_status = st.selectbox("Status", statuses,
                                           index=statuses.index(current) if current in statuses else 0,
                                           key=f"fu_{s['id']}")
                if new_status != current:
                    if st.button("Update", key=f"fu_btn_{s['id']}"):
                        update_followup_status(s["id"], new_status)
                        st.success("Follow-up status updated.")
                        st.rerun()
            else:
                st.markdown(f"**{s.get('followup_status','Pending')}**")
        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# ANALYTICS
# =============================================================================
def page_analytics():
    render_header()
    render_section_title("Analytics")

    patients = get_patients()
    screenings = get_screenings()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Patients", len(patients))
    c2.metric("Total Screenings", len(screenings))
    if screenings:
        high_pct = 100 * sum(1 for s in screenings if str(s.get("risk_level", "")).lower() in ("high", "critical")) / len(screenings)
        avg_conf = sum((s.get("confidence") or 0) for s in screenings) / len(screenings)
        avg_conf_display = f"{avg_conf*100:.1f}%" if avg_conf <= 1 else f"{avg_conf:.1f}%"
    else:
        high_pct = 0
        avg_conf_display = "N/A"
    c3.metric("High-Risk %", f"{high_pct:.1f}%")

    st.metric("Average AI Confidence", avg_conf_display)

    if screenings:
        df = pd.DataFrame(screenings)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**DR Class Distribution**")
            st.bar_chart(df["dr_class"].fillna("Unknown").value_counts())
        with col2:
            st.markdown("**Risk Distribution**")
            st.bar_chart(df["risk_level"].fillna("Unknown").value_counts())
        st.markdown("**Follow-up Status**")
        st.bar_chart(df["followup_status"].fillna("Pending").value_counts())
    else:
        st.info("No screening data available yet for analytics.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    init_session_state()
    inject_css()

    if not st.session_state.logged_in:
        login_page()
        return

    render_sidebar()

    page = st.session_state.page
    try:
        if page == "Dashboard":
            page_dashboard()
        elif page == "Register Patient":
            page_register_patient()
        elif page == "New Screening":
            page_new_screening()
        elif page == "Patient Records":
            page_patient_records()
        elif page == "Doctor Review":
            page_doctor_review()
        elif page == "Follow-up":
            page_followup()
        elif page == "Analytics":
            page_analytics()
        else:
            page_dashboard()
    except Exception as e:
        st.error("Something went wrong while loading this page.")
        with st.expander("Technical details"):
            st.exception(e)


if __name__ == "__main__":
    main()
