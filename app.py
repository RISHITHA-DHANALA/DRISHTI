"""
app.py
------
DRISHTI-XAI — Explainable AI Diabetic Retinopathy Screening
for Rural India

Main Streamlit entry point.

Workflow:
    Login
      ↓
    Patient Registration
      ↓
    Fundus Image Upload
      ↓
    Image Quality Check
      ↓
    DR Prediction
      ↓
    Explainable AI / Grad-CAM
      ↓
    Risk Assessment
      ↓
    Smart Referral
      ↓
    Doctor Dashboard
      ↓
    Follow-up Tracking

Run with:
    streamlit run app.py

*** PROTOTYPE DISCLAIMER ***
This application is an AI-ASSISTED SCREENING PROTOTYPE built for
a hackathon (SIH 2026) context.

It is NOT a certified medical device and has NOT undergone
clinical validation.

Do not use it for real patient diagnosis or treatment decisions.
"""

import os
import sys

import streamlit as st
import pandas as pd

from PIL import Image


# ============================================================================
# PATH SETUP
# ============================================================================

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================================
# PROJECT IMPORTS
# ============================================================================

from config import (  # noqa: E402
    APP_NAME,
    APP_TAGLINE,
    IS_PROTOTYPE,
    DR_CLASSES,
    SUPPORTED_LANGUAGES,
    FOLLOWUP_STATUSES,
    SAMPLE_IMAGES_DIR,
    DATA_DIR,
)

from i18n.translations import t  # noqa: E402

from db.database import (  # noqa: E402
    init_db,
    authenticate,
    add_patient,
    get_patients,
    get_patient_by_id,
    add_screening,
    get_screenings,
    update_followup_status,
)

from core.image_quality import check_image_quality  # noqa: E402
from core.model import DRModel  # noqa: E402
from core.gradcam import generate_explanation  # noqa: E402
from core.risk_engine import assess_risk  # noqa: E402

from utils.demo_data import (  # noqa: E402
    generate_all_samples,
    seed_demo_patients,
)

from utils.helpers import (  # noqa: E402
    pil_to_bgr,
    bgr_to_rgb,
    save_uploaded_image,
    risk_badge,
    dr_class_badge,
)


# ============================================================================
# DIRECTORIES
# ============================================================================

UPLOAD_DIR = os.path.join(
    DATA_DIR,
    "uploads"
)

HEATMAP_DIR = os.path.join(
    DATA_DIR,
    "heatmaps"
)


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# APPLICATION BOOTSTRAP
# ============================================================================

@st.cache_resource
def bootstrap():
    """
    Initialize database and demo data.
    """
    init_db()
    generate_all_samples()
    seed_demo_patients()

    return True


@st.cache_resource
def load_model():
    """
    Load and cache the DR model.
    """
    return DRModel()


bootstrap()

dr_model = load_model()


# ============================================================================
# HOSPITAL / MEDICAL UI THEME
# ============================================================================

st.markdown(
    """
<style>

/* =========================================================
   GLOBAL COLORS
   ========================================================= */

:root {
    --navy: #0B1F3A;
    --teal: #008C95;
    --teal-dark: #006B73;
    --teal-light: #E8F6F7;
    --background: #F4F8FA;
    --white: #FFFFFF;
    --text: #263238;
    --muted: #607D8B;
    --border: #D9E3E8;
    --warning: #D97706;
    --danger: #C62828;
    --success: #16803C;
}


/* =========================================================
   MAIN BACKGROUND
   ========================================================= */

.stApp {
    background-color: var(--background);
}


/* =========================================================
   MAIN CONTENT
   ========================================================= */

.main {
    background-color: var(--background) !important;
}


.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}


/* =========================================================
   HEADINGS
   ========================================================= */

h1,
h2,
h3,
h4,
h5,
h6 {
    color: var(--navy) !important;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"] {
    background-color: var(--navy) !important;
}


section[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}


section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.20) !important;
}


/* Sidebar select boxes */

section[data-testid="stSidebar"]
div[data-baseweb="select"] {
    background-color: #FFFFFF !important;
}


section[data-testid="stSidebar"]
div[data-baseweb="select"] * {
    color: #263238 !important;
}


/* =========================================================
   DRISHTI CARD
   ========================================================= */

.drishti-card {
    background-color: var(--white) !important;

    border: 1px solid var(--border) !important;

    border-radius: 14px !important;

    padding: 20px 22px !important;

    margin-bottom: 16px !important;

    box-shadow:
        0 2px 8px rgba(11,31,58,0.05) !important;

    color: var(--text) !important;
}


/* Text inside cards */

.drishti-card p,
.drishti-card span,
.drishti-card label,
.drishti-card strong {
    color: var(--text) !important;
}


/* Card headings */

.drishti-card h1,
.drishti-card h2,
.drishti-card h3,
.drishti-card h4,
.drishti-card h5,
.drishti-card h6 {
    color: var(--navy) !important;
}


/* =========================================================
   METRIC CARDS
   ========================================================= */

div[data-testid="stMetric"] {
    background-color: #FFFFFF !important;

    border: 1px solid var(--border) !important;

    border-radius: 14px !important;

    padding: 16px !important;

    box-shadow:
        0 2px 8px rgba(11,31,58,0.05) !important;
}


div[data-testid="stMetric"] label,
div[data-testid="stMetric"] label *,
div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricValue"] * {
    color: var(--navy) !important;
}


div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: var(--teal-dark) !important;
}


/* =========================================================
   PRIMARY BUTTONS
   ========================================================= */

.stButton > button[kind="primary"],
button[data-testid="baseButton-primary"] {
    background-color: var(--teal) !important;

    color: #FFFFFF !important;

    border: none !important;

    border-radius: 9px !important;

    font-weight: 600 !important;

    padding: 0.55rem 1rem !important;
}


.stButton > button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover {
    background-color: var(--teal-dark) !important;

    color: #FFFFFF !important;
}


/* =========================================================
   NORMAL BUTTONS
   ========================================================= */

.stButton > button {
    border-radius: 9px !important;

    border: 1px solid var(--border) !important;

    color: var(--navy) !important;

    background-color: #FFFFFF !important;

    font-weight: 500 !important;
}


.stButton > button:hover {
    border-color: var(--teal) !important;

    color: var(--teal-dark) !important;
}


/* =========================================================
   INPUT FIELDS
   ========================================================= */

input,
textarea {
    background-color: #FFFFFF !important;

    color: var(--text) !important;

    border-radius: 8px !important;
}


/* =========================================================
   SELECT BOX
   ========================================================= */

div[data-baseweb="select"] {
    background-color: #FFFFFF !important;

    border-radius: 8px !important;
}


div[data-baseweb="select"] * {
    color: var(--text) !important;
}


/* =========================================================
   FILE UPLOADER
   ========================================================= */

[data-testid="stFileUploader"] {
    background-color: #FFFFFF !important;

    border: 1px dashed var(--teal) !important;

    border-radius: 12px !important;

    padding: 10px !important;
}


/* =========================================================
   INFORMATION BOX
   ========================================================= */

div[data-testid="stAlert"] {
    border-radius: 10px !important;
}


/* =========================================================
   PROTOTYPE DISCLAIMER
   ========================================================= */

.drishti-banner {
    background-color: #FFF7E6 !important;

    border: 1px solid #F5D08A !important;

    border-radius: 10px !important;

    padding: 11px 16px !important;

    margin-bottom: 16px !important;

    color: #7A5200 !important;

    font-size: 0.88rem !important;
}


.drishti-banner * {
    color: #7A5200 !important;
}


/* =========================================================
   HEALTHCARE HEADER
   ========================================================= */

.drishti-header {
    background: linear-gradient(
        135deg,
        #0B1F3A,
        #008C95
    );

    border-radius: 16px;

    padding: 24px 28px;

    margin-bottom: 22px;

    color: white;

    box-shadow:
        0 4px 14px rgba(11,31,58,0.12);
}


.drishti-header h1,
.drishti-header h2,
.drishti-header h3,
.drishti-header p,
.drishti-header span {
    color: white !important;
}


/* =========================================================
   SECTION TITLE
   ========================================================= */

.section-title {
    color: var(--navy);

    font-size: 1.15rem;

    font-weight: 700;

    margin-top: 10px;

    margin-bottom: 12px;
}


/* =========================================================
   STEP BADGE
   ========================================================= */

.step-badge {
    display: inline-block;

    background-color: var(--teal-light);

    color: var(--teal-dark);

    padding: 5px 11px;

    border-radius: 20px;

    font-size: 0.82rem;

    font-weight: 700;

    margin-bottom: 8px;
}


/* =========================================================
   TABLE
   ========================================================= */

div[data-testid="stDataFrame"] {
    background-color: #FFFFFF !important;

    border-radius: 10px !important;
}


/* =========================================================
   EXPANDER
   ========================================================= */

div[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;

    border-radius: 10px !important;

    background-color: #FFFFFF !important;
}


/* =========================================================
   TABS
   ========================================================= */

button[data-baseweb="tab"] {
    color: var(--muted) !important;

    font-weight: 600 !important;
}


button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--teal-dark) !important;
}


/* =========================================================
   DIVIDER
   ========================================================= */

hr {
    border-color: var(--border) !important;
}


/* =========================================================
   CAPTIONS
   ========================================================= */

.stCaption {
    color: var(--muted) !important;
}


/* =========================================================
   LOGIN CARD
   ========================================================= */

.login-card {
    background-color: #FFFFFF;

    border: 1px solid var(--border);

    border-radius: 18px;

    padding: 30px;

    box-shadow:
        0 8px 25px rgba(11,31,58,0.08);
}


/* =========================================================
   FOOTER
   ========================================================= */

.drishti-footer {
    text-align: center;

    color: var(--muted);

    font-size: 0.78rem;

    margin-top: 30px;

    padding-top: 15px;

    border-top: 1px solid var(--border);
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================================
# SESSION STATE
# ============================================================================

_defaults = {

    "logged_in": False,

    "user": None,

    "lang": "en",

    "page": "Register",

    "wf_image_bgr": None,

    "wf_image_source": None,

    "wf_quality": None,

    "wf_prediction": None,

    "wf_heatmap": None,

    "wf_risk": None,

    "wf_patient_id": None,
}


for key, value in _defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================================
# RESET SCREENING WORKFLOW
# ============================================================================

def reset_workflow():

    workflow_keys = [

        "wf_image_bgr",

        "wf_image_source",

        "wf_quality",

        "wf_prediction",

        "wf_heatmap",

        "wf_risk",

        "wf_patient_id",
    ]

    for key in workflow_keys:

        st.session_state[key] = None


# ============================================================================
# APPLICATION HEADER
# ============================================================================

def show_header(title, subtitle=None):

    if subtitle is None:

        subtitle = APP_TAGLINE

    st.markdown(
        f"""
        <div class="drishti-header">

            <h1>
                👁️ {title}
            </h1>

            <p>
                {subtitle}
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================================
# LOGIN PAGE
# ============================================================================

def login_page():

    lang = st.session_state.lang

    col_logo, col_lang = st.columns(
        [4, 1]
    )

    with col_logo:

        st.markdown(
            f"""
            <div style="
                padding: 10px 0 5px 0;
            ">

                <h1 style="
                    color:#0B1F3A;
                    margin-bottom:0;
                ">
                    👁️ {t('app_title', lang)}
                </h1>

                <p style="
                    color:#607D8B;
                    margin-top:4px;
                ">
                    {t('app_tagline', lang)}
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col_lang:

        chosen = st.selectbox(
            t(
                "select_language",
                lang
            ),
            list(
                SUPPORTED_LANGUAGES.keys()
            ),
            index=0
        )

        st.session_state.lang = (
            SUPPORTED_LANGUAGES[chosen]
        )

        lang = st.session_state.lang


    # Prototype warning

    st.markdown(
        f"""
        <div class="drishti-banner">
            ⚠️ {t('prototype_banner', lang)}
        </div>
        """,
        unsafe_allow_html=True
    )


    # Login card

    _, mid, _ = st.columns(
        [1, 1.2, 1]
    )

    with mid:

        st.markdown(
            '<div class="login-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div style="
                text-align:center;
                margin-bottom:20px;
            ">

                <div style="
                    font-size:50px;
                ">
                    🏥
                </div>

                <h2 style="
                    color:#0B1F3A;
                    margin-bottom:5px;
                ">
                    DRISHTI-XAI
                </h2>

                <p style="
                    color:#607D8B;
                ">
                    AI-assisted retinal screening
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


        st.subheader(
            t(
                "login_title",
                lang
            )
        )


        username = st.text_input(
            t(
                "username",
                lang
            ),
            key="login_user"
        )


        password = st.text_input(
            t(
                "password",
                lang
            ),
            type="password",
            key="login_pass"
        )


        if st.button(
            t(
                "login_button",
                lang
            ),
            use_container_width=True,
            type="primary"
        ):

            user = authenticate(
                username,
                password
            )


            if user:

                st.session_state.logged_in = True

                st.session_state.user = user

                st.session_state.page = "Register"

                st.rerun()

            else:

                st.error(
                    t(
                        "login_error",
                        lang
                    )
                )


        st.caption(
            "Demo accounts — "
            "Health Worker: `healthworker1` / `worker123`  ·  "
            "Doctor: `doctor1` / `doctor123`"
        )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

def sidebar_nav():

    lang = st.session_state.lang

    user = st.session_state.user


    with st.sidebar:

        st.markdown(
            """
            <div style="
                text-align:center;
                padding:10px 0 5px 0;
            ">

                <div style="
                    font-size:38px;
                ">
                    👁️
                </div>

                <h2 style="
                    color:white !important;
                    margin:0;
                ">
                    DRISHTI-XAI
                </h2>

                <p style="
                    color:#B2DFDB !important;
                    font-size:0.8rem;
                ">
                    Explainable AI Screening
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


        st.divider()


        st.markdown(
            f"""
            <div style="
                background:rgba(255,255,255,0.08);
                padding:12px;
                border-radius:10px;
                margin-bottom:10px;
            ">

                <strong>
                    {t('welcome', lang)},
                    {user['display_name']}
                </strong>

                <br>

                <span style="
                    font-size:0.8rem;
                    color:#B2DFDB !important;
                ">
                    {t('role', lang)}:
                    {user['role']}
                </span>

            </div>
            """,
            unsafe_allow_html=True
        )


        # Language

        chosen = st.selectbox(
            t(
                "select_language",
                lang
            ),
            list(
                SUPPORTED_LANGUAGES.keys()
            ),
            index=list(
                SUPPORTED_LANGUAGES.values()
            ).index(lang),
        )


        st.session_state.lang = (
            SUPPORTED_LANGUAGES[chosen]
        )


        lang = st.session_state.lang


        st.divider()


        # Navigation

        nav_options = {

            "🏠 Home / Dashboard":
                "Dashboard",

            t(
                "nav_register",
                lang
            ):
                "Register",

            t(
                "nav_upload",
                lang
            ):
                "Screening",
        }


        choice_label = st.radio(
            "Navigation",
            list(
                nav_options.keys()
            ),
            label_visibility="collapsed"
        )


        st.session_state.page = (
            nav_options[choice_label]
        )


        st.divider()


        # Model status

        st.markdown(
            "**System Status**"
        )


        if dr_model.mode == "mock":

            st.caption(
                "🧪 Demo AI model active"
            )

        else:

            st.caption(
                "✅ AI model loaded"
            )


        st.caption(
            "🏥 Rural screening workflow"
        )


        st.divider()


        # Logout

        if st.button(
            "↩️ " +
            t(
                "logout",
                lang
            ),
            use_container_width=True
        ):

            st.session_state.logged_in = False

            st.session_state.user = None

            reset_workflow()

            st.rerun()


# ============================================================================
# PAGE: HOME / DASHBOARD
# ============================================================================

def page_home():

    lang = st.session_state.lang

    screenings = get_screenings()

    patients = get_patients()


    total_patients = len(patients)

    total_screenings = len(screenings)


    if screenings:

        df = pd.DataFrame(screenings)


        high_risk = int(
            df["risk_level"].isin(
                [
                    "High",
                    "Critical"
                ]
            ).sum()
        )


        pending_followup = int(
            (
                df["followup_status"]
                == "Pending"
            ).sum()
        )


        avg_confidence = float(
            df["confidence"].mean()
        )

    else:

        df = pd.DataFrame()

        high_risk = 0

        pending_followup = 0

        avg_confidence = 0.0


    # Header

    show_header(
        "DRISHTI-XAI",
        "Explainable AI for Diabetic Retinopathy Screening in Rural India"
    )


    # Workflow

    st.markdown(
        """
        <div class="drishti-card">

            <h3>
                🏥 PHC → 🤖 AI Screening → 👨‍⚕️ Specialist
            </h3>

            <p>
                Early detection and timely referral
                for patients at risk of diabetic retinopathy.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # KPI

    st.markdown(
        '<div class="section-title">'
        '📊 Screening Overview'
        '</div>',
        unsafe_allow_html=True
    )


    k1, k2, k3, k4, k5 = st.columns(5)


    k1.metric(
        "👥 Total Patients",
        total_patients
    )


    k2.metric(
        "🔬 Total Screenings",
        total_screenings
    )


    k3.metric(
        "⚠️ High/Critical Risk",
        high_risk
    )


    k4.metric(
        "📅 Pending Follow-up",
        pending_followup
    )


    k5.metric(
        "🤖 Avg. AI Confidence",
        f"{avg_confidence:.1f}%"
    )


    st.divider()


    # Quick actions

    st.markdown(
        '<div class="section-title">'
        '⚡ Quick Actions'
        '</div>',
        unsafe_allow_html=True
    )


    q1, q2, q3 = st.columns(3)


    with q1:

        if st.button(
            "👤 Register Patient",
            use_container_width=True,
            type="primary"
        ):

            st.session_state.page = "Register"

            st.rerun()


    with q2:

        if st.button(
            "🔬 Start Screening",
            use_container_width=True,
            type="primary"
        ):

            st.session_state.page = "Screening"

            st.rerun()


    with q3:

        st.info(
            "💡 Use the sidebar to access "
            "patient records and analytics."
        )


    # Analytics

    if screenings:

        st.divider()

        st.markdown(
            '<div class="section-title">'
            '📈 Screening Analytics'
            '</div>',
            unsafe_allow_html=True
        )


        c1, c2 = st.columns(2)


        with c1:

            st.markdown(
                "<div class='drishti-card'>",
                unsafe_allow_html=True
            )

            st.markdown(
                "### 🧠 DR Severity Distribution"
            )

            st.bar_chart(
                df[
                    "dr_class"
                ].value_counts()
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


        with c2:

            st.markdown(
                "<div class='drishti-card'>",
                unsafe_allow_html=True
            )

            st.markdown(
                "### ⚠️ Risk Distribution"
            )

            st.bar_chart(
                df[
                    "risk_level"
                ].value_counts()
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


        # Recent screenings

        st.markdown(
            "<div class='drishti-card'>",
            unsafe_allow_html=True
        )

        st.markdown(
            "### 📋 Recent Screening Activity"
        )


        show_df = df[
            [
                "patient_code",
                "patient_name",
                "age",
                "dr_class",
                "confidence",
                "risk_level",
                "referral",
                "followup_status",
                "screened_by",
                "screened_at",
            ]
        ].rename(
            columns={
                "patient_code":
                    "Patient ID",

                "patient_name":
                    "Name",

                "age":
                    "Age",

                "dr_class":
                    "DR Class",

                "confidence":
                    "Confidence (%)",

                "risk_level":
                    "Risk",

                "referral":
                    "Referral",

                "followup_status":
                    "Follow-up",

                "screened_by":
                    "Screened By",

                "screened_at":
                    "Screened At",
            }
        )


        st.dataframe(
            show_df,
            use_container_width=True,
            hide_index=True
        )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    else:

        st.markdown(
            """
            <div class="drishti-card">

                <h3>
                    👁️ Start your first screening
                </h3>

                <p>
                    Register a patient and upload
                    a fundus image to begin AI-assisted
                    diabetic retinopathy screening.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================================
# PAGE: PATIENT REGISTRATION
# ============================================================================

def page_register():

    lang = st.session_state.lang


    show_header(
        f"👤 {t('nav_register', lang)}",
        "Register and manage patients before retinal screening."
    )


    col1, col2 = st.columns(
        [1, 1.3]
    )


    # ------------------------------------------------------------------------
    # Registration form
    # ------------------------------------------------------------------------

    with col1:

        st.markdown(
            "<div class='drishti-card'>",
            unsafe_allow_html=True
        )


        st.markdown(
            "### 👤 Patient Information"
        )


        with st.form(
            "register_form",
            clear_on_submit=True
        ):

            patient_code = st.text_input(
                t(
                    "patient_id",
                    lang
                ),
                placeholder="e.g. PT-1005"
            )


            name = st.text_input(
                t(
                    "patient_name",
                    lang
                )
            )


            age = st.number_input(
                t(
                    "age",
                    lang
                ),
                min_value=1,
                max_value=120,
                value=45
            )


            duration = st.number_input(
                t(
                    "diabetes_duration",
                    lang
                ),
                min_value=0,
                max_value=70,
                value=5
            )


            prev_screen = st.selectbox(
                t(
                    "previous_screening",
                    lang
                ),
                [
                    "No",
                    "Yes"
                ]
            )


            submitted = st.form_submit_button(
                t(
                    "register_button",
                    lang
                ),
                type="primary",
                use_container_width=True
            )


            if submitted:

                if not patient_code or not name:

                    st.error(
                        "Patient ID and Name are required."
                    )

                else:

                    add_patient(
                        patient_code,
                        name,
                        int(age),
                        int(duration),
                        prev_screen,
                        st.session_state.user[
                            "display_name"
                        ]
                    )


                    st.success(
                        t(
                            "registered_success",
                            lang
                        )
                    )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # ------------------------------------------------------------------------
    # Patient list
    # ------------------------------------------------------------------------

    with col2:

        st.markdown(
            "<div class='drishti-card'>",
            unsafe_allow_html=True
        )


        st.markdown(
            "### 👥 Registered Patients"
        )


        patients = get_patients()


        if not patients:

            st.info(
                t(
                    "no_patients",
                    lang
                )
            )

        else:

            df = pd.DataFrame(
                patients
            )


            columns = [
                "patient_code",
                "name",
                "age",
                "diabetes_duration",
                "previous_screening",
                "registered_at",
            ]


            df = df[
                [
                    c for c in columns
                    if c in df.columns
                ]
            ]


            df.columns = [
                "Patient ID",
                "Name",
                "Age",
                "DM Duration (y)",
                "Prev. Screening",
                "Registered At",
            ][:len(df.columns)]


            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


# ============================================================================
# PAGE: SCREENING
# ============================================================================

def page_screening():

    lang = st.session_state.lang


    show_header(
        f"🔬 {t('nav_upload', lang)}",
        "Fundus image quality check, AI screening, explanation and referral."
    )


    patients = get_patients()


    if not patients:

        st.warning(
            t(
                "no_patients",
                lang
            )
            +
            " Please register a patient first."
        )

        return


    # ------------------------------------------------------------------------
    # Patient selection
    # ------------------------------------------------------------------------

    patient_labels = {

        f"{p['patient_code']} — {p['name']}":
            p["id"]

        for p in patients
    }


    chosen_label = st.selectbox(
        t(
            "select_patient",
            lang
        ),
        list(
            patient_labels.keys()
        )
    )


    patient_id = patient_labels[
        chosen_label
    ]


    patient = get_patient_by_id(
        patient_id
    )


    if (
        st.session_state.wf_patient_id
        != patient_id
    ):

        reset_workflow()

        st.session_state.wf_patient_id = (
            patient_id
        )


    # ------------------------------------------------------------------------
    # Patient summary
    # ------------------------------------------------------------------------

    st.markdown(
        "<div class='drishti-card'>",
        unsafe_allow_html=True
    )


    st.markdown(
        f"""
        ### 👤 Patient Summary

        **Patient:** {patient['name']}

        **Age:** {patient['age']}

        **Diabetes duration:**
        {patient['diabetes_duration']} years

        **Previous screening:**
        {patient['previous_screening']}
        """
    )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


    # ------------------------------------------------------------------------
    # Step 1 — Image upload
    # ------------------------------------------------------------------------

    st.markdown(
        '<span class="step-badge">'
        'STEP 1'
        '</span>',
        unsafe_allow_html=True
    )


    st.subheader(
        t(
            "upload_image",
            lang
        )
    )


    col_up, col_sample = st.columns(
        [2, 1]
    )


    with col_up:

        uploaded_file = st.file_uploader(
            "Upload fundus image",
            type=[
                "jpg",
                "jpeg",
                "png"
            ]
        )


        if uploaded_file is not None:

            pil_img = Image.open(
                uploaded_file
            )


            st.session_state.wf_image_bgr = (
                pil_to_bgr(
                    pil_img
                )
            )


            st.session_state.wf_image_source = (
                "upload"
            )


            st.session_state.wf_quality = None

            st.session_state.wf_prediction = None

            st.session_state.wf_heatmap = None

            st.session_state.wf_risk = None


    with col_sample:

        st.caption(
            t(
                "use_sample",
                lang
            )
        )


        for severity in DR_CLASSES:

            fname = (
                "sample_"
                +
                severity.replace(
                    " ",
                    "_"
                ).lower()
                +
                ".png"
            )


            fpath = os.path.join(
                SAMPLE_IMAGES_DIR,
                fname
            )


            if st.button(
                f"Use: {severity} sample",
                key=f"sample_{severity}",
                use_container_width=True
            ):

                pil_img = Image.open(
                    fpath
                )


                st.session_state.wf_image_bgr = (
                    pil_to_bgr(
                        pil_img
                    )
                )


                st.session_state.wf_image_source = (
                    f"sample:{severity}"
                )


                st.session_state.wf_quality = None

                st.session_state.wf_prediction = None

                st.session_state.wf_heatmap = None

                st.session_state.wf_risk = None


    if (
        st.session_state.wf_image_bgr
        is None
    ):

        st.info(
            "Upload a fundus image or select a demo sample to begin."
        )

        return


    # Image preview

    st.markdown(
        "<div class='drishti-card'>",
        unsafe_allow_html=True
    )


    st.markdown(
        "### 👁️ Fundus Image Preview"
    )


    st.image(
        bgr_to_rgb(
            st.session_state.wf_image_bgr
        ),
        caption=t(
            "image_preview",
            lang
        ),
        width=360
    )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


    # ------------------------------------------------------------------------
    # Step 2 — Quality check
    # ------------------------------------------------------------------------

    st.markdown(
        '<span class="step-badge">'
        'STEP 2'
        '</span>',
        unsafe_allow_html=True
    )


    st.subheader(
        t(
            "quality_check",
            lang
        )
    )


    if st.button(
        "Run Quality Check",
        type="primary"
    ):

        st.session_state.wf_quality = (
            check_image_quality(
                st.session_state.wf_image_bgr
            )
        )


    quality = (
        st.session_state.wf_quality
    )


    if quality:

        cols = st.columns(3)


        cols[0].metric(
            "Blur Score",
            quality["blur_score"]
        )


        cols[1].metric(
            "Brightness",
            quality["brightness"]
        )


        cols[2].metric(
            "Resolution",
            f"{quality['resolution'][0]}"
            f" × "
            f"{quality['resolution'][1]}"
        )


        if quality["passed"]:

            st.success(
                t(
                    "quality_pass",
                    lang
                )
            )

        else:

            st.error(
                t(
                    "quality_fail",
                    lang
                )
            )


            for issue in quality["issues"]:

                st.write(
                    f"- {t(issue, lang)}"
                )


            st.stop()


    if (
        not quality
        or not quality["passed"]
    ):

        return


    # ------------------------------------------------------------------------
    # Step 3 — AI Screening
    # ------------------------------------------------------------------------

    st.markdown(
        '<span class="step-badge">'
        'STEP 3'
        '</span>',
        unsafe_allow_html=True
    )


    st.subheader(
        t(
            "run_screening",
            lang
        )
    )


    if st.button(
        t(
            "run_screening",
            lang
        ),
        type="primary"
    ):

        with st.spinner(
            "Running AI screening..."
        ):

            dr_class, confidence, probs = (
                dr_model.predict(
                    st.session_state.wf_image_bgr
                )
            )


            st.session_state.wf_prediction = {

                "dr_class":
                    dr_class,

                "confidence":
                    confidence,

                "probs":
                    probs,
            }


            st.session_state.wf_heatmap = (
                generate_explanation(
                    dr_model,
                    st.session_state.wf_image_bgr
                )
            )


            risk_result = assess_risk(
                dr_class,
                confidence,
                patient[
                    "diabetes_duration"
                ],
                patient[
                    "previous_screening"
                ]
            )


            st.session_state.wf_risk = (
                risk_result
            )


    prediction = (
        st.session_state.wf_prediction
    )


    if not prediction:

        return


    if dr_model.mode == "mock":

        st.caption(
            "🧪 "
            +
            t(
                "demo_mode_notice",
                lang
            )
        )


    # ------------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------------

    col_res, col_cam = st.columns(
        2
    )


    # Prediction result

    with col_res:

        st.markdown(
            "<div class='drishti-card'>",
            unsafe_allow_html=True
        )


        st.markdown(
            "### 🤖 AI Screening Result"
        )


        dr_class_badge(
            prediction[
                "dr_class"
            ]
        )


        st.metric(
            t(
                "confidence",
                lang
            ),
            f"{prediction['confidence']}%"
        )


        probs_df = pd.DataFrame(
            {
                "DR Class":
                    list(
                        prediction[
                            "probs"
                        ].keys()
                    ),

                "Probability":
                    list(
                        prediction[
                            "probs"
                        ].values()
                    ),
            }
        )


        st.markdown(
            "#### Probability Distribution"
        )


        st.bar_chart(
            probs_df.set_index(
                "DR Class"
            )
        )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # Grad-CAM

    with col_cam:

        st.markdown(
            "<div class='drishti-card'>",
            unsafe_allow_html=True
        )


        st.markdown(
            "### 🔥 Explainable AI"
        )


        st.caption(
            "Grad-CAM highlights image regions "
            "that influenced the AI prediction."
        )


        st.image(
            bgr_to_rgb(
                st.session_state.wf_heatmap
            ),
            caption=t(
                "gradcam_caption",
                lang
            ),
            use_container_width=True
        )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # ------------------------------------------------------------------------
    # Step 4 — Risk and referral
    # ------------------------------------------------------------------------

    risk = (
        st.session_state.wf_risk
    )


    st.markdown(
        '<span class="step-badge">'
        'STEP 4'
        '</span>',
        unsafe_allow_html=True
    )


    st.markdown(
        "<div class='drishti-card'>",
        unsafe_allow_html=True
    )


    st.markdown(
        "### ⚠️ Risk Assessment & Smart Referral"
    )


    c1, c2 = st.columns(
        [1, 2]
    )


    with c1:

        risk_badge(
            risk[
                "risk_level"
            ]
        )


    with c2:

        st.write(
            f"**Referral recommendation:** "
            f"{risk['referral']}"
        )


    with st.expander(
        "Why this risk level?"
    ):

        for line in risk[
            "rationale"
        ]:

            st.write(
                f"- {line}"
            )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


    # ------------------------------------------------------------------------
    # Step 5 — Save
    # ------------------------------------------------------------------------

    st.markdown(
        '<span class="step-badge">'
        'STEP 5'
        '</span>',
        unsafe_allow_html=True
    )


    st.subheader(
        "Save Screening Result"
    )


    if st.button(
        t(
            "save_result",
            lang
        ),
        type="primary",
        use_container_width=True
    ):

        image_path = (
            save_uploaded_image(
                st.session_state.wf_image_bgr,
                UPLOAD_DIR
            )
        )


        heatmap_path = (
            save_uploaded_image(
                st.session_state.wf_heatmap,
                HEATMAP_DIR
            )
        )


        add_screening(
            patient_id=patient_id,

            image_path=image_path,

            dr_class=prediction[
                "dr_class"
            ],

            confidence=prediction[
                "confidence"
            ],

            risk_level=risk[
                "risk_level"
            ],

            referral=risk[
                "referral"
            ],

            heatmap_path=heatmap_path,

            screened_by=st.session_state.user[
                "display_name"
            ],
        )


        st.success(
            t(
                "saved_success",
                lang
            )
        )


        reset_workflow()

        st.session_state.wf_patient_id = (
            patient_id
        )


# ============================================================================
# PAGE: DOCTOR DASHBOARD
# ============================================================================

def page_dashboard():

    lang = st.session_state.lang


    show_header(
        f"🩺 {t('dashboard_title', lang)}",
        "Review screening results, risk levels and follow-up status."
    )


    screenings = get_screenings()

    patients = get_patients()


    # ------------------------------------------------------------------------
    # KPI
    # ------------------------------------------------------------------------

    total_patients = len(
        patients
    )


    total_screenings = len(
        screenings
    )


    if screenings:

        df = pd.DataFrame(
            screenings
        )


        high_risk = int(
            df[
                "risk_level"
            ].isin(
                [
                    "High",
                    "Critical"
                ]
            ).sum()
        )


        pending = int(
            (
                df[
                    "followup_status"
                ]
                ==
                "Pending"
            ).sum()
        )


        avg_confidence = float(
            df[
                "confidence"
            ].mean()
        )

    else:

        df = pd.DataFrame()

        high_risk = 0

        pending = 0

        avg_confidence = 0


    # KPI cards

    k1, k2, k3, k4, k5 = st.columns(5)


    k1.metric(
        "👥 Patients",
        total_patients
    )


    k2.metric(
        "🔬 Screenings",
        total_screenings
    )


    k3.metric(
        "⚠️ High/Critical",
        high_risk
    )


    k4.metric(
        "📅 Pending",
        pending
    )


    k5.metric(
        "🤖 Avg Confidence",
        f"{avg_confidence:.1f}%"
    )


    if not screenings:

        st.info(
            "No screenings recorded yet."
        )

        return


    st.divider()


    # Tabs

    tab_overview, tab_patients, tab_followup = (
        st.tabs(
            [
                "📊 Overview",
                "👥 Patient Records",
                "📅 Follow-up",
            ]
        )
    )


    # ========================================================================
    # OVERVIEW
    # ========================================================================

    with tab_overview:

        c1, c2 = st.columns(2)


        with c1:

            st.markdown(
                "<div class='drishti-card'>",
                unsafe_allow_html=True
            )


            st.markdown(
                "### 🧠 DR Severity Distribution"
            )


            st.bar_chart(
                df[
                    "dr_class"
                ].value_counts()
            )


            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


        with c2:

            st.markdown(
                "<div class='drishti-card'>",
                unsafe_allow_html=True
            )


            st.markdown(
                "### ⚠️ Risk Level Distribution"
            )


            st.bar_chart(
                df[
                    "risk_level"
                ].value_counts()
            )


            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


        # Screening table

        st.markdown(
            "<div class='drishti-card'>",
            unsafe_allow_html=True
        )


        st.markdown(
            "### 📋 All Screenings"
        )


        show_df = df[
            [
                "patient_code",
                "patient_name",
                "age",
                "dr_class",
                "confidence",
                "risk_level",
                "referral",
                "followup_status",
                "screened_by",
                "screened_at",
            ]
        ].rename(
            columns={

                "patient_code":
                    "Patient ID",

                "patient_name":
                    "Name",

                "age":
                    "Age",

                "dr_class":
                    "DR Class",

                "confidence":
                    "Confidence (%)",

                "risk_level":
                    "Risk",

                "referral":
                    "Referral",

                "followup_status":
                    "Follow-up",

                "screened_by":
                    "Screened By",

                "screened_at":
                    "Screened At",
            }
        )


        st.dataframe(
            show_df,
            use_container_width=True,
            hide_index=True
        )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # ========================================================================
    # PATIENT RECORDS
    # ========================================================================

    with tab_patients:

        st.markdown(
            "<div class='drishti-card'>",
            unsafe_allow_html=True
        )


        st.markdown(
            "### 👥 Registered Patients"
        )


        if patients:

            patient_df = pd.DataFrame(
                patients
            )


            patient_columns = [
                "patient_code",
                "name",
                "age",
                "diabetes_duration",
                "previous_screening",
                "registered_at",
            ]


            available_columns = [
                c for c in patient_columns
                if c in patient_df.columns
            ]


            patient_display = (
                patient_df[
                    available_columns
                ]
                .rename(
                    columns={
                        "patient_code":
                            "Patient ID",

                        "name":
                            "Name",

                        "age":
                            "Age",

                        "diabetes_duration":
                            "Diabetes Duration (Years)",

                        "previous_screening":
                            "Previous Screening",

                        "registered_at":
                            "Registered At",
                    }
                )
            )


            st.dataframe(
                patient_display,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No registered patients."
            )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # ========================================================================
    # FOLLOW-UP
    # ========================================================================

    with tab_followup:

        st.markdown(
            "<div class='drishti-card'>",
            unsafe_allow_html=True
        )


        st.markdown(
            "### 📅 Follow-up Tracking"
        )


        st.caption(
            "Doctors can update follow-up status "
            "for screened patients."
        )


        for row in screenings:

            with st.container():

                cols = st.columns(
                    [
                        2,
                        2,
                        1.5,
                        1.5,
                        2,
                    ]
                )


                cols[0].write(
                    f"**{row['patient_name']}** "
                    f"({row['patient_code']})"
                )


                cols[1].write(
                    row["dr_class"]
                )


                with cols[2]:

                    risk_badge(
                        row["risk_level"]
                    )


                cols[3].write(
                    row["screened_at"][:10]
                )


                current_status = row[
                    "followup_status"
                ]


                new_status = cols[4].selectbox(

                    t(
                        "followup_status",
                        lang
                    ),

                    FOLLOWUP_STATUSES,

                    index=FOLLOWUP_STATUSES.index(
                        current_status
                    ),

                    key=f"status_{row['id']}",

                    label_visibility="collapsed",

                    disabled=(
                        st.session_state.user[
                            "role"
                        ]
                        !=
                        "Doctor"
                    ),
                )


                if (
                    new_status
                    !=
                    current_status
                    and
                    st.session_state.user[
                        "role"
                    ]
                    ==
                    "Doctor"
                ):

                    update_followup_status(
                        row["id"],
                        new_status
                    )


                    st.success(
                        "Follow-up status updated."
                    )


                    st.rerun()


            st.divider()


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


# ============================================================================
# FOOTER
# ============================================================================

def show_footer():

    st.markdown(
        """
        <div class="drishti-footer">

            <strong>
                DRISHTI-XAI
            </strong>
            · Explainable AI Diabetic Retinopathy Screening

            <br>

            SIH 2026 Prototype ·
            AI-assisted screening only ·
            Not a certified medical device

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================================
# ROUTER
# ============================================================================

def main():

    # ------------------------------------------------------------------------
    # Login
    # ------------------------------------------------------------------------

    if not st.session_state.logged_in:

        login_page()

        return


    # ------------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------------

    sidebar_nav()


    # ------------------------------------------------------------------------
    # Page routing
    # ------------------------------------------------------------------------

    page = st.session_state.page


    if page == "Dashboard":

        page_home()


    elif page == "Register":

        page_register()


    elif page == "Screening":

        page_screening()


    # ------------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------------

    show_footer()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    main()
