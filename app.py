"""
DRISHTI-XAI
-----------
Explainable AI Diabetic Retinopathy Screening
for Rural India

SIH 2026 Prototype

DISCLAIMER:
This is an AI-assisted screening prototype.
It is NOT a certified medical diagnostic device.
"""

import os
import sys
import html
import base64
from io import BytesIO

import streamlit as st
import pandas as pd
from PIL import Image

sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

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
    add_doctor,
    get_doctors,
    add_patient,
    get_patients,
    get_patient_by_id,
    add_screening,
    get_screenings,
    get_screening_by_id,
    update_followup_status,
    save_doctor_review,
)

from core.image_quality import check_image_quality
from core.model import DRModel
from core.gradcam import generate_explanation
from core.risk_engine import assess_risk

from utils.demo_data import (
    generate_all_samples,
    seed_demo_patients,
)

from utils.helpers import (
    pil_to_bgr,
    bgr_to_rgb,
    save_uploaded_image,
    risk_badge,
    dr_class_badge,
)


# ============================================================
# PATHS
# ============================================================

UPLOAD_DIR = os.path.join(
    DATA_DIR,
    "uploads"
)

HEATMAP_DIR = os.path.join(
    DATA_DIR,
    "heatmaps"
)

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)


# ============================================================
# DATABASE / MODEL
# ============================================================

@st.cache_resource
def bootstrap():

    init_db()

    try:
        generate_all_samples()
    except Exception:
        pass

    try:
        seed_demo_patients()
    except Exception:
        pass

    return True


@st.cache_resource
def load_model():
    return DRModel()


bootstrap()
dr_model = load_model()


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

/* =========================================================
   GLOBAL
   ========================================================= */

:root {
    --navy: #0B1F3A;
    --blue: #1976D2;
    --teal: #008C95;
    --teal-light: #E8F7F8;
    --border: #D9E2EC;
    --card-light: #FFFFFF;
}


/* Main background */

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(0, 140, 149, 0.07),
            transparent 28%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(25, 118, 210, 0.07),
            transparent 30%
        );
}


/* Main content */

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #071A2D 0%,
            #0B3048 55%,
            #073B4C 100%
        );
}

section[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.16);
}


/* =========================================================
   HEADINGS
   ========================================================= */

h1 {
    font-weight: 800 !important;
    letter-spacing: -0.5px;
}

h2 {
    font-weight: 750 !important;
}

h3 {
    font-weight: 700 !important;
}


/* =========================================================
   MEDICAL CARDS
   ========================================================= */

.drishti-card {
    padding: 22px;
    border-radius: 18px;
    border: 1px solid var(--border);
    background: var(--card-light);
    margin-bottom: 18px;
    box-shadow:
        0 8px 28px rgba(11,31,58,0.07);
    animation: cardEnter 0.55s ease;
}


/* =========================================================
   ANIMATION
   ========================================================= */

@keyframes cardEnter {

    from {
        opacity: 0;
        transform: translateY(12px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}


@keyframes pulse {

    0% {
        box-shadow:
            0 0 0 0 rgba(0,140,149,0.25);
    }

    70% {
        box-shadow:
            0 0 0 12px rgba(0,140,149,0);
    }

    100% {
        box-shadow:
            0 0 0 0 rgba(0,140,149,0);
    }
}


/* =========================================================
   HERO
   ========================================================= */

.hero {
    padding: 30px;
    border-radius: 22px;
    margin-bottom: 25px;

    background:
        linear-gradient(
            135deg,
            #0B1F3A,
            #075B70
        );

    color: white;

    box-shadow:
        0 14px 40px rgba(7,59,76,0.22);

    animation: cardEnter 0.7s ease;
}

.hero h1 {
    color: white !important;
    margin-bottom: 5px;
}

.hero p {
    color: #D7F1F4 !important;
}


/* =========================================================
   METRICS
   ========================================================= */

div[data-testid="stMetric"] {

    border-radius: 16px;

    border: 1px solid var(--border);

    padding: 17px;

    background: var(--card-light);

    box-shadow:
        0 5px 20px rgba(11,31,58,0.06);

    transition:
        transform 0.25s ease,
        box-shadow 0.25s ease;
}

div[data-testid="stMetric"]:hover {

    transform: translateY(-4px);

    box-shadow:
        0 12px 28px rgba(11,31,58,0.12);
}


/* =========================================================
   BUTTONS
   ========================================================= */

.stButton > button {

    border-radius: 10px;

    font-weight: 650;

    border: 1px solid #1976D2;

    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease;
}

.stButton > button:hover {

    transform: translateY(-2px);

    box-shadow:
        0 6px 16px rgba(25,118,210,0.20);
}


/* =========================================================
   ALERTS
   ========================================================= */

.drishti-banner {

    border-radius: 12px;

    padding: 13px 17px;

    background: #FFF8E7;

    border: 1px solid #F2D38A;

    color: #704F00;

    margin-bottom: 18px;
}


/* =========================================================
   FEATURE CHIPS
   ========================================================= */

.feature-chip {

    display: inline-block;

    padding: 7px 12px;

    margin: 4px;

    border-radius: 30px;

    background: var(--teal-light);

    color: #00636A;

    font-size: 0.86rem;

    font-weight: 600;
}


/* =========================================================
   STATUS
   ========================================================= */

.status-good {

    border-left: 5px solid #16803C;

    padding: 13px;

    background: #EFFAF3;

    border-radius: 10px;
}

.status-warning {

    border-left: 5px solid #D97706;

    padding: 13px;

    background: #FFF7E6;

    border-radius: 10px;
}


/* =========================================================
   PRINT
   ========================================================= */

@media print {

    section[data-testid="stSidebar"],
    header,
    footer,
    .stButton,
    button {
        display: none !important;
    }

    .block-container {
        max-width: 100%;
    }

}


/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 768px) {

    .hero {
        padding: 20px;
    }

    .drishti-card {
        padding: 16px;
    }

}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {

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


for key, value in DEFAULTS.items():

    if key not in st.session_state:
        st.session_state[key] = value


def reset_workflow():

    keys = [
        "wf_image_bgr",
        "wf_image_source",
        "wf_quality",
        "wf_prediction",
        "wf_heatmap",
        "wf_risk",
        "wf_patient_id",
    ]

    for key in keys:
        st.session_state[key] = None


# ============================================================
# HERO
# ============================================================

def hero(title, subtitle=None):

    subtitle = subtitle or APP_TAGLINE

    st.markdown(
        f"""
        <div class="hero">

            <h1>👁️ {html.escape(title)}</h1>

            <p>{html.escape(subtitle)}</p>

            <span class="feature-chip">
                Explainable AI
            </span>

            <span class="feature-chip">
                Rural Screening
            </span>

            <span class="feature-chip">
                Early Detection
            </span>

            <span class="feature-chip">
                Smart Referral
            </span>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# LOGIN
# ============================================================

def login_page():

    left, right = st.columns(
        [1.25, 1]
    )

    with left:

        st.markdown(
            """
            <div style="
                padding-top:50px;
                animation:cardEnter .8s ease;
            ">

                <div style="
                    font-size:70px;
                    animation:pulse 2s infinite;
                    width:90px;
                    border-radius:50%;
                ">
                    👁️
                </div>

                <h1>
                    DRISHTI-XAI
                </h1>

                <h3>
                    Explainable AI Diabetic
                    Retinopathy Screening
                </h3>

                <p>
                    A rural-friendly AI-assisted
                    screening workflow connecting
                    PHC health workers with eye specialists.
                </p>

                <p>
                    <b>PHC → AI Screening → Specialist</b>
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with right:

        st.markdown(
            "<br><br>",
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="drishti-card">',
            unsafe_allow_html=True
        )

        st.subheader("Secure Login")

        username = st.text_input(
            "Username",
            key="login_username"
        )

        password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        if st.button(
            "Login",
            type="primary",
            use_container_width=True
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
                    "Invalid username or password."
                )

        st.caption(
            "Demo Health Worker: "
            "`healthworker1 / worker123`"
        )

        st.caption(
            "Demo Doctor: "
            "`doctor1 / doctor123`"
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="drishti-banner">

        ⚠️ <b>Prototype:</b>
        AI-assisted screening aid only.
        Not a certified medical diagnostic device.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SIDEBAR
# ============================================================

def sidebar_nav():

    user = st.session_state.user

    with st.sidebar:

        st.markdown(
            """
            <div style="
                text-align:center;
                padding:12px 0 8px;
            ">

                <div style="
                    font-size:48px;
                ">
                    👁️
                </div>

                <h2 style="
                    margin:0;
                    color:white !important;
                ">
                    DRISHTI-XAI
                </h2>

                <p style="
                    color:#B8D8E5 !important;
                    font-size:.8rem;
                ">
                    Rural Retina Screening
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.divider()

        st.markdown(
            f"""
            <div style="
                padding:12px;
                border-radius:12px;
                background:rgba(255,255,255,.08);
            ">

            <b>Welcome</b><br>
            {html.escape(user["display_name"])}

            <br><br>

            <small>
            Role: {html.escape(user["role"])}
            </small>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        # ----------------------------------------------------
        # NAVIGATION
        # ----------------------------------------------------

        if user["role"] == "Doctor":

            nav_options = {

                "📝 Patient Registration":
                    "Register",

                "🔬 AI Screening":
                    "Screening",

                "🩺 Doctor Review":
                    "Doctor Review",

                "📊 Dashboard":
                    "Dashboard",

            }

        else:

            nav_options = {

                "📝 Patient Registration":
                    "Register",

                "🔬 AI Screening":
                    "Screening",

                "📊 Dashboard":
                    "Dashboard",

            }

        labels = list(
            nav_options.keys()
        )

        current_page = st.session_state.page

        current_label = next(
            (
                label
                for label, page in nav_options.items()
                if page == current_page
            ),
            labels[0]
        )

        choice = st.radio(
            "Navigation",
            labels,
            index=labels.index(current_label),
            label_visibility="collapsed"
        )

        st.session_state.page = nav_options[choice]

        st.divider()

        if st.button(
            "🚪 Logout",
            use_container_width=True
        ):

            st.session_state.logged_in = False

            st.session_state.user = None

            reset_workflow()

            st.rerun()


# ============================================================
# PATIENT REGISTRATION
# ============================================================

def page_register():

    hero(
        "Patient Registration",
        "Create and manage patient records before retinal screening."
    )

    patients = get_patients()

    c1, c2 = st.columns(
        [1, 1.45]
    )

    with c1:

        st.markdown(
            '<div class="drishti-card">',
            unsafe_allow_html=True
        )

        st.subheader(
            "➕ Register New Patient"
        )

        with st.form(
            "patient_registration",
            clear_on_submit=True
        ):

            patient_code = st.text_input(
                "Patient ID",
                placeholder="PT-1005"
            )

            name = st.text_input(
                "Patient Name"
            )

            age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=45
            )

            duration = st.number_input(
                "Diabetes Duration (years)",
                min_value=0,
                max_value=70,
                value=5
            )

            previous = st.selectbox(
                "Previous Screening",
                ["No", "Yes"]
            )

            submit = st.form_submit_button(
                "Register Patient",
                type="primary",
                use_container_width=True
            )

        if submit:

            if not patient_code.strip() or not name.strip():

                st.error(
                    "Patient ID and Name are required."
                )

            else:

                new_id = add_patient(
                    patient_code.strip(),
                    name.strip(),
                    int(age),
                    int(duration),
                    previous,
                    st.session_state.user[
                        "display_name"
                    ]
                )

                if new_id:

                    st.success(
                        "Patient registered successfully."
                    )

                    st.rerun()

                else:

                    st.error(
                        "Patient ID already exists."
                    )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            '<div class="drishti-card">',
            unsafe_allow_html=True
        )

        st.subheader(
            f"👥 Registered Patients ({len(patients)})"
        )

        if patients:

            search = st.text_input(
                "Search patient",
                placeholder="Name or Patient ID"
            )

            filtered = patients

            if search:

                search_lower = search.lower()

                filtered = [

                    p for p in patients

                    if (
                        search_lower
                        in p["name"].lower()
                        or
                        search_lower
                        in p["patient_code"].lower()
                    )
                ]

            df = pd.DataFrame(filtered)

            if not df.empty:

                display = df[
                    [
                        "patient_code",
                        "name",
                        "age",
                        "diabetes_duration",
                        "previous_screening",
                        "registered_at",
                    ]
                ].copy()

                display.columns = [
                    "Patient ID",
                    "Name",
                    "Age",
                    "DM Duration",
                    "Previous Screening",
                    "Registered At",
                ]

                st.dataframe(
                    display,
                    use_container_width=True,
                    hide_index=True
                )

        else:

            st.info(
                "No registered patients yet."
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


# ============================================================
# SCREENING
# ============================================================

def page_screening():

    hero(
        "AI Retinal Screening",
        "Fundus image quality → AI screening → explanation → risk → referral."
    )

    patients = get_patients()

    if not patients:

        st.warning(
            "Please register a patient first."
        )

        return

    patient_labels = {

        f"{p['patient_code']} — {p['name']}":
            p["id"]

        for p in patients
    }

    selected_label = st.selectbox(
        "Select Patient",
        list(patient_labels.keys())
    )

    patient_id = patient_labels[
        selected_label
    ]

    patient = get_patient_by_id(
        patient_id
    )

    if (
        st.session_state.wf_patient_id
        != patient_id
    ):

        reset_workflow()

        st.session_state.wf_patient_id = patient_id

    st.markdown(
        f"""
        <div class="drishti-card">

        <b>Patient:</b>
        {html.escape(patient["name"])}
        &nbsp;&nbsp; | &nbsp;&nbsp;

        <b>Age:</b>
        {patient["age"]}

        &nbsp;&nbsp; | &nbsp;&nbsp;

        <b>Diabetes:</b>
        {patient["diabetes_duration"]} years

        &nbsp;&nbsp; | &nbsp;&nbsp;

        <b>Previous screening:</b>
        {html.escape(patient["previous_screening"])}

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    st.subheader(
        "1️⃣ Fundus Image"
    )

    up_col, sample_col = st.columns(
        [2, 1]
    )

    with up_col:

        uploaded = st.file_uploader(
            "Upload fundus image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded:

            image = Image.open(
                uploaded
            ).convert("RGB")

            st.session_state.wf_image_bgr = (
                pil_to_bgr(image)
            )

            st.session_state.wf_image_source = (
                "upload"
            )

            st.session_state.wf_quality = None
            st.session_state.wf_prediction = None
            st.session_state.wf_heatmap = None
            st.session_state.wf_risk = None

    with sample_col:

        st.caption(
            "Demo samples"
        )

        for severity in DR_CLASSES:

            filename = (
                "sample_"
                + severity.replace(
                    " ", "_"
                ).lower()
                + ".png"
            )

            path = os.path.join(
                SAMPLE_IMAGES_DIR,
                filename
            )

            if os.path.exists(path):

                if st.button(
                    f"Use {severity}",
                    key=f"sample_{severity}",
                    use_container_width=True
                ):

                    image = Image.open(
                        path
                    ).convert("RGB")

                    st.session_state.wf_image_bgr = (
                        pil_to_bgr(image)
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
            "Upload a fundus image or choose a demo image."
        )

        return

    image_rgb = bgr_to_rgb(
        st.session_state.wf_image_bgr
    )

    st.image(
        image_rgb,
        caption="Fundus image preview",
        width=360
    )

    # --------------------------------------------------------
    # QUALITY
    # --------------------------------------------------------

    st.subheader(
        "2️⃣ Image Quality Check"
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

    quality = st.session_state.wf_quality

    if quality:

        q1, q2, q3 = st.columns(3)

        q1.metric(
            "Blur Score",
            quality["blur_score"]
        )

        q2.metric(
            "Brightness",
            quality["brightness"]
        )

        q3.metric(
            "Resolution",
            f"{quality['resolution'][0]} × "
            f"{quality['resolution'][1]}"
        )

        if quality["passed"]:

            st.success(
                "Image quality is suitable for screening."
            )

        else:

            st.error(
                "Image quality is not suitable."
            )

            for issue in quality["issues"]:

                st.write(
                    f"• {issue}"
                )

            return

    else:

        return

    # --------------------------------------------------------
    # AI
    # --------------------------------------------------------

    st.subheader(
        "3️⃣ AI Screening & Explainability"
    )

    if st.button(
        "Run AI Screening",
        type="primary",
        use_container_width=False
    ):

        with st.spinner(
            "Analyzing retinal image..."
        ):

            dr_class, confidence, probs = (
                dr_model.predict(
                    st.session_state.wf_image_bgr
                )
            )

            st.session_state.wf_prediction = {

                "dr_class": dr_class,

                "confidence": confidence,

                "probs": probs,

            }

            st.session_state.wf_heatmap = (
                generate_explanation(
                    dr_model,
                    st.session_state.wf_image_bgr
                )
            )

            st.session_state.wf_risk = (
                assess_risk(
                    dr_class,
                    confidence,
                    patient[
                        "diabetes_duration"
                    ],
                    patient[
                        "previous_screening"
                    ]
                )
            )

    prediction = (
        st.session_state.wf_prediction
    )

    if not prediction:

        return

    if dr_model.mode == "mock":

        st.info(
            "Prototype demo mode: model output is simulated."
        )

    r1, r2 = st.columns(2)

    with r1:

        st.markdown(
            '<div class="drishti-card">',
            unsafe_allow_html=True
        )

        st.subheader(
            "AI Result"
        )

        dr_class_badge(
            prediction["dr_class"]
        )

        st.metric(
            "Confidence",
            f"{prediction['confidence']}%"
        )

        probs_df = pd.DataFrame({

            "DR Class":
                list(
                    prediction["probs"].keys()
                ),

            "Probability":
                list(
                    prediction["probs"].values()
                ),

        })

        st.bar_chart(
            probs_df.set_index(
                "DR Class"
            )
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    with r2:

        st.markdown(
            '<div class="drishti-card">',
            unsafe_allow_html=True
        )

        st.subheader(
            "Explainable AI — Grad-CAM"
        )

        st.image(
            bgr_to_rgb(
                st.session_state.wf_heatmap
            ),
            caption="Regions contributing to the model output.",
            use_container_width=True
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    risk = st.session_state.wf_risk

    st.markdown(
        '<div class="drishti-card">',
        unsafe_allow_html=True
    )

    st.subheader(
        "4️⃣ Risk Assessment & Referral"
    )

    a, b = st.columns(
        [1, 2]
    )

    with a:

        risk_badge(
            risk["risk_level"]
        )

    with b:

        st.write(
            f"**Referral:** {risk['referral']}"
        )

    with st.expander(
        "Why was this risk level assigned?"
    ):

        for reason in risk["rationale"]:

            st.write(
                f"• {reason}"
            )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    if st.button(
        "💾 Save Screening Record",
        type="primary"
    ):

        image_path = save_uploaded_image(
            st.session_state.wf_image_bgr,
            UPLOAD_DIR
        )

        heatmap_path = save_uploaded_image(
            st.session_state.wf_heatmap,
            HEATMAP_DIR
        )

        add_screening(
            patient_id=patient_id,
            image_path=image_path,
            dr_class=prediction["dr_class"],
            confidence=prediction["confidence"],
            risk_level=risk["risk_level"],
            referral=risk["referral"],
            heatmap_path=heatmap_path,
            screened_by=st.session_state.user[
                "display_name"
            ],
        )

        st.success(
            "Screening saved successfully."
        )

        reset_workflow()

        st.session_state.wf_patient_id = (
            patient_id
        )


# ============================================================
# DOCTOR REGISTRATION
# ============================================================

def doctor_registration():

    st.subheader(
        "👨‍⚕️ Register New Doctor"
    )

    with st.form(
        "doctor_registration_form"
    ):

        name = st.text_input(
            "Doctor Name"
        )

        username = st.text_input(
            "Doctor Username"
        )

        password = st.text_input(
            "Password",
            type="password"
        )

        confirm = st.text_input(
            "Confirm Password",
            type="password"
        )

        submit = st.form_submit_button(
            "Create Doctor Account",
            type="primary"
        )

        if submit:

            if not name or not username or not password:

                st.error(
                    "All fields are required."
                )

            elif password != confirm:

                st.error(
                    "Passwords do not match."
                )

            else:

                result = add_doctor(
                    username.strip(),
                    password,
                    name.strip()
                )

                if result:

                    st.success(
                        "Doctor account created successfully."
                    )

                else:

                    st.error(
                        "Username already exists."
                    )


# ============================================================
# DOCTOR REVIEW
# ============================================================

def page_doctor_review():

    hero(
        "Doctor Review",
        "Specialist review, clinical suggestions and final patient record."
    )

    if st.session_state.user["role"] != "Doctor":

        st.error(
            "Doctor access required."
        )

        return

    screenings = get_screenings()

    if not screenings:

        st.info(
            "No screening records are available for review."
        )

        return

    # --------------------------------------------------------
    # DOCTOR REGISTRATION EXPANDER
    # --------------------------------------------------------

    with st.expander(
        "➕ Register another doctor"
    ):

        doctor_registration()

    # --------------------------------------------------------
    # SCREENING SELECT
    # --------------------------------------------------------

    options = {

        f"{s['patient_code']} — "
        f"{s['patient_name']} — "
        f"{s['dr_class']} — "
        f"{s['screened_at'][:10]}":
            s["id"]

        for s in screenings
    }

    selected = st.selectbox(
        "Select screening record",
        list(options.keys())
    )

    screening_id = options[
        selected
    ]

    record = get_screening_by_id(
        screening_id
    )

    if not record:

        st.error(
            "Unable to load screening record."
        )

        return

    # --------------------------------------------------------
    # PATIENT DETAILS
    # --------------------------------------------------------

    st.markdown(
        '<div class="drishti-card">',
        unsafe_allow_html=True
    )

    st.subheader(
        "👤 Patient Information"
    )

    p1, p2, p3, p4 = st.columns(4)

    p1.metric(
        "Patient ID",
        record["patient_code"]
    )

    p2.metric(
        "Patient",
        record["patient_name"]
    )

    p3.metric(
        "Age",
        record["age"]
    )

    p4.metric(
        "Diabetes",
        f"{record['diabetes_duration']} years"
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # AI RESULT
    # --------------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            '<div class="drishti-card">',
            unsafe_allow_html=True
        )

        st.subheader(
            "AI Screening Result"
        )

        dr_class_badge(
            record["dr_class"]
        )

        st.metric(
            "AI Confidence",
            f"{record['confidence']}%"
        )

        st.markdown(
            f"""
            **Risk:** {record["risk_level"]}

            **Referral:** {record["referral"]}

            **Screened by:** {record["screened_by"]}

            **Screened at:** {record["screened_at"]}
            """
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            '<div class="drishti-card">',
            unsafe_allow_html=True
        )

        st.subheader(
            "Explainable AI"
        )

        if record.get("heatmap_path") and os.path.exists(
            record["heatmap_path"]
        ):

            st.image(
                record["heatmap_path"],
                caption="Grad-CAM explanation",
                use_container_width=True
            )

        else:

            st.info(
                "Grad-CAM image unavailable."
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # DOCTOR DECISION
    # --------------------------------------------------------

    st.subheader(
        "🩺 Doctor Review"
    )

    existing_decision = (
        record.get("doctor_decision")
        or "Needs Specialist Review"
    )

    decision_options = [
        "Confirmed — Refer to Specialist",
        "Confirmed — Routine Follow-up",
        "Needs Further Examination",
        "AI Result Requires Reassessment",
    ]

    decision_index = (
        decision_options.index(
            existing_decision
        )
        if existing_decision
        in decision_options
        else 0
    )

    decision = st.selectbox(
        "Doctor's assessment",
        decision_options,
        index=decision_index
    )

    # --------------------------------------------------------
    # STANDARD ADVICE
    # --------------------------------------------------------

    st.subheader(
        "📋 Doctor's Recommendations"
    )

    col1, col2 = st.columns(2)

    with col1:

        specialist_exam = st.text_area(
            "1. Specialist Eye Examination",
            value=record.get(
                "specialist_exam"
            ) or
            "Get a dilated retinal examination and OCT scan at an eye hospital within 2–4 weeks.",
            height=110
        )

        blood_sugar = st.text_area(
            "2. Control Blood Sugar",
            value=record.get(
                "blood_sugar_advice"
            ) or
            "Target HbA1c below 7%. Take prescribed medicines/insulin regularly and never skip doses.",
            height=110
        )

        bp_cholesterol = st.text_area(
            "3. Blood Pressure & Cholesterol",
            value=record.get(
                "bp_cholesterol_advice"
            ) or
            "Keep BP under 130/80 mmHg and check lipid profile regularly. Both can worsen retinopathy.",
            height=110
        )

    with col2:

        diet = st.text_area(
            "4. Diet & Lifestyle",
            value=record.get(
                "diet_lifestyle_advice"
            ) or
            "Follow a high-fibre, lower-sugar diet, stay physically active, avoid smoking and limit alcohol.",
            height=110
        )

        warning = st.text_area(
            "5. Warning Signs",
            value=record.get(
                "warning_signs_advice"
            ) or
            "Sudden blurred vision, new floaters, dark spots or eye pain require prompt medical attention.",
            height=110
        )

        followup = st.text_area(
            "6. Follow-up Screening",
            value=record.get(
                "followup_advice"
            ) or
            "Repeat retinal screening according to the specialist's advice and patient risk level.",
            height=110
        )

    additional = st.text_area(
        "7. Additional Advice from Doctor",
        value=record.get(
            "additional_doctor_advice"
        ) or "",
        placeholder=(
            "Enter any patient-specific advice, "
            "medication instructions, referral details "
            "or follow-up notes..."
        ),
        height=130
    )

    # --------------------------------------------------------
    # SAVE REVIEW
    # --------------------------------------------------------

    if st.button(
        "💾 Save Doctor Review",
        type="primary",
        use_container_width=True
    ):

        save_doctor_review(

            screening_id,

            decision,

            specialist_exam,

            blood_sugar,

            bp_cholesterol,

            diet,

            warning,

            followup,

            additional,

            st.session_state.user[
                "display_name"
            ],
        )

        st.success(
            "Doctor review saved successfully."
        )

        st.rerun()

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    st.divider()

    reviewed = get_screening_by_id(
        screening_id
    )

    if reviewed.get(
        "reviewed_at"
    ):

        st.success(
            f"Reviewed by {reviewed['reviewed_by']} "
            f"on {reviewed['reviewed_at']}"
        )

        st.markdown(
            """
            ### 🖨️ Patient Record

            Use your browser's **Print** option
            to print or save this reviewed record
            as PDF.
            """
        )

        if st.button(
            "🖨️ Print / Save Patient Record",
            use_container_width=True
        ):

            st.components.v1.html(
                f"""
                <script>

                window.parent.print();

                </script>
                """,
                height=0
            )

        # ----------------------------------------------------
        # PRINTABLE RECORD
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div class="drishti-card">

            <h2>DRISHTI-XAI — Patient Screening Record</h2>

            <hr>

            <h3>Patient Information</h3>

            <p>
            <b>Patient ID:</b>
            {html.escape(str(reviewed["patient_code"]))}
            </p>

            <p>
            <b>Name:</b>
            {html.escape(str(reviewed["patient_name"]))}
            </p>

            <p>
            <b>Age:</b>
            {reviewed["age"]}
            </p>

            <p>
            <b>Diabetes Duration:</b>
            {reviewed["diabetes_duration"]} years
            </p>

            <h3>AI Screening</h3>

            <p>
            <b>DR Classification:</b>
            {html.escape(str(reviewed["dr_class"]))}
            </p>

            <p>
            <b>AI Confidence:</b>
            {reviewed["confidence"]}%
            </p>

            <p>
            <b>Risk Level:</b>
            {html.escape(str(reviewed["risk_level"]))}
            </p>

            <p>
            <b>Referral:</b>
            {html.escape(str(reviewed["referral"]))}
            </p>

            <h3>Doctor Review</h3>

            <p>
            <b>Assessment:</b>
            {html.escape(str(reviewed["doctor_decision"]))}
            </p>

            <h3>Doctor's Recommendations</h3>

            <p>
            <b>Specialist Eye Examination:</b><br>
            {html.escape(str(reviewed["specialist_exam"]))}
            </p>

            <p>
            <b>Blood Sugar:</b><br>
            {html.escape(str(reviewed["blood_sugar_advice"]))}
            </p>

            <p>
            <b>Blood Pressure & Cholesterol:</b><br>
            {html.escape(str(reviewed["bp_cholesterol_advice"]))}
            </p>

            <p>
            <b>Diet & Lifestyle:</b><br>
            {html.escape(str(reviewed["diet_lifestyle_advice"]))}
            </p>

            <p>
            <b>Warning Signs:</b><br>
            {html.escape(str(reviewed["warning_signs_advice"]))}
            </p>

            <p>
            <b>Follow-up Screening:</b><br>
            {html.escape(str(reviewed["followup_advice"]))}
            </p>

            <p>
            <b>Additional Doctor Advice:</b><br>
            {html.escape(str(reviewed["additional_doctor_advice"]))}
            </p>

            <hr>

            <p>
            <b>Reviewed By:</b>
            {html.escape(str(reviewed["reviewed_by"]))}
            </p>

            <p>
            <b>Reviewed At:</b>
            {html.escape(str(reviewed["reviewed_at"]))}
            </p>

            <br>

            <small>
            DRISHTI-XAI is an AI-assisted screening
            prototype and is not a certified medical
            diagnostic device.
            </small>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# DASHBOARD
# ============================================================

def page_dashboard():

    hero(
        "Clinical Dashboard",
        "Screening, risk and follow-up overview."
    )

    screenings = get_screenings()

    patients = get_patients()

    total_screenings = len(
        screenings
    )

    high_risk = sum(
        1
        for s in screenings
        if s["risk_level"]
        in ["High", "Critical"]
    )

    pending = sum(
        1
        for s in screenings
        if s["followup_status"]
        == "Pending"
    )

    reviewed = sum(
        1
        for s in screenings
        if s.get("reviewed_at")
    )

    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric(
        "Patients",
        len(patients)
    )

    k2.metric(
        "Screenings",
        total_screenings
    )

    k3.metric(
        "High / Critical",
        high_risk
    )

    k4.metric(
        "Pending Follow-up",
        pending
    )

    k5.metric(
        "Doctor Reviewed",
        reviewed
    )

    if not screenings:

        st.info(
            "No screening records available yet."
        )

        return

    df = pd.DataFrame(
        screenings
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "📊 Overview",
            "👥 Screening Records",
            "🔄 Follow-up",
        ]
    )

    with tab1:

        c1, c2 = st.columns(2)

        with c1:

            st.markdown(
                '<div class="drishti-card">',
                unsafe_allow_html=True
            )

            st.subheader(
                "DR Classification"
            )

            st.bar_chart(
                df["dr_class"].value_counts()
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

        with c2:

            st.markdown(
                '<div class="drishti-card">',
                unsafe_allow_html=True
            )

            st.subheader(
                "Risk Distribution"
            )

            st.bar_chart(
                df["risk_level"].value_counts()
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

    with tab2:

        search = st.text_input(
            "Search screening records"
        )

        show = df.copy()

        if search:

            s = search.lower()

            show = show[
                show.apply(
                    lambda row:
                    s in str(
                        row.to_dict()
                    ).lower(),
                    axis=1
                )
            ]

        columns = [

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

        show_df = show[
            columns
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

    with tab3:

        st.subheader(
            "Follow-up Management"
        )

        for row in screenings:

            cols = st.columns(
                [2, 1.5, 1.5, 2]
            )

            cols[0].write(
                f"**{row['patient_name']}** "
                f"({row['patient_code']})"
            )

            cols[1].write(
                row["risk_level"]
            )

            cols[2].write(
                row["screened_at"][:10]
            )

            current = (
                row["followup_status"]
            )

            new_status = cols[3].selectbox(
                "Status",
                FOLLOWUP_STATUSES,
                index=FOLLOWUP_STATUSES.index(
                    current
                ),
                key=f"followup_{row['id']}",
                label_visibility="collapsed",
                disabled=(
                    st.session_state.user[
                        "role"
                    ] != "Doctor"
                )
            )

            if (
                new_status != current
                and
                st.session_state.user[
                    "role"
                ] == "Doctor"
            ):

                update_followup_status(
                    row["id"],
                    new_status
                )

                st.rerun()


# ============================================================
# MAIN ROUTER
# ============================================================

def main():

    if not st.session_state.logged_in:

        login_page()

        return

    sidebar_nav()

    page = st.session_state.page

    if page == "Register":

        page_register()

    elif page == "Screening":

        page_screening()

    elif page == "Doctor Review":

        page_doctor_review()

    elif page == "Dashboard":

        page_dashboard()


if __name__ == "__main__":

    main()
