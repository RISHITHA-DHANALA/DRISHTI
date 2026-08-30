"""
app.py
------
DRISHTI-XAI — Explainable AI Diabetic Retinopathy Screening for Rural India
Main Streamlit entry point.

Workflow implemented:
    Login -> Patient Registration -> Fundus Image Upload -> Image Quality
    Check -> DR Prediction -> Grad-CAM Explanation -> Risk Assessment ->
    Smart Referral -> Doctor Dashboard -> Follow-up Tracking

Run with:
    streamlit run app.py

*** PROTOTYPE DISCLAIMER ***
This application is an AI-ASSISTED SCREENING PROTOTYPE built for a
hackathon (SIH 2026) context. It is NOT a certified medical device and
has NOT undergone clinical validation. Do not use it for real patient
diagnosis or care decisions.
"""

import os
import sys
import streamlit as st
import pandas as pd
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (  # noqa: E402
    APP_NAME, APP_TAGLINE, IS_PROTOTYPE, DR_CLASSES, SUPPORTED_LANGUAGES,
    FOLLOWUP_STATUSES, SAMPLE_IMAGES_DIR, DATA_DIR,
)
from i18n.translations import t  # noqa: E402
from db.database import (  # noqa: E402
    init_db, authenticate, add_patient, get_patients, get_patient_by_id,
    add_screening, get_screenings, update_followup_status,
)
from core.image_quality import check_image_quality  # noqa: E402
from core.model import DRModel  # noqa: E402
from core.gradcam import generate_explanation  # noqa: E402
from core.risk_engine import assess_risk  # noqa: E402
from utils.demo_data import generate_all_samples, seed_demo_patients  # noqa: E402
from utils.helpers import (  # noqa: E402
    pil_to_bgr, bgr_to_rgb, save_uploaded_image, risk_badge, dr_class_badge,
)

UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
HEATMAP_DIR = os.path.join(DATA_DIR, "heatmaps")

# ---------------------------------------------------------------------------
# One-time setup (DB tables, demo users, synthetic sample images, seed data)
# ---------------------------------------------------------------------------
@st.cache_resource
def bootstrap():
    init_db()
    generate_all_samples()
    seed_demo_patients()
    return True


@st.cache_resource
def load_model():
    """Cache the (mock or real) DR model across reruns/sessions."""
    return DRModel()


bootstrap()
dr_model = load_model()

st.set_page_config(
    page_title=APP_NAME,
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global styling — clean, modern medical-dashboard look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #F7F9FB; }
    .block-container { padding-top: 1.5rem; }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #E3E8EF;
        border-radius: 12px;
        padding: 14px 16px;
    }
    .drishti-card {
        background: white;
        border: 1px solid #E3E8EF;
        border-radius: 14px;
        padding: 20px 22px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(16,24,40,0.04);
    }
    .drishti-banner {
        background: #FFF7E6;
        border: 1px solid #FFD98E;
        color: #8A5A00;
        padding: 10px 16px;
        border-radius: 10px;
        font-size: 0.88rem;
        margin-bottom: 14px;
    }
    h1, h2, h3 { color: #10243E; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
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
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_workflow():
    for key in ["wf_image_bgr", "wf_image_source", "wf_quality", "wf_prediction",
                "wf_heatmap", "wf_risk", "wf_patient_id"]:
        st.session_state[key] = None


# ---------------------------------------------------------------------------
# Login page
# ---------------------------------------------------------------------------
def login_page():
    lang = st.session_state.lang

    col_logo, col_lang = st.columns([4, 1])
    with col_logo:
        st.markdown(f"## 👁️ {t('app_title', lang)}")
        st.caption(t("app_tagline", lang))
    with col_lang:
        chosen = st.selectbox(t("select_language", lang), list(SUPPORTED_LANGUAGES.keys()), index=0)
        st.session_state.lang = SUPPORTED_LANGUAGES[chosen]
        lang = st.session_state.lang

    st.markdown(f"<div class='drishti-banner'>⚠️ {t('prototype_banner', lang)}</div>", unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown("<div class='drishti-card'>", unsafe_allow_html=True)
        st.subheader(t("login_title", lang))
        username = st.text_input(t("username", lang), key="login_user")
        password = st.text_input(t("password", lang), type="password", key="login_pass")

        if st.button(t("login_button", lang), use_container_width=True, type="primary"):
            user = authenticate(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.user = user
                st.session_state.page = "Register"
                st.rerun()
            else:
                st.error(t("login_error", lang))

        st.caption("Demo accounts — Health Worker: `healthworker1` / `worker123`  ·  "
                   "Doctor: `doctor1` / `doctor123`")
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar navigation (shown once logged in)
# ---------------------------------------------------------------------------
def sidebar_nav():
    lang = st.session_state.lang
    user = st.session_state.user

    with st.sidebar:
        st.markdown(f"### 👁️ {t('app_title', lang)}")
        st.caption(t("app_tagline", lang))
        st.divider()
        st.markdown(f"**{t('welcome', lang)}, {user['display_name']}**")
        st.caption(f"{t('role', lang)}: {user['role']}")

        chosen = st.selectbox(
            t("select_language", lang),
            list(SUPPORTED_LANGUAGES.keys()),
            index=list(SUPPORTED_LANGUAGES.values()).index(lang),
        )
        st.session_state.lang = SUPPORTED_LANGUAGES[chosen]

        st.divider()
        nav_options = {
            t("nav_register", st.session_state.lang): "Register",
            t("nav_upload", st.session_state.lang): "Screening",
            t("nav_dashboard", st.session_state.lang): "Dashboard",
        }
        choice_label = st.radio("Navigation", list(nav_options.keys()), label_visibility="collapsed")
        st.session_state.page = nav_options[choice_label]

        st.divider()
        if IS_PROTOTYPE:
            if dr_model.mode == "mock":
                st.caption("🧪 " + t("demo_mode_notice", st.session_state.lang))
            else:
                st.caption("✅ Real model loaded.")

        if st.button(t("logout", st.session_state.lang), use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user = None
            reset_workflow()
            st.rerun()


# ---------------------------------------------------------------------------
# Page: Patient Registration
# ---------------------------------------------------------------------------
def page_register():
    lang = st.session_state.lang
    st.header(f"📝 {t('nav_register', lang)}")

    col1, col2 = st.columns([1, 1.3])

    with col1:
        st.markdown("<div class='drishti-card'>", unsafe_allow_html=True)
        with st.form("register_form", clear_on_submit=True):
            patient_code = st.text_input(t("patient_id", lang), placeholder="e.g. PT-1005")
            name = st.text_input(t("patient_name", lang))
            age = st.number_input(t("age", lang), min_value=1, max_value=120, value=45)
            duration = st.number_input(t("diabetes_duration", lang), min_value=0, max_value=70, value=5)
            prev_screen = st.selectbox(t("previous_screening", lang), ["No", "Yes"])
            submitted = st.form_submit_button(t("register_button", lang), type="primary", use_container_width=True)

            if submitted:
                if not patient_code or not name:
                    st.error("Patient ID and Name are required.")
                else:
                    add_patient(patient_code, name, int(age), int(duration), prev_screen,
                                st.session_state.user["display_name"])
                    st.success(t("registered_success", lang))
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='drishti-card'>", unsafe_allow_html=True)
        st.subheader("Registered Patients")
        patients = get_patients()
        if not patients:
            st.info(t("no_patients", lang))
        else:
            df = pd.DataFrame(patients)[
                ["patient_code", "name", "age", "diabetes_duration", "previous_screening", "registered_at"]
            ]
            df.columns = ["Patient ID", "Name", "Age", "DM Duration (y)", "Prev. Screening", "Registered At"]
            st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page: Image Upload -> Quality Check -> Prediction -> Grad-CAM -> Risk -> Save
# ---------------------------------------------------------------------------
def page_screening():
    lang = st.session_state.lang
    st.header(f"🔬 {t('nav_upload', lang)}")

    patients = get_patients()
    if not patients:
        st.warning(t("no_patients", lang) + " Please register a patient first.")
        return

    patient_labels = {f"{p['patient_code']} — {p['name']}": p["id"] for p in patients}
    chosen_label = st.selectbox(t("select_patient", lang), list(patient_labels.keys()))
    patient_id = patient_labels[chosen_label]
    patient = get_patient_by_id(patient_id)

    if st.session_state.wf_patient_id != patient_id:
        # Patient changed — reset the in-progress workflow state.
        reset_workflow()
        st.session_state.wf_patient_id = patient_id

    st.markdown("<div class='drishti-card'>", unsafe_allow_html=True)
    st.markdown(f"**Patient:** {patient['name']}  |  **Age:** {patient['age']}  |  "
                f"**Diabetes duration:** {patient['diabetes_duration']}y  |  "
                f"**Previous screening:** {patient['previous_screening']}")
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Step 1: Image source -------------------------------------------------
    st.subheader("Step 1 · " + t("upload_image", lang))
    col_up, col_sample = st.columns([2, 1])
    with col_up:
        uploaded_file = st.file_uploader(t("upload_image", lang), type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file)
            st.session_state.wf_image_bgr = pil_to_bgr(pil_img)
            st.session_state.wf_image_source = "upload"
            st.session_state.wf_quality = None
            st.session_state.wf_prediction = None
            st.session_state.wf_heatmap = None
            st.session_state.wf_risk = None

    with col_sample:
        st.caption(t("use_sample", lang))
        for severity in DR_CLASSES:
            fname = f"sample_{severity.replace(' ', '_').lower()}.png"
            fpath = os.path.join(SAMPLE_IMAGES_DIR, fname)
            if st.button(f"Use: {severity} sample", key=f"sample_{severity}", use_container_width=True):
                pil_img = Image.open(fpath)
                st.session_state.wf_image_bgr = pil_to_bgr(pil_img)
                st.session_state.wf_image_source = f"sample:{severity}"
                st.session_state.wf_quality = None
                st.session_state.wf_prediction = None
                st.session_state.wf_heatmap = None
                st.session_state.wf_risk = None

    if st.session_state.wf_image_bgr is None:
        st.info("Upload a fundus image or pick a demo sample to begin.")
        return

    st.image(bgr_to_rgb(st.session_state.wf_image_bgr), caption=t("image_preview", lang), width=320)

    # --- Step 2: Quality check --------------------------------------------
    st.subheader("Step 2 · " + t("quality_check", lang))
    if st.button("Run Quality Check", type="primary"):
        st.session_state.wf_quality = check_image_quality(st.session_state.wf_image_bgr)

    quality = st.session_state.wf_quality
    if quality:
        cols = st.columns(3)
        cols[0].metric("Blur score (Laplacian var.)", quality["blur_score"])
        cols[1].metric("Brightness (0-255)", quality["brightness"])
        cols[2].metric("Resolution", f"{quality['resolution'][0]}×{quality['resolution'][1]}")

        if quality["passed"]:
            st.success(t("quality_pass", lang))
        else:
            st.error(t("quality_fail", lang))
            for issue in quality["issues"]:
                st.write(f"- {t(issue, lang)}")
            st.stop()

    if not quality or not quality["passed"]:
        return

    # --- Step 3: DR Prediction + Grad-CAM ----------------------------------
    st.subheader("Step 3 · " + t("run_screening", lang))
    if st.button(t("run_screening", lang), type="primary"):
        dr_class, confidence, probs = dr_model.predict(st.session_state.wf_image_bgr)
        st.session_state.wf_prediction = {"dr_class": dr_class, "confidence": confidence, "probs": probs}
        st.session_state.wf_heatmap = generate_explanation(dr_model, st.session_state.wf_image_bgr)

        risk_result = assess_risk(
            dr_class, confidence, patient["diabetes_duration"], patient["previous_screening"]
        )
        st.session_state.wf_risk = risk_result

    prediction = st.session_state.wf_prediction
    if not prediction:
        return

    if dr_model.mode == "mock":
        st.caption("🧪 " + t("demo_mode_notice", lang))

    col_res, col_cam = st.columns(2)
    with col_res:
        st.markdown("<div class='drishti-card'>", unsafe_allow_html=True)
        st.markdown(f"#### {t('prediction_result', lang)}")
        dr_class_badge(prediction["dr_class"])
        st.metric(t("confidence", lang), f"{prediction['confidence']}%")
        probs_df = pd.DataFrame(
            {"DR Class": list(prediction["probs"].keys()), "Probability": list(prediction["probs"].values())}
        )
        st.bar_chart(probs_df.set_index("DR Class"))
        st.markdown("</div>", unsafe_allow_html=True)

    with col_cam:
        st.markdown("<div class='drishti-card'>", unsafe_allow_html=True)
        st.markdown(f"#### {t('gradcam_title', lang)}")
        st.image(bgr_to_rgb(st.session_state.wf_heatmap), caption=t("gradcam_caption", lang), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Step 4: Risk & referral --------------------------------------------
    risk = st.session_state.wf_risk
    st.markdown("<div class='drishti-card'>", unsafe_allow_html=True)
    st.markdown(f"#### {t('risk_level', lang)} & {t('referral', lang)}")
    c1, c2 = st.columns([1, 2])
    with c1:
        risk_badge(risk["risk_level"])
    with c2:
        st.write(f"**{t('referral', lang)}:** {risk['referral']}")
    with st.expander("Why this risk level? (rationale)"):
        for line in risk["rationale"]:
            st.write(f"- {line}")
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Step 5: Save --------------------------------------------------------
    if st.button(t("save_result", lang), type="primary"):
        image_path = save_uploaded_image(st.session_state.wf_image_bgr, UPLOAD_DIR)
        heatmap_path = save_uploaded_image(st.session_state.wf_heatmap, HEATMAP_DIR)
        add_screening(
            patient_id=patient_id,
            image_path=image_path,
            dr_class=prediction["dr_class"],
            confidence=prediction["confidence"],
            risk_level=risk["risk_level"],
            referral=risk["referral"],
            heatmap_path=heatmap_path,
            screened_by=st.session_state.user["display_name"],
        )
        st.success(t("saved_success", lang))
        reset_workflow()
        st.session_state.wf_patient_id = patient_id


# ---------------------------------------------------------------------------
# Page: Doctor Dashboard + Follow-up tracking
# ---------------------------------------------------------------------------
def page_dashboard():
    lang = st.session_state.lang
    st.header(f"🩺 {t('dashboard_title', lang)}")

    screenings = get_screenings()
    if not screenings:
        st.info("No screenings recorded yet.")
        return

    df = pd.DataFrame(screenings)

    # --- KPI row -----------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Screenings", len(df))
    k2.metric("High/Critical Risk", int(df["risk_level"].isin(["High", "Critical"]).sum()))
    k3.metric("Pending Follow-up", int((df["followup_status"] == "Pending").sum()))
    k4.metric("Avg. Confidence", f"{df['confidence'].mean():.1f}%")

    tab_overview, tab_followup = st.tabs([t("dashboard_title", lang), t("nav_followup", lang)])

    # --- Overview tab --------------------------------------------------------
    with tab_overview:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='drishti-card'>", unsafe_allow_html=True)
            st.markdown("##### DR Class Distribution")
            st.bar_chart(df["dr_class"].value_counts())
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='drishti-card'>", unsafe_allow_html=True)
            st.markdown("##### Risk Level Distribution")
            st.bar_chart(df["risk_level"].value_counts())
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='drishti-card'>", unsafe_allow_html=True)
        st.markdown("##### All Screenings")
        show_df = df[[
            "patient_code", "patient_name", "age", "dr_class", "confidence",
            "risk_level", "referral", "followup_status", "screened_by", "screened_at",
        ]].rename(columns={
            "patient_code": "Patient ID", "patient_name": "Name", "age": "Age",
            "dr_class": "DR Class", "confidence": "Confidence (%)", "risk_level": "Risk",
            "referral": "Referral", "followup_status": "Follow-up", "screened_by": "Screened By",
            "screened_at": "Screened At",
        })
        st.dataframe(show_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Follow-up tracking tab -----------------------------------------------
    with tab_followup:
        st.markdown("<div class='drishti-card'>", unsafe_allow_html=True)
        st.caption("Update follow-up status per screening (Doctor role).")
        for row in screenings:
            with st.container():
                cols = st.columns([2, 2, 1.5, 1.5, 2])
                cols[0].write(f"**{row['patient_name']}** ({row['patient_code']})")
                cols[1].write(row["dr_class"])
                with cols[2]:
                    risk_badge(row["risk_level"])
                cols[3].write(row["screened_at"][:10])
                current_status = row["followup_status"]
                new_status = cols[4].selectbox(
                    t("followup_status", lang),
                    FOLLOWUP_STATUSES,
                    index=FOLLOWUP_STATUSES.index(current_status),
                    key=f"status_{row['id']}",
                    label_visibility="collapsed",
                    disabled=(st.session_state.user["role"] != "Doctor"),
                )
                if new_status != current_status and st.session_state.user["role"] == "Doctor":
                    update_followup_status(row["id"], new_status)
                    st.rerun()
            st.divider()
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
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
    elif page == "Dashboard":
        page_dashboard()


if __name__ == "__main__":
    main()
