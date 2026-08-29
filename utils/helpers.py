
"""
utils/helpers.py
------------------
Small shared helpers: PIL<->OpenCV conversion, image saving, and a
couple of tiny Streamlit UI helpers (status badges) reused across pages.
"""

import os
import uuid
import numpy as np
import cv2
from PIL import Image
import streamlit as st

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RISK_COLORS, DR_CLASS_COLORS  # noqa: E402


def pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    """Convert a PIL RGB image to an OpenCV-style BGR numpy array."""
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_uploaded_image(image_bgr: np.ndarray, upload_dir: str) -> str:
    """Save an uploaded/processed image to disk with a unique filename; return its path."""
    os.makedirs(upload_dir, exist_ok=True)
    filename = f"{uuid.uuid4().hex[:12]}.png"
    filepath = os.path.join(upload_dir, filename)
    cv2.imwrite(filepath, image_bgr)
    return filepath


def render_badge(label: str, color: str):
    """Render a small colored pill/badge using inline HTML (Streamlit supports unsafe_allow_html)."""
    st.markdown(
        f"""
        <span style="
            background-color:{color}20;
            color:{color};
            border:1px solid {color};
            padding:4px 12px;
            border-radius:16px;
            font-weight:600;
            font-size:0.85rem;
        ">{label}</span>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(risk_level: str):
    render_badge(risk_level, RISK_COLORS.get(risk_level, "#607D8B"))


def dr_class_badge(dr_class: str):
    render_badge(dr_class, DR_CLASS_COLORS.get(dr_class, "#607D8B"))
