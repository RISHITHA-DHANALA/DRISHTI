
"""
core/image_quality.py
----------------------
Basic image quality gating using OpenCV, run BEFORE the fundus image
is sent to the DR classifier. Rejects images that are too blurry, too
dark/bright, or too low resolution — mirroring real-world screening
workflows where camera operators need immediate recapture feedback.
"""

import cv2
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    BLUR_LAPLACIAN_THRESHOLD,
    MIN_BRIGHTNESS,
    MAX_BRIGHTNESS,
    MIN_RESOLUTION,
)


def _to_gray(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def compute_blur_score(image_bgr: np.ndarray) -> float:
    """
    Variance of the Laplacian is a standard, cheap blur metric:
    a sharp image has lots of high-frequency edge content -> high variance.
    A blurry image has smoothed-out edges -> low variance.
    """
    gray = _to_gray(image_bgr)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_brightness(image_bgr: np.ndarray) -> float:
    """Mean pixel intensity (0-255) on the grayscale image."""
    gray = _to_gray(image_bgr)
    return float(np.mean(gray))


def check_image_quality(image_bgr: np.ndarray) -> dict:
    """
    Run all quality checks and return a structured verdict:
        {
            "passed": bool,
            "blur_score": float,
            "brightness": float,
            "resolution": (w, h),
            "issues": [list of translation keys for failed checks],
        }
    The caller (UI layer) maps issue keys to localized user-facing messages.
    """
    h, w = image_bgr.shape[:2]
    issues = []

    blur_score = compute_blur_score(image_bgr)
    if blur_score < BLUR_LAPLACIAN_THRESHOLD:
        issues.append("blur_detected")

    brightness = compute_brightness(image_bgr)
    if brightness < MIN_BRIGHTNESS:
        issues.append("brightness_low")
    elif brightness > MAX_BRIGHTNESS:
        issues.append("brightness_high")

    if w < MIN_RESOLUTION[0] or h < MIN_RESOLUTION[1]:
        issues.append("resolution_low")

    return {
        "passed": len(issues) == 0,
        "blur_score": round(blur_score, 2),
        "brightness": round(brightness, 2),
        "resolution": (w, h),
        "issues": issues,
    }
