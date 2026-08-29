
"""
core/risk_engine.py
---------------------
Combines the AI's DR severity prediction with basic patient risk
factors (diabetes duration, prior screening history) to produce a
Low/Medium/High/Critical risk level and a referral recommendation.

This is a transparent, rule-based layer on purpose: in a clinical
screening tool, the logic that decides "who gets referred" should be
auditable and easy to reason about, not another opaque model.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RISK_LEVELS, REFERRAL_ACTIONS  # noqa: E402

# Base risk purely from DR class (before adjusting for patient factors)
_BASE_RISK_BY_CLASS = {
    "No DR": "Low",
    "Mild DR": "Medium",
    "Moderate DR": "High",
    "Severe DR": "Critical",
}


def _escalate(risk: str, steps: int = 1) -> str:
    """Move `risk` up the RISK_LEVELS ladder by `steps`, capped at Critical."""
    idx = RISK_LEVELS.index(risk)
    new_idx = min(idx + steps, len(RISK_LEVELS) - 1)
    return RISK_LEVELS[new_idx]


def assess_risk(dr_class: str, confidence: float, diabetes_duration: int,
                 previous_screening: str) -> dict:
    """
    Args:
        dr_class: one of config.DR_CLASSES
        confidence: model confidence 0-100
        diabetes_duration: years since diabetes diagnosis
        previous_screening: "Yes" or "No"

    Returns:
        {
            "risk_level": str,
            "referral": str,
            "rationale": [str, ...]   # human-readable reasons, for transparency
        }
    """
    risk = _BASE_RISK_BY_CLASS.get(dr_class, "Medium")
    rationale = [f"AI classified image as '{dr_class}' -> base risk '{risk}'."]

    # Long-standing diabetes increases risk of rapid progression.
    if diabetes_duration >= 15:
        risk = _escalate(risk, 1)
        rationale.append(f"Diabetes duration {diabetes_duration}y (>=15y) -> risk escalated.")
    elif diabetes_duration >= 10:
        rationale.append(f"Diabetes duration {diabetes_duration}y (10-15y) -> monitor closely, no escalation.")

    # Never screened before + any DR detected -> escalate, since the
    # patient has no screening history to compare against / may have
    # undetected earlier progression.
    if previous_screening == "No" and dr_class != "No DR":
        risk = _escalate(risk, 1)
        rationale.append("No previous screening on record with detected DR -> risk escalated.")

    # Low model confidence on a concerning class should not be
    # silently downgraded — flag for human review instead of trusting
    # a low-confidence "safe" reading.
    if confidence < 60 and dr_class != "No DR":
        rationale.append(f"Model confidence low ({confidence:.1f}%) — recommend manual review by doctor.")

    referral = REFERRAL_ACTIONS[risk]
    return {"risk_level": risk, "referral": referral, "rationale": rationale}
