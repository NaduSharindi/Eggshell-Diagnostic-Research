"""utils.py — small shared helpers for the view and API layers."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from flask import current_app


def allowed_file(filename: str) -> bool:
    return Path(str(filename)).suffix.lower() in current_app.config["ALLOWED_EXTENSIONS"]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def read_threshold(form) -> tuple[float, float]:
    """Read the decision settings from a submitted form, falling back to the evidenced defaults."""
    cfg = current_app.config
    try:
        t = float(form.get("threshold", cfg["DETECTION_THRESHOLD"]))
    except (TypeError, ValueError):
        t = cfg["DETECTION_THRESHOLD"]
    try:
        lc = float(form.get("low_confidence", cfg["LOW_CONFIDENCE"]))
    except (TypeError, ValueError):
        lc = cfg["LOW_CONFIDENCE"]
    return clamp(t, 0.05, 0.95), clamp(lc, 0.0, 0.95)


def valid_age_class(value: str) -> str:
    """Never trust a submitted age class: it indexes the rule base."""
    rb = current_app.rulebase
    ages = rb.age_classes or ["Early lay", "Peak production", "Late lay"]
    v = str(value or "").strip()
    return v if v in ages else (ages[1] if len(ages) > 1 else ages[0])


def summarise(records: Iterable[dict]) -> dict:
    recs = list(records)
    readable = [r for r in recs if r.get("status") != "unreadable"]
    cracked = [r for r in readable if r.get("status") == "Micro_cracked"]
    review = [r for r in readable if r.get("needs_review")]
    return {
        "n_total": len(recs),
        "n_readable": len(readable),
        "n_cracked": len(cracked),
        "n_review": len(review),
        "n_unreadable": len(recs) - len(readable),
        "crack_rate": (len(cracked) / len(readable) * 100.0) if readable else 0.0,
    }
