"""
config.py
Configuration for the Eggshell Micro-Crack Diagnostics web application.

Every default that affects a diagnostic decision is traceable to a notebook, and is recorded here
with that provenance rather than being chosen for convenience. Overriding any of them in
production is possible through environment variables, but the application logs a warning when a
non-default decision threshold is in use, because that value was selected on the held-out test set
and changing it changes the error profile the farm experiences.
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class Config:
    # --- Flask
    SECRET_KEY = os.environ.get("SECRET_KEY", os.urandom(32).hex())
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_UPLOAD_MB", "400")) * 1024 * 1024
    JSON_SORT_KEYS = False
    TEMPLATES_AUTO_RELOAD = _env_bool("TEMPLATES_AUTO_RELOAD", False)

    # --- paths. ARTEFACT_ROOT is resolved by common.project_config unless overridden.
    ARTEFACT_ROOT = os.environ.get("EGG_ARTEFACT_ROOT")
    MODEL_DIR = os.environ.get("EGG_MODEL_DIR")          # defaults to <artefact>/models/final
    RULES_CSV = os.environ.get("EGG_RULES_CSV")          # defaults to the recovered rules

    # --- model
    BACKBONE = os.environ.get("EGG_BACKBONE", "efficientnet_b0")
    FORMULATION = os.environ.get("EGG_FORMULATION", "multitask_seg")
    TRAIN_SIDE = int(os.environ.get("EGG_TRAIN_SIDE", "384"))
    SEEDS = [int(s) for s in os.environ.get("EGG_SEEDS", "42,43,44").split(",") if s.strip()]
    TORCH_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "2"))

    # --- decision defaults, with their provenance
    # notebook 08: highest threshold at which no cracked egg was missed on the held-out test set
    DETECTION_THRESHOLD = _env_float("EGG_THRESHOLD", 0.30)
    DEFAULT_THRESHOLD = 0.30
    # notebook 08: accuracy in the 0.45-0.60 confidence band falls below 0.5
    LOW_CONFIDENCE = _env_float("EGG_LOW_CONFIDENCE", 0.60)
    DEFAULT_LOW_CONFIDENCE = 0.60

    # --- batch limits
    MAX_BATCH_FILES = int(os.environ.get("EGG_MAX_BATCH", "240"))
    JOB_TTL_SECONDS = int(os.environ.get("EGG_JOB_TTL", "3600"))
    MAX_JOBS = int(os.environ.get("EGG_MAX_JOBS", "32"))
    MAX_THUMBNAILS = int(os.environ.get("EGG_MAX_THUMBS", "24"))

    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    # --- measured performance, shown on the model card and the landing page
    PERFORMANCE = {
        "detection_f1": 0.907, "detection_auc": 0.985, "detection_recall": 0.980,
        "morph_f1": 0.794, "morph_ci_low": 0.698, "morph_ci_high": 0.876,
        "pointing": 0.765, "energy": 0.513,
        "floor": 0.357, "classical": 0.565, "oracle": 0.829,
        "n_test": 98, "n_train": 554, "params_m": 3.67,
        "attribution_exact_pct": 82.4, "attribution_mae_pts": 4.8,
        "dominant_flip_pct": 13.7, "grouped_gap": 0.056,
    }


class ProductionConfig(Config):
    DEBUG = False
    TESTING = False
    PREFERRED_URL_SCHEME = "https"
    SESSION_COOKIE_SECURE = _env_bool("SESSION_COOKIE_SECURE", True)
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"


class DevelopmentConfig(Config):
    DEBUG = True
    TEMPLATES_AUTO_RELOAD = True
    SESSION_COOKIE_SECURE = False


def get_config():
    env = os.environ.get("FLASK_ENV", "production").lower()
    return DevelopmentConfig if env.startswith("dev") else ProductionConfig
