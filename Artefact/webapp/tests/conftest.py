"""
conftest.py
Test fixtures for the Flask application.

The predictor is replaced by a deterministic stub so the web layer can be tested without PyTorch,
model weights or a GPU. This is what lets the CI integration job run on a stock GitHub runner in
seconds. The model itself is tested separately by Artefact/tests/test_artefact.py, which runs the
real checkpoints wherever they exist.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

WEBAPP = Path(__file__).resolve().parents[1]
ARTEFACT = WEBAPP.parent
sys.path.insert(0, str(WEBAPP))
sys.path.insert(0, str(ARTEFACT / "src"))

RULE_TABLE = {
    ("Hairline", "Early lay"): 30, ("Hairline", "Peak production"): 40, ("Hairline", "Late lay"): 55,
    ("Star", "Early lay"): 15, ("Star", "Peak production"): 20, ("Star", "Late lay"): 30,
    ("Network", "Early lay"): 50, ("Network", "Peak production"): 60, ("Network", "Late lay"): 75,
}


def _install_stubs(pattern="Star", p_intact=0.05):
    import app.inference as inf
    import app.rules as rules_mod
    from app.inference import Prediction
    from app.rules import RuleBase

    class StubPredictor:
        def __init__(self, cfg): self.cfg = cfg
        def load(self): pass
        ready = property(lambda self: True)
        error = property(lambda self: None)
        n_models = property(lambda self: 3)
        def require(self): pass

        def preprocess(self, bgr):
            if bgr is None or getattr(bgr, "size", 0) == 0:
                return None
            return np.full((448, 448, 3), 180, np.uint8)

        def predict(self, rgb, threshold, low_conf):
            others = (1 - p_intact) / 3
            probs = {"Intact": p_intact, "Hairline": others, "Star": others, "Network": others}
            probs[pattern] = (1 - p_intact) - 2 * others * 0.5
            rest = [k for k in ("Hairline", "Star", "Network") if k != pattern]
            for k in rest:
                probs[k] = others * 0.5
            p_cr = 1 - p_intact
            cracked = p_cr >= threshold
            pat_vals = np.array([probs["Hairline"], probs["Star"], probs["Network"]])
            pat_vals = pat_vals / pat_vals.sum()
            conf = float(p_cr * pat_vals.max()) if cracked else p_intact
            pred = Prediction("Micro_cracked" if cracked else "Intact",
                              pattern if cracked else None, p_cr, conf,
                              conf < low_conf, probs, 12.0)
            seg = np.zeros((384, 384), np.float32)
            seg[150:230, 150:260] = 1.0
            return pred, seg

    class StubRules(RuleBase):
        def load(self):
            self.patterns = ["Hairline", "Star", "Network"]
            self.age_classes = ["Early lay", "Peak production", "Late lay"]
            self._rules = {k: {"phys": float(v), "mech": 100.0 - v,
                               "interpretation": f"Interpretation for {k[0]} in {k[1]}.",
                               "actions": f"Actions for {k[0]} in {k[1]} (about 3.2% Ca)."}
                           for k, v in RULE_TABLE.items()}
            self._error = None

    inf.Predictor = StubPredictor
    rules_mod.RuleBase = StubRules


@pytest.fixture()
def app():
    import os
    os.environ.setdefault("SECRET_KEY", "test")
    _install_stubs()
    from app import create_app
    from config import DevelopmentConfig
    application = create_app(DevelopmentConfig)
    application.config.update(TESTING=True)
    return application


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def egg_jpg():
    def _make(seed=0):
        img = np.zeros((900, 900, 3), np.uint8)
        cv2.ellipse(img, (450, 450), (300, 380), 0, 0, 360, (60, 150, 235), -1)
        return cv2.imencode(".jpg", img)[1].tobytes()
    return _make


def upload(data: bytes, name: str):
    return (io.BytesIO(data), name)
