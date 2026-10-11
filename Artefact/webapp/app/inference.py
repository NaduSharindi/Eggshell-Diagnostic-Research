"""
inference.py
Model loading and prediction for the web application.

One `Predictor` instance is created at application start and shared by every request. Loading the
three-seed ensemble takes several seconds and allocates a few hundred megabytes, so doing it per
request would be both slow and a memory leak under load.

Preprocessing is imported from `common.preprocess`, the same module that built the training cache
and that the Streamlit application uses. It is deliberately not reimplemented here: a divergence
between training and serving preprocessing produces confident wrong answers rather than visible
errors, which is the worst failure mode a diagnostic tool can have.

PyTorch is imported lazily so the application can start, serve its health check and report a clear
error if the models are missing, rather than failing at import time with a stack trace.
"""
from __future__ import annotations

import base64
import io
import logging
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


class _Settings:
    """Attribute view over a Flask config mapping.

    `app.config` is a dict subclass, so `cfg.TRAIN_SIDE` raises AttributeError while
    `cfg["TRAIN_SIDE"]` works. Wrapping it here keeps the call sites readable and means a missing
    key fails with a message naming the key instead of a bare AttributeError. The wrapper also
    accepts a plain object, so tests can pass a SimpleNamespace.
    """

    __slots__ = ("_m",)

    def __init__(self, mapping):
        object.__setattr__(self, "_m", mapping)

    def __getattr__(self, name):
        m = object.__getattribute__(self, "_m")
        try:
            return m[name] if hasattr(m, "__getitem__") else getattr(m, name)
        except (KeyError, TypeError, AttributeError):
            raise AttributeError(
                f"configuration key '{name}' is not set. Check config.py and any "
                f"environment overrides.") from None

    def get(self, name, default=None):
        try:
            return getattr(self, name)
        except AttributeError:
            return default


@dataclass
class Prediction:
    status: str                 # "Intact" | "Micro_cracked"
    pattern: Optional[str]      # None when intact
    crack_probability: float
    confidence: float
    needs_review: bool
    probabilities: dict         # class name -> probability
    elapsed_ms: float

    def as_dict(self) -> dict:
        d = asdict(self)
        d["crack_probability"] = round(self.crack_probability, 4)
        d["confidence"] = round(self.confidence, 4)
        d["probabilities"] = {k: round(v, 4) for k, v in self.probabilities.items()}
        d["elapsed_ms"] = round(self.elapsed_ms, 1)
        return d


class ModelsUnavailable(RuntimeError):
    """Raised when no checkpoint could be loaded. Carries a message safe to show a user."""


class Predictor:
    def __init__(self, cfg):
        self.cfg = _Settings(cfg)
        self._models = []
        self._labels: list[str] = []
        self._patterns: list[str] = []
        self._lock = threading.Lock()
        self._loaded = False
        self._error: Optional[str] = None
        self._torch = None
        self._F = None

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        with self._lock:
            if self._loaded or self._error:
                return
            try:
                import torch
                import torch.nn.functional as F
                from common import project_config as CFG
                from common.models import build_model

                torch.set_num_threads(max(1, int(self.cfg.TORCH_THREADS)))
                # NOTE: torch.set_grad_enabled is thread-local, so setting it here would only
                # affect the thread that loaded the models. Gunicorn serves each request on a
                # different thread, where gradients would be enabled again and the forward pass
                # would build a graph that .numpy() then refuses to convert. Autograd is
                # therefore disabled per call inside predict(), not globally here.
                self._torch, self._F = torch, F
                self._labels = list(CFG.LABEL4)
                self._patterns = list(CFG.PATTERNS)

                md = self.cfg.get("MODEL_DIR")
                d = Path(md) if md else CFG.MODELS_DIR / "final"
                found = []
                for s in self.cfg.SEEDS:
                    p = d / f"final_{self.cfg.BACKBONE}_{self.cfg.FORMULATION}_seed{s}.pt"
                    if not p.exists():
                        log.warning("checkpoint missing: %s", p)
                        continue
                    sd = torch.load(p, map_location="cpu", weights_only=False)
                    m = build_model(self.cfg.BACKBONE, self.cfg.FORMULATION, 4,
                                    pretrained=False, age_mode="none",
                                    img_size=self.cfg.TRAIN_SIDE)
                    m.load_state_dict(sd["state_dict"])
                    m.eval()
                    found.append(m)
                if not found:
                    raise ModelsUnavailable(
                        f"No model checkpoints found in {d}. Expected files named "
                        f"final_{self.cfg.BACKBONE}_{self.cfg.FORMULATION}_seed<N>.pt")
                self._models = found
                self._loaded = True
                log.info("loaded %d model(s) from %s", len(found), d)
            except ModelsUnavailable as e:
                self._error = str(e)
                log.error("%s", e)
            except Exception as e:  # noqa: BLE001
                self._error = f"{type(e).__name__}: {e}"
                log.exception("model load failed")

    @property
    def ready(self) -> bool:
        return self._loaded

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def n_models(self) -> int:
        return len(self._models)

    def require(self) -> None:
        if not self._loaded:
            self.load()
        if not self._loaded:
            raise ModelsUnavailable(self._error or "models not loaded")

    # --------------------------------------------------------------- predict
    def preprocess(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        from common.preprocess import preprocess_bgr
        return preprocess_bgr(img_bgr)

    def predict(self, rgb448: np.ndarray, threshold: float, low_conf: float):
        """Return (Prediction, segmentation map in [0, 1])."""
        import time
        from common.preprocess import to_model_input
        self.require()
        torch, F = self._torch, self._F

        t0 = time.perf_counter()
        x = torch.from_numpy(to_model_input(rgb448, self.cfg.TRAIN_SIDE))
        p4s, segs = [], []
        # no_grad is applied here, per call, because it is thread-local and this method runs on
        # whichever worker thread handled the request
        with torch.no_grad():
            for m in self._models:
                feats = m.backbone(x)
                pooled = m.drop(F.adaptive_avg_pool2d(feats[-1], 1).flatten(1))
                det = torch.softmax(m.head_det(pooled), 1)
                pat = torch.softmax(m.head_pat(pooled), 1)
                p4s.append(torch.cat([det[:, :1], det[:, 1:2] * pat], 1)[0]
                           .detach().cpu().numpy())
                segs.append(torch.sigmoid(m.seg_head(feats))[0, 0].detach().cpu().numpy())
        p4 = np.mean(p4s, axis=0)
        seg = np.mean(segs, axis=0)
        seg = cv2.resize(seg, (self.cfg.TRAIN_SIDE, self.cfg.TRAIN_SIDE),
                         interpolation=cv2.INTER_LINEAR)
        seg = seg / (seg.max() + 1e-9)
        elapsed = (time.perf_counter() - t0) * 1000.0

        p_cracked = float(1.0 - p4[0])
        cracked = p_cracked >= threshold
        pat_p = p4[1:] / max(float(p4[1:].sum()), 1e-9)
        pattern = self._patterns[int(np.argmax(pat_p))] if cracked else None
        conf = float(p_cracked * float(pat_p.max())) if cracked else float(p4[0])

        pred = Prediction(
            status="Micro_cracked" if cracked else "Intact",
            pattern=pattern,
            crack_probability=p_cracked,
            confidence=conf,
            needs_review=bool(conf < low_conf),
            probabilities={lab: float(v) for lab, v in zip(self._labels, p4)},
            elapsed_ms=elapsed,
        )
        return pred, seg


# ----------------------------------------------------------------- rendering
def overlay_png(rgb: np.ndarray, seg: np.ndarray, alpha: float = 0.45) -> str:
    base = cv2.resize(rgb, seg.shape[::-1], interpolation=cv2.INTER_AREA)
    hm = cv2.applyColorMap((np.clip(seg, 0, 1) * 255).astype(np.uint8),
                           cv2.COLORMAP_JET)[:, :, ::-1]
    blend = (base * (1 - alpha) + hm * alpha).astype(np.uint8)
    return to_data_uri(blend)


def to_data_uri(rgb: np.ndarray, max_side: int = 640, quality: int = 86) -> str:
    """Encode an RGB array as a data URI so no uploaded pixel is ever written to disk."""
    h, w = rgb.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def decode_upload(data: bytes) -> Optional[np.ndarray]:
    """Decode uploaded bytes to BGR, returning None rather than raising on malformed input."""
    if not data:
        return None
    try:
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        return None
    if img is None or img.size == 0:
        return None
    return img
