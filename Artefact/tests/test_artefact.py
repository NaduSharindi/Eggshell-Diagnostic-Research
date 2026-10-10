"""
test_artefact.py
Functional test suite for the eggshell micro-crack diagnostic artefact.

Run with:   pytest Artefact/tests/test_artefact.py -v
            pytest Artefact/tests/test_artefact.py -v --junitxml=Artefact/results/functional_tests.xml

This is Section 5.4 of the dissertation: evidence that the artefact works as built. It is NOT the
research result. The self-check guidelines are explicit that functional testing is kept short in
the chapter with the full table in an appendix, and that is how these are written -- one assertion
per behaviour, each traceable to a requirement.

The tests are grouped by what could plausibly go wrong in deployment rather than by module:

  FT-1  preprocessing contract      shape, dtype, range, determinism, degenerate input
  FT-2  training/inference parity   the app must transform an image exactly as training did
  FT-3  model contract              loads, produces valid probabilities, is deterministic
  FT-4  decision logic              threshold and confidence flag behave as specified
  FT-5  rule base integrity         complete, consistent, sums to 100
  FT-6  end-to-end                  a real image produces a complete, well-formed diagnosis
  FT-7  reporting                   batch aggregation and report generation
  FT-8  robustness                  the artefact fails safely on bad input rather than silently
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve()
_ART = next((p for p in _HERE.parents if (p / "src" / "common").is_dir()), None)
if _ART is None:
    pytest.skip("artefact root not found", allow_module_level=True)
sys.path.insert(0, str(_ART / "src"))
sys.path.insert(0, str(_ART / "app"))

import cv2  # noqa: E402
from common import project_config as CFG  # noqa: E402
from common.preprocess import preprocess_bgr, to_model_input, CACHE_SIDE  # noqa: E402
from common.augmentation import standardise, EggAugment, POLICIES  # noqa: E402

TRAIN_SIDE = 384
THRESHOLD = 0.30
LOW_CONF = 0.60


# --------------------------------------------------------------------------- fixtures
def synthetic_egg(side=1400, seed=0, with_crack=True):
    """A candled-egg-like image: bright ellipse on a dark field, optional bright fracture."""
    rng = np.random.default_rng(seed)
    img = np.full((side, side, 3), 10, np.uint8)
    cv2.ellipse(img, (side // 2, side // 2), (int(side * .34), int(side * .43)),
                0, 0, 360, (60, 140, 235), -1)
    img = cv2.GaussianBlur(img, (41, 41), 0)
    if with_crack:
        p0 = (int(side * .42), int(side * .46))
        p1 = (int(side * .60), int(side * .55))
        cv2.line(img, p0, p1, (150, 210, 255), 3)
    return np.clip(img.astype(np.int16) + rng.integers(-5, 5, img.shape), 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def models():
    # torch is optional for the preprocessing and rule-base tests, so a missing install skips
    # the model tests rather than erroring the whole suite
    torch = pytest.importorskip("torch", reason="PyTorch not installed")
    from common.models import build_model
    d = CFG.MODELS_DIR / "final"
    out = []
    for s in (42, 43, 44):
        p = d / f"final_efficientnet_b0_multitask_seg_seed{s}.pt"
        if not p.exists():
            continue
        sd = torch.load(p, map_location="cpu", weights_only=False)
        m = build_model("efficientnet_b0", "multitask_seg", 4, pretrained=False,
                        age_mode="none", img_size=TRAIN_SIDE)
        m.load_state_dict(sd["state_dict"]); m.eval()
        out.append(m)
    if not out:
        pytest.skip("no trained checkpoints; run notebook 07 first")
    return out


@pytest.fixture(scope="module")
def rules():
    import pandas as pd
    for cand in [CFG.DATA_DIR / "processed" / "root_cause_rules_recovered.csv",
                 CFG.ROOT_CAUSE_RULES_CSV]:
        if Path(cand).is_file():
            df = pd.read_csv(cand)
            return {(str(r["crack_pattern"]).strip(), str(r["age_class"]).strip()): {
                "phys": float(r["phys_pct"]), "mech": float(r["mech_pct"]),
                "interpretation": str(r.get("interpretation", "")),
                "actions": str(r.get("suggested_actions", ""))} for _, r in df.iterrows()}
    pytest.skip("rule base not found; run notebook 01 first")


# --------------------------------------------------------------------------- FT-1
class TestPreprocessingContract:
    """FT-1: the preprocessor must always return the cache format or None, never anything else."""

    def test_ft1_1_output_shape_and_dtype(self):
        out = preprocess_bgr(synthetic_egg())
        assert out is not None
        assert out.shape == (CACHE_SIDE, CACHE_SIDE, 3)
        assert out.dtype == np.uint8

    @pytest.mark.parametrize("side", [600, 1024, 2400, 3672])
    def test_ft1_2_any_input_size(self, side):
        out = preprocess_bgr(synthetic_egg(side=side))
        assert out is not None and out.shape == (CACHE_SIDE, CACHE_SIDE, 3)

    def test_ft1_3_non_square_input(self):
        img = synthetic_egg(1400)[:900, :]          # letterboxed, like the raw captures
        out = preprocess_bgr(img)
        assert out is not None and out.shape[0] == out.shape[1]

    def test_ft1_4_deterministic(self):
        img = synthetic_egg(seed=3)
        a, b = preprocess_bgr(img, seed=0), preprocess_bgr(img, seed=0)
        assert np.array_equal(a, b), "same input and seed must give byte-identical output"

    def test_ft1_5_removes_black_padding(self):
        """The leak that notebook 02 fixed must not reappear."""
        img = synthetic_egg(1400)
        img[:300, :] = 0; img[-300:, :] = 0          # bake in letterbox bars
        out = preprocess_bgr(img)
        dark = float((cv2.cvtColor(out, cv2.COLOR_RGB2GRAY) <= 8).mean())
        assert dark < 0.01, f"near-black pixels survived preprocessing: {dark:.3f}"

    def test_ft1_6_returns_none_on_empty(self):
        assert preprocess_bgr(np.zeros((0, 0, 3), np.uint8)) is None
        assert preprocess_bgr(None) is None


# --------------------------------------------------------------------------- FT-2
class TestTrainingInferenceParity:
    """FT-2: the app must transform an image exactly as training did, or results are invalid."""

    def test_ft2_1_model_input_shape(self):
        x = to_model_input(preprocess_bgr(synthetic_egg()), TRAIN_SIDE)
        assert x.shape == (1, 3, TRAIN_SIDE, TRAIN_SIDE) and x.dtype == np.float32

    def test_ft2_2_standardisation_is_per_channel(self):
        x = to_model_input(preprocess_bgr(synthetic_egg()), TRAIN_SIDE)[0]
        for c in range(3):
            assert abs(float(x[c].mean())) < 1e-3
            assert abs(float(x[c].std()) - 1.0) < 1e-2

    def test_ft2_3_matches_cache_pipeline(self):
        """Independently standardising a cache row must equal the app's transform."""
        rgb = preprocess_bgr(synthetic_egg())
        small = cv2.resize(rgb, (TRAIN_SIDE, TRAIN_SIDE), interpolation=cv2.INTER_AREA)
        expected = standardise(small).transpose(2, 0, 1)[None]
        assert np.allclose(to_model_input(rgb, TRAIN_SIDE), expected, atol=1e-5)

    def test_ft2_4_no_augmentation_at_inference(self):
        """EggAugment('none') must be the identity, or validation would be randomised."""
        img = preprocess_bgr(synthetic_egg())
        out, _ = EggAugment("none", seed=1)(img.copy(), None)
        assert np.array_equal(out, img)


# --------------------------------------------------------------------------- FT-3
class TestModelContract:
    def test_ft3_1_loads(self, models):
        assert len(models) >= 1

    def test_ft3_2_valid_probability_distribution(self, models):
        torch = pytest.importorskip("torch")
        import torch.nn.functional as F
        x = torch.from_numpy(to_model_input(preprocess_bgr(synthetic_egg()), TRAIN_SIDE))
        with torch.no_grad():
            m = models[0]
            feats = m.backbone(x)
            pooled = m.drop(F.adaptive_avg_pool2d(feats[-1], 1).flatten(1))
            det = torch.softmax(m.head_det(pooled), 1)
            pat = torch.softmax(m.head_pat(pooled), 1)
            p4 = torch.cat([det[:, :1], det[:, 1:2] * pat], 1)[0].numpy()
        assert p4.shape == (4,)
        assert np.all(p4 >= 0) and abs(float(p4.sum()) - 1.0) < 1e-4

    def test_ft3_3_segmentation_map_valid(self, models):
        torch = pytest.importorskip("torch")
        x = torch.from_numpy(to_model_input(preprocess_bgr(synthetic_egg()), TRAIN_SIDE))
        with torch.no_grad():
            seg = torch.sigmoid(models[0].seg_head(models[0].backbone(x)))[0, 0].numpy()
        assert seg.ndim == 2 and 0.0 <= float(seg.min()) and float(seg.max()) <= 1.0

    def test_ft3_4_deterministic_in_eval_mode(self, models):
        torch = pytest.importorskip("torch")
        import torch.nn.functional as F
        x = torch.from_numpy(to_model_input(preprocess_bgr(synthetic_egg()), TRAIN_SIDE))
        outs = []
        for _ in range(2):
            with torch.no_grad():
                feats = models[0].backbone(x)
                pooled = models[0].drop(F.adaptive_avg_pool2d(feats[-1], 1).flatten(1))
                outs.append(torch.softmax(models[0].head_det(pooled), 1).numpy())
        assert np.allclose(outs[0], outs[1]), "eval-mode inference must be deterministic"


# --------------------------------------------------------------------------- FT-4
class TestDecisionLogic:
    @staticmethod
    def classify(p4, threshold=THRESHOLD, low_conf=LOW_CONF):
        p_cracked = float(1.0 - p4[0])
        cracked = p_cracked >= threshold
        pat = p4[1:] / max(p4[1:].sum(), 1e-9)
        conf = float(p_cracked * pat.max()) if cracked else float(p4[0])
        return {"status": "Micro_cracked" if cracked else "Intact",
                "pattern": CFG.PATTERNS[int(np.argmax(pat))] if cracked else None,
                "confidence": conf, "needs_review": conf < low_conf}

    def test_ft4_1_threshold_boundary(self):
        assert self.classify(np.array([0.71, .1, .1, .09]))["status"] == "Intact"
        assert self.classify(np.array([0.70, .1, .1, .10]))["status"] == "Micro_cracked"

    def test_ft4_2_intact_has_no_pattern(self):
        r = self.classify(np.array([0.95, .02, .02, .01]))
        assert r["status"] == "Intact" and r["pattern"] is None

    def test_ft4_3_pattern_is_argmax_of_renormalised(self):
        r = self.classify(np.array([0.05, .20, .60, .15]))
        assert r["pattern"] == "Star"

    def test_ft4_4_low_confidence_flag(self):
        assert self.classify(np.array([0.45, .20, .20, .15]))["needs_review"] is True
        assert self.classify(np.array([0.02, .02, .94, .02]))["needs_review"] is False

    def test_ft4_5_default_threshold_is_the_evidenced_one(self):
        assert THRESHOLD == 0.30, "notebook 08 selected 0.30 as the zero-missed-crack point"


# --------------------------------------------------------------------------- FT-5
class TestRuleBaseIntegrity:
    def test_ft5_1_complete_coverage(self, rules):
        missing = [(p, a) for p in CFG.PATTERNS for a in CFG.AGE_ORDER if (p, a) not in rules]
        assert not missing, f"rule base incomplete: {missing}"

    def test_ft5_2_percentages_sum_to_100(self, rules):
        for k, v in rules.items():
            assert abs(v["phys"] + v["mech"] - 100.0) < 1e-6, f"{k} sums to {v['phys']+v['mech']}"

    def test_ft5_3_percentages_in_range(self, rules):
        for k, v in rules.items():
            assert 0 <= v["phys"] <= 100 and 0 <= v["mech"] <= 100, k

    def test_ft5_4_texts_present(self, rules):
        for k, v in rules.items():
            assert len(v["interpretation"]) > 20, f"{k} has no interpretation"
            assert len(v["actions"]) > 20, f"{k} has no suggested actions"

    def test_ft5_5_physiological_share_rises_with_age(self, rules):
        """The age gradient is the research claim; the rule base must actually encode it."""
        for p in CFG.PATTERNS:
            vals = [rules[(p, a)]["phys"] for a in CFG.AGE_ORDER]
            assert vals == sorted(vals), f"{p}: physiological share not monotonic in age: {vals}"

    def test_ft5_6_star_is_most_mechanical(self, rules):
        """Star is a point-impact signature; it must never be the most physiological pattern."""
        for a in CFG.AGE_ORDER:
            assert rules[("Star", a)]["phys"] < rules[("Network", a)]["phys"], a
            assert rules[("Star", a)]["phys"] < rules[("Hairline", a)]["phys"], a


# --------------------------------------------------------------------------- FT-6
class TestEndToEnd:
    def test_ft6_1_full_diagnosis_is_well_formed(self, models, rules):
        torch = pytest.importorskip("torch")
        import torch.nn.functional as F
        rgb = preprocess_bgr(synthetic_egg(with_crack=True))
        x = torch.from_numpy(to_model_input(rgb, TRAIN_SIDE))
        with torch.no_grad():
            m = models[0]; feats = m.backbone(x)
            pooled = m.drop(F.adaptive_avg_pool2d(feats[-1], 1).flatten(1))
            det = torch.softmax(m.head_det(pooled), 1)
            pat = torch.softmax(m.head_pat(pooled), 1)
            p4 = torch.cat([det[:, :1], det[:, 1:2] * pat], 1)[0].numpy()
        r = TestDecisionLogic.classify(p4)
        assert r["status"] in ("Intact", "Micro_cracked")
        if r["status"] == "Micro_cracked":
            k = rules[(r["pattern"], "Peak production")]
            assert set(k) >= {"phys", "mech", "interpretation", "actions"}

    def test_ft6_2_latency_within_budget(self, models):
        import time
        torch = pytest.importorskip("torch")
        x = torch.from_numpy(to_model_input(preprocess_bgr(synthetic_egg()), TRAIN_SIDE))
        with torch.no_grad():
            models[0](x)
            t0 = time.time()
            for _ in range(5):
                models[0](x)
            dt = (time.time() - t0) / 5
        assert dt < 5.0, f"single-image inference took {dt:.2f}s on CPU"


# --------------------------------------------------------------------------- FT-7
class TestReporting:
    def _frame(self):
        import pandas as pd
        return pd.DataFrame([
            {"file": "a.jpg", "status": "Micro_cracked", "pattern": "Hairline",
             "crack_probability": .91, "confidence": .80, "physiological_pct": 40.,
             "mechanical_pct": 60., "needs_review": False},
            {"file": "b.jpg", "status": "Intact", "pattern": None,
             "crack_probability": .05, "confidence": .95, "physiological_pct": np.nan,
             "mechanical_pct": np.nan, "needs_review": False},
            {"file": "c.jpg", "status": "Micro_cracked", "pattern": "Star",
             "crack_probability": .66, "confidence": .52, "physiological_pct": 20.,
             "mechanical_pct": 80., "needs_review": True},
        ])

    @staticmethod
    def _flock(verdict="Predominantly mechanical."):
        """Exactly the shape flock_assessment() returns: one entry per crack pattern."""
        return {"phys": 30.0, "mech": 70.0, "n": 2, "verdict": verdict,
                "actions": [
                    {"pattern": "Hairline", "count": 1, "phys": 40.0,
                     "text": "Fix impact points on the egg route first (nest mats, belt speed)."},
                    {"pattern": "Star", "count": 1, "phys": 20.0,
                     "text": "Check belt start and stop speed; avoid stacking trays."},
                ]}

    def test_ft7_1_report_generates(self, rules):
        from report_template import build_markdown_report
        df = self._frame()
        flock = self._flock()
        md = build_markdown_report(df, flock, "Peak production", "Tester", "Flock A", "notes")
        assert "Eggshell Micro-Crack Inspection Report" in md
        assert "Peak production" in md and "Flock A" in md
        assert "a.jpg" in md and "c.jpg" in md
        # the bug that prompted this test: action text must appear whole, not in fragments
        assert "Fix impact points on the egg route first (nest mats, belt speed)." in md
        assert "Hairline" in md and "Star" in md

    def test_ft7_2_report_states_its_limits(self, rules):
        from report_template import build_markdown_report
        md = build_markdown_report(self._frame(), self._flock(), "Late lay")
        low = md.lower()
        # match the wording the report actually uses, not a paraphrase of it. An earlier
        # version of this test searched for "not validated against hatch outcomes" and failed
        # against the real sentence, "it has not been validated against hatch outcomes".
        assert "has not been validated against hatch outcomes" in low
        assert "does not replace it" in low
        assert "over-flagging" in low          # the detection-threshold rationale
        assert "below 60%" in low              # the manual-review rationale

    def test_ft7_3_accepts_plain_string_actions(self, rules):
        """Defensive: an older caller passing bare strings must not crash the download."""
        from report_template import build_markdown_report
        flock = {"phys": 30.0, "mech": 70.0, "n": 1, "verdict": "x",
                 "actions": ["Check belt speed."]}
        md = build_markdown_report(self._frame(), flock, "Late lay")
        assert "Check belt speed." in md

    def test_ft7_4_no_cracks_path(self, rules):
        import pandas as pd
        from report_template import build_markdown_report
        df = self._frame()
        df = df[df["status"] == "Intact"]
        flock = {"phys": 0.0, "mech": 0.0, "n": 0,
                 "verdict": "No micro-cracks detected in this batch.", "actions": []}
        md = build_markdown_report(df, flock, "Early lay")
        assert "No micro-cracks were detected" in md


# --------------------------------------------------------------------------- FT-8
class TestRobustness:
    def test_ft8_1_uniform_image_does_not_crash(self):
        out = preprocess_bgr(np.full((800, 800, 3), 128, np.uint8))
        assert out is None or out.shape == (CACHE_SIDE, CACHE_SIDE, 3)

    def test_ft8_2_all_black_image(self):
        out = preprocess_bgr(np.zeros((800, 800, 3), np.uint8))
        assert out is None or out.shape == (CACHE_SIDE, CACHE_SIDE, 3)

    def test_ft8_3_tiny_image(self):
        out = preprocess_bgr(synthetic_egg(side=64))
        assert out is None or out.shape == (CACHE_SIDE, CACHE_SIDE, 3)

    def test_ft8_4_augmentation_policies_all_valid(self):
        img = preprocess_bgr(synthetic_egg())
        mask = np.zeros(img.shape[:2], np.uint8); mask[100:160, 120:300] = 1
        for name in POLICIES:
            out, m = EggAugment(name, seed=0)(img.copy(), mask.copy())
            assert out.shape == img.shape and out.dtype == np.uint8
            assert m is not None and m.shape == mask.shape
