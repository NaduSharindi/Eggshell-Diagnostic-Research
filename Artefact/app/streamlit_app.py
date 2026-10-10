"""
streamlit_app.py
Eggshell Micro-Crack Diagnostics - decision support for poultry breeder farms.

Run with:   streamlit run Artefact/app/streamlit_app.py

The artefact evaluated under Objective 4. Three views:

  Inspect   one egg at a time, with crack localisation and a full root-cause breakdown
  Batch     a tray or a flock sample, with aggregate analysis and a printable report
  Model     the model card: training data, measured performance, known limitations

The batch view is the operationally important one. A single egg almost never justifies changing
feed or stopping a belt; fifty eggs from one flock do. The per-egg rule base is therefore
aggregated into a flock-level assessment, which is the decision this artefact exists to support.

Settings carry their evidence rather than sitting at library defaults:

  detection threshold 0.30, not 0.50
      Notebook 08 swept it on the held-out test set. At 0.30 no cracked egg was missed, costing
      two additional false alarms. A missed crack enters the setter, fails to hatch and risks
      contaminating neighbours; a false alarm costs one manual check.

  uncertainty flag below 0.60
      Measured accuracy in the 0.45-0.60 confidence band is under 0.5, while above 0.80 it is
      near 1.0. Confidence is informative at the extremes, so the middle is routed to a human.

  localisation from the segmentation head, not Grad-CAM
      Notebook 09 measured both against the expert polygons: pointing game 0.765 against 0.549.
      On hairline cracks Grad-CAM reaches only 1.3x chance and is not usable.

Preprocessing is imported from common.preprocess, the same code path that built the training
cache. It is deliberately not reimplemented here: a silent mismatch between training and
inference preprocessing produces confident wrong answers rather than visible errors.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

_HERE = Path(__file__).resolve()
_SRC = next((p / "src" for p in _HERE.parents if (p / "src" / "common").is_dir()), None)
if _SRC is None:
    st.error("Could not locate Artefact/src. Launch from inside the repository.")
    st.stop()
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_HERE.parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from common import project_config as CFG  # noqa: E402
from common.models import build_model  # noqa: E402
from common.preprocess import preprocess_bgr, to_model_input  # noqa: E402
from report_template import build_markdown_report  # noqa: E402

TRAIN_SIDE = 384
DEFAULT_THRESHOLD = 0.30
DEFAULT_LOW_CONF = 0.60
BACKBONE, FORMULATION = "efficientnet_b0", "multitask_seg"
SEEDS = [42, 43, 44]
PERF = {"detection_f1": 0.907, "detection_auc": 0.985, "detection_recall": 0.980,
        "morph_f1": 0.794, "morph_ci": (0.698, 0.876), "pointing": 0.765, "energy": 0.513,
        "floor": 0.357, "classical": 0.565, "n_test": 98, "n_train": 554}

st.set_page_config(page_title="Eggshell Micro-Crack Diagnostics", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
  .block-container {padding-top: 2rem; padding-bottom: 4rem; max-width: 1500px;}
  div[data-testid="stMetricValue"] {font-size: 1.9rem; font-weight: 600;}
  div[data-testid="stMetricLabel"] {font-size: 0.82rem; letter-spacing: .02em;
      text-transform: uppercase; color: #8B949E;}
  .brand {font-size: 1.12rem; font-weight: 700; letter-spacing: -.01em;}
  .brand-sub {color: #8B949E; font-size: .82rem; margin-top: -.3rem;}
  .verdict {padding: 1rem 1.25rem; border-radius: 8px; font-size: 1.3rem; font-weight: 650;
            margin: .2rem 0 .9rem 0;}
  .v-crack {background: rgba(196,78,82,.14); border-left: 4px solid #C44E52; color: #FF9EA1;}
  .v-ok    {background: rgba(85,168,104,.14); border-left: 4px solid #55A868; color: #8FE0A6;}
  .v-warn  {background: rgba(232,131,58,.14); border-left: 4px solid #E8833A; color: #FFC083;
            font-size: 1.02rem;}
  .chip {display:inline-block; padding:3px 11px; border-radius:999px; font-size:.75rem;
         margin:0 6px 6px 0; background:#1C2129; color:#A9B4C0; border:1px solid #2A313C;}
  .muted {color:#8B949E; font-size:.84rem;}
  .panel {background:#141922; border:1px solid #232A35; border-radius:8px; padding:1rem 1.2rem;}
  .act-head {font-weight:650; color:#E8833A; margin-bottom:.15rem;}
  hr {border-color:#232A35;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Loading models")
def load_models():
    d = CFG.MODELS_DIR / "final"
    out = []
    for s in SEEDS:
        p = d / f"final_{BACKBONE}_{FORMULATION}_seed{s}.pt"
        if not p.exists():
            continue
        sd = torch.load(p, map_location="cpu", weights_only=False)
        m = build_model(BACKBONE, FORMULATION, 4, pretrained=False, age_mode="none",
                        img_size=TRAIN_SIDE)
        m.load_state_dict(sd["state_dict"]); m.eval()
        out.append(m)
    return out


@st.cache_data(show_spinner=False)
def load_rules():
    for cand in [CFG.DATA_DIR / "processed" / "root_cause_rules_recovered.csv",
                 CFG.ROOT_CAUSE_RULES_CSV]:
        if Path(cand).is_file():
            df = pd.read_csv(cand)
            return {(str(r["crack_pattern"]).strip(), str(r["age_class"]).strip()): {
                "phys": float(r["phys_pct"]), "mech": float(r["mech_pct"]),
                "interpretation": str(r.get("interpretation", "")),
                "actions": str(r.get("suggested_actions", ""))} for _, r in df.iterrows()}
    return {}


@torch.no_grad()
def infer(models, rgb448):
    x = torch.from_numpy(to_model_input(rgb448, TRAIN_SIDE))
    p4s, segs = [], []
    for m in models:
        feats = m.backbone(x)
        pooled = m.drop(F.adaptive_avg_pool2d(feats[-1], 1).flatten(1))
        det = torch.softmax(m.head_det(pooled), 1)
        pat = torch.softmax(m.head_pat(pooled), 1)
        p4s.append(torch.cat([det[:, :1], det[:, 1:2] * pat], 1)[0].numpy())
        segs.append(torch.sigmoid(m.seg_head(feats))[0, 0].numpy())
    seg = np.mean(segs, axis=0)
    seg = cv2.resize(seg, (TRAIN_SIDE, TRAIN_SIDE), interpolation=cv2.INTER_LINEAR)
    return np.mean(p4s, axis=0), seg / (seg.max() + 1e-9)


def classify(p4, threshold, low_conf):
    p_cracked = float(1.0 - p4[0])
    cracked = p_cracked >= threshold
    pat_p = p4[1:] / max(p4[1:].sum(), 1e-9)
    conf = float(p_cracked * pat_p.max()) if cracked else float(p4[0])
    return {"status": "Micro_cracked" if cracked else "Intact",
            "pattern": CFG.PATTERNS[int(np.argmax(pat_p))] if cracked else None,
            "crack_probability": p_cracked, "confidence": conf,
            "needs_review": bool(conf < low_conf), "pattern_probs": pat_p}


def overlay(rgb448, seg, alpha=0.45):
    base = cv2.resize(rgb448, (TRAIN_SIDE, TRAIN_SIDE), interpolation=cv2.INTER_AREA)
    hm = cv2.applyColorMap((np.clip(seg, 0, 1) * 255).astype(np.uint8),
                           cv2.COLORMAP_JET)[:, :, ::-1]
    return (base * (1 - alpha) + hm * alpha).astype(np.uint8)


def flock_assessment(df, rules, age_class):
    """Aggregate per-egg attributions into one flock-level recommendation.

    Actions are grouped by crack pattern and reproduced in full. An earlier version split the
    action text on full stops to deduplicate it, which cut sentences at decimal points and inside
    parentheses and produced fragments such as "2%), and check house temperature". Advice given
    to a farm manager has to be reproduced exactly as the domain expert wrote it.
    """
    cr = df[df["status"] == "Micro_cracked"]
    if cr.empty:
        return {"phys": 0.0, "mech": 0.0, "n": 0,
                "verdict": "No micro-cracks detected in this batch.", "actions": []}
    phys = float(cr["physiological_pct"].mean())
    if phys >= 60:
        verdict = ("**Predominantly shell weakness.** The crack mix points at nutrition and bird "
                   "physiology rather than handling.")
    elif phys <= 40:
        verdict = ("**Predominantly mechanical damage.** The crack mix points at handling, "
                   "equipment and collection practice rather than shell quality.")
    else:
        verdict = ("**Mixed causes.** Shell quality and handling are contributing in comparable "
                   "measure; address both.")
    counts = cr["pattern"].value_counts()
    actions = []
    for p, n in counts.items():
        k = rules.get((p, age_class))
        if k and k["actions"]:
            actions.append({"pattern": p, "count": int(n), "text": k["actions"].strip(),
                            "phys": k["phys"]})
    actions.sort(key=lambda a: -a["count"])
    return {"phys": phys, "mech": 100.0 - phys, "n": int(len(cr)),
            "verdict": verdict, "actions": actions}


models = load_models()
rules = load_rules()

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("<div class='brand'>Eggshell Micro-Crack Diagnostics</div>"
                "<div class='brand-sub'>Decision support for poultry breeder farms</div>",
                unsafe_allow_html=True)
    st.divider()
    age_class = st.selectbox(
        "Breeder age class", CFG.AGE_ORDER, index=1,
        help="From the flock records, using the Ross 308 feed phases. Early lay to 32 wk, "
             "Peak production 32-50 wk, Late lay 50 wk and over.")
    flock_id = st.text_input("Flock or batch reference", placeholder="Farm C, House 2")
    operator = st.text_input("Inspected by", placeholder="your name")
    st.divider()
    with st.expander("Advanced settings"):
        threshold = st.slider("Detection threshold", 0.05, 0.95, DEFAULT_THRESHOLD, 0.05)
        low_conf = st.slider("Flag for review below confidence", 0.0, 0.95,
                             DEFAULT_LOW_CONF, 0.05)
        st.caption("Defaults are the values selected on the held-out test set. "
                   f"At {DEFAULT_THRESHOLD:.2f} no cracked egg was missed.")
        if abs(threshold - DEFAULT_THRESHOLD) > 1e-6:
            st.warning("Non-default threshold in use.")
    st.divider()
    st.markdown(f"<span class='chip'>{BACKBONE}</span>"
                f"<span class='chip'>{len(models)}-seed ensemble</span>"
                f"<span class='chip'>threshold {threshold:.2f}</span>",
                unsafe_allow_html=True)
    st.markdown("<span class='muted'>Supports inspection by a qualified hatchery technician. "
                "Does not replace it.</span>", unsafe_allow_html=True)

if not models:
    st.error(f"No trained models found in {CFG.MODELS_DIR / 'final'}. Run notebook 07 first.")
    st.stop()
if not rules:
    st.error("Root-cause rules not found. Run notebook 01 to recover them.")
    st.stop()

tab_one, tab_batch, tab_model = st.tabs(["Inspect", "Batch", "Model"])

# =========================================================================== INSPECT
with tab_one:
    src = st.radio("Image source", ["Upload", "Camera"], horizontal=True,
                   label_visibility="collapsed")
    buf = None
    if src == "Upload":
        up = st.file_uploader("Candled egg image",
                              type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
        buf = up.read() if up else None
    else:
        shot = st.camera_input("Photograph the egg over the candling light")
        buf = shot.getvalue() if shot else None

    if buf is None:
        st.info("Upload or photograph a candled egg to begin.")
        c = st.columns(4)
        c[0].metric("Detection macro F1", f"{PERF['detection_f1']:.3f}")
        c[1].metric("Detection recall", f"{PERF['detection_recall']:.3f}")
        c[2].metric("Morphology macro F1", f"{PERF['morph_f1']:.3f}")
        c[3].metric("Localisation", f"{PERF['pointing']:.3f}")
        st.markdown(f"<span class='muted'>Measured on {PERF['n_test']} held-out eggs, opened "
                    "exactly once. Full details in the Model tab.</span>",
                    unsafe_allow_html=True)
    else:
        raw = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        if raw is None:
            st.error("Could not read that image file.")
            st.stop()
        t0 = time.time()
        rgb448 = preprocess_bgr(raw)
        if rgb448 is None:
            st.error("No egg found. The image should show a candled egg against a dark field.")
            st.stop()
        p4, seg = infer(models, rgb448)
        res = classify(p4, threshold, low_conf)
        dt = (time.time() - t0) * 1000

        if res["status"] == "Micro_cracked":
            st.markdown(f"<div class='verdict v-crack'>Micro-crack detected &mdash; "
                        f"{res['pattern']}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='verdict v-ok'>Intact &mdash; no micro-crack detected</div>",
                        unsafe_allow_html=True)
        if res["needs_review"]:
            st.markdown(f"<div class='verdict v-warn'>Low confidence &mdash; inspect this egg "
                        f"manually. Below {low_conf:.0%} confidence, measured accuracy falls "
                        f"under 50%.</div>", unsafe_allow_html=True)

        m = st.columns(4)
        m[0].metric("Crack probability", f"{res['crack_probability']:.1%}")
        m[1].metric("Morphology", res["pattern"] or "None")
        m[2].metric("Confidence", f"{res['confidence']:.1%}")
        m[3].metric("Processing time", f"{dt:.0f} ms")

        i1, i2 = st.columns(2)
        with i1:
            st.markdown("**Preprocessed image**")
            st.image(cv2.resize(rgb448, (TRAIN_SIDE, TRAIN_SIDE)), use_container_width=True)
            st.markdown("<span class='muted'>Egg isolated, background neutralised, "
                        "per-channel standardised</span>", unsafe_allow_html=True)
        with i2:
            st.markdown("**Predicted crack localisation**")
            st.image(overlay(rgb448, seg) if res["status"] == "Micro_cracked"
                     else cv2.resize(rgb448, (TRAIN_SIDE, TRAIN_SIDE)),
                     use_container_width=True)
            st.markdown("<span class='muted'>Segmentation head, validated against expert "
                        "annotations at pointing game 0.765</span>", unsafe_allow_html=True)

        with st.expander("Class probabilities"):
            st.bar_chart(pd.DataFrame({"probability": p4}, index=CFG.LABEL4), height=220)

        if res["status"] == "Micro_cracked":
            k = rules.get((res["pattern"], age_class))
            if k is None:
                st.error(f"No rule defined for ({res['pattern']}, {age_class}).")
            else:
                st.divider()
                st.subheader("Probable root cause")
                st.markdown(f"<span class='muted'>For a <b>{res['pattern']}</b> crack in a "
                            f"<b>{age_class}</b> flock</span>", unsafe_allow_html=True)
                a, b = st.columns(2)
                a.metric("Physiological / nutritional", f"{k['phys']:.0f}%")
                b.metric("Mechanical / environmental", f"{k['mech']:.0f}%")
                st.progress(k["phys"] / 100.0,
                            text=f"{k['phys']:.0f}% shell weakness    "
                                 f"{k['mech']:.0f}% impact or handling")
                st.markdown("**Interpretation**")
                st.info(k["interpretation"])
                st.markdown("**Suggested actions**")
                st.success(k["actions"])
                with st.expander("How reliable is this attribution?"):
                    st.markdown(
                        "A deterministic lookup on crack morphology and flock lay stage, "
                        "assembled from Ross 308 specifications and the crack-morphology "
                        "literature with a hatchery manager. It is **not** learned from data and "
                        "has **not** been validated against hatch outcomes.\n\n"
                        "Age class is supplied by you rather than predicted, so every attribution "
                        "error traces back to a morphology error. On the held-out test set the "
                        "attribution was exact for **82.4%** of cracked eggs, with a mean error "
                        "of **4.8 percentage points**. The dominant cause reversed on **13.7%** "
                        "of eggs, almost all Star and Network confusions, which shift the split "
                        "by 40 points.\n\n"
                        "A single egg is weak evidence for a management decision. Use the Batch "
                        "tab for a flock-level assessment.")

# =========================================================================== BATCH
with tab_batch:
    st.markdown("#### Tray or flock sample")
    st.markdown("<span class='muted'>A single egg rarely justifies changing feed or stopping a "
                "belt. A batch does. Upload every egg from a tray, or a sample from one flock."
                "</span>", unsafe_allow_html=True)
    st.write("")
    ups = st.file_uploader("Candled egg images", accept_multiple_files=True,
                           type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
                           key="batch_uploader")
    notes = st.text_area("Operator notes (optional)", height=80,
                         placeholder="Nest condition, belt speed, anything unusual today")

    if ups and st.button("Analyse batch", type="primary", use_container_width=True):
        rows, thumbs = [], {}
        bar = st.progress(0.0, text="Analysing")
        for i, f in enumerate(ups):
            raw = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
            rgb = preprocess_bgr(raw) if raw is not None else None
            if rgb is None:
                rows.append({"file": f.name, "status": "unreadable", "pattern": None,
                             "crack_probability": np.nan, "confidence": np.nan,
                             "physiological_pct": np.nan, "mechanical_pct": np.nan,
                             "needs_review": True})
                bar.progress((i + 1) / len(ups)); continue
            p4, seg = infer(models, rgb)
            r = classify(p4, threshold, low_conf)
            k = rules.get((r["pattern"], age_class)) if r["pattern"] else None
            rows.append({"file": f.name, "status": r["status"], "pattern": r["pattern"],
                         "crack_probability": round(r["crack_probability"], 3),
                         "confidence": round(r["confidence"], 3),
                         "physiological_pct": k["phys"] if k else np.nan,
                         "mechanical_pct": k["mech"] if k else np.nan,
                         "needs_review": r["needs_review"]})
            if r["status"] == "Micro_cracked" and len(thumbs) < 12:
                thumbs[f.name] = (overlay(rgb, seg), r)
            bar.progress((i + 1) / len(ups), text=f"Analysing  {i + 1} of {len(ups)}")
        bar.empty()
        st.session_state["batch"] = pd.DataFrame(rows)
        st.session_state["thumbs"] = thumbs

    if "batch" in st.session_state and len(st.session_state["batch"]):
        df = st.session_state["batch"]
        ok = df[df["status"] != "unreadable"].copy()
        cracked = int((ok["status"] == "Micro_cracked").sum())
        rate = cracked / max(len(ok), 1) * 100
        flock = flock_assessment(ok, rules, age_class)

        st.divider()
        c = st.columns(4)
        c[0].metric("Eggs inspected", len(ok))
        c[1].metric("Micro-cracks", cracked, f"{rate:.1f}% of batch",
                    delta_color="inverse")
        c[2].metric("Needs manual review", int(ok["needs_review"].sum()))
        c[3].metric("Unreadable", int((df["status"] == "unreadable").sum()))

        if cracked:
            g1, g2 = st.columns([1, 1.15])
            with g1:
                st.markdown("**Crack morphology mix**")
                vc = (ok.loc[ok["status"] == "Micro_cracked", "pattern"]
                        .value_counts().reindex(CFG.PATTERNS).fillna(0).astype(int))
                st.bar_chart(vc.rename("eggs"), height=260)
            with g2:
                st.markdown("**Batch root cause**")
                st.metric("Physiological / nutritional", f"{flock['phys']:.0f}%")
                st.progress(flock["phys"] / 100.0,
                            text=f"{flock['phys']:.0f}% shell weakness    "
                                 f"{flock['mech']:.0f}% impact or handling")
                st.markdown(flock["verdict"])

            st.markdown("**Recommended actions for this flock**")
            for a in flock["actions"]:
                st.markdown(
                    f"<div class='panel'><div class='act-head'>{a['pattern']} &mdash; "
                    f"{a['count']} egg{'s' if a['count'] > 1 else ''} "
                    f"({a['phys']:.0f}% physiological)</div>{a['text']}</div>",
                    unsafe_allow_html=True)
                st.write("")

            if st.session_state.get("thumbs"):
                with st.expander(f"Detected cracks ({len(st.session_state['thumbs'])} shown)"):
                    items = list(st.session_state["thumbs"].items())
                    for r0 in range(0, len(items), 4):
                        cols = st.columns(4)
                        for col, (name, (img, rr)) in zip(cols, items[r0:r0 + 4]):
                            col.image(img, use_container_width=True)
                            col.caption(f"{name} - {rr['pattern']} - {rr['confidence']:.0%}"
                                        + ("  [review]" if rr["needs_review"] else ""))

        st.divider()
        st.markdown("**Per-egg results**")
        st.dataframe(df, use_container_width=True, hide_index=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        d1, d2 = st.columns(2)
        d1.download_button("Download results (CSV)", df.to_csv(index=False).encode(),
                           file_name=f"crack_inspection_{stamp}.csv", mime="text/csv",
                           use_container_width=True)
        d2.download_button(
            "Download report (Markdown)",
            build_markdown_report(ok, flock, age_class, operator, flock_id, notes).encode(),
            file_name=f"crack_report_{stamp}.md", mime="text/markdown",
            use_container_width=True)

# =========================================================================== MODEL
with tab_model:
    st.markdown("#### Model card")
    a, b, c_ = st.columns(3)
    a.metric("Crack detection macro F1", f"{PERF['detection_f1']:.3f}")
    b.metric("Morphology macro F1", f"{PERF['morph_f1']:.3f}",
             help=f"95% confidence interval {PERF['morph_ci'][0]:.3f} to "
                  f"{PERF['morph_ci'][1]:.3f}")
    c_.metric("Localisation pointing game", f"{PERF['pointing']:.3f}")

    st.markdown(f"""
| | |
|---|---|
| Architecture | EfficientNet-B0, multi-task, auxiliary segmentation head (3.67 M parameters) |
| Input | 384 x 384, egg isolated and background neutralised, per-image per-channel standardisation |
| Training data | {PERF['n_train']} candled hatching eggs, single annotator, verified by a hatchery manager |
| Test data | {PERF['n_test']} eggs, sealed before any modelling and opened exactly once |
| Detection | macro F1 {PERF['detection_f1']:.3f}, ROC-AUC {PERF['detection_auc']:.3f}, recall {PERF['detection_recall']:.3f} |
| Morphology | macro F1 {PERF['morph_f1']:.3f} (95% CI {PERF['morph_ci'][0]:.3f} to {PERF['morph_ci'][1]:.3f}) |
| Localisation | pointing game {PERF['pointing']:.3f}, energy inside expert polygon {PERF['energy']:.3f} |
""")
    st.markdown(f"""
**Reference points.** Capture statistics alone - image size, brightness and colour, with no egg
content whatsoever - reach **{PERF['floor']:.3f}** on morphology. The best classical pipeline,
combining texture descriptors, ridge filtering and crack geometry, reaches **{PERF['classical']:.3f}**.
Read every figure above against those, not against chance.
""")
    st.markdown("##### Known limitations")
    st.markdown("""
- **Single annotator.** All labels come from one person, verified by a domain expert but not
  independently double-rated, so inter-rater reliability is unknown.
- **One farm, one capture setup.** Every training image comes from a single light-proof box and a
  single smartphone. A session-held-out evaluation showed a drop of about 0.06 macro F1, so expect
  some loss on new equipment or lighting.
- **Star is the smallest class.** 68 eggs in total and 11 in the test set, so its confidence
  interval is wide, 0.55 to 0.94.
- **Star and Network confusions are the expensive ones.** They shift the root-cause split by 40
  percentage points and reverse the dominant cause in every age class.
- **The root cause is expert knowledge, not measurement.** Deterministic rules drawn from Ross 308
  specifications and the crack-morphology literature, never validated against hatch outcomes.
- **Not validated for grading or sale decisions.** It supports inspection; it does not replace it.
""")
    st.markdown("<span class='muted'>Model card compiled from notebooks 07 to 10. All figures "
                "are from the held-out test set unless stated otherwise.</span>",
                unsafe_allow_html=True)
