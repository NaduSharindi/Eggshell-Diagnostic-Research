"""
api.py
JSON API, so the model can be driven by something other than a browser.

This is what a future conveyor-mounted camera would call. It is versioned under /api/v1 from the
start, because a deployed endpoint that other equipment depends on cannot be changed quietly.

Endpoints
    GET  /api/v1/health     liveness: is the process up
    GET  /api/v1/ready      readiness: are models and rules loaded
    GET  /api/v1/info       model card as JSON
    GET  /api/v1/rules      the root-cause rule base
    POST /api/v1/inspect    one image, multipart "image", returns a full diagnosis
    POST /api/v1/batch      many images, multipart "images", returns records plus flock assessment

Health and readiness are separate on purpose. A load balancer that confuses them will cycle
containers forever when the only problem is a missing checkpoint.
"""
from __future__ import annotations

import logging

from flask import Blueprint, current_app, jsonify, request

from .inference import ModelsUnavailable, decode_upload, overlay_png, to_data_uri
from .utils import allowed_file, read_threshold, summarise, valid_age_class

bp = Blueprint("api", __name__)
log = logging.getLogger(__name__)


@bp.get("/health")
def health():
    return jsonify(status="ok", service="eggshell-diagnostics"), 200


@bp.get("/ready")
def ready():
    p, r = current_app.predictor, current_app.rulebase
    ok = p.ready and r.ready
    body = {"ready": ok, "models_loaded": p.n_models,
            "model_error": p.error, "rules_error": r.error,
            "jobs": current_app.jobs.stats()}
    return jsonify(body), (200 if ok else 503)


@bp.get("/info")
def info():
    cfg = current_app.config
    return jsonify({
        "model": {"backbone": cfg["BACKBONE"], "formulation": cfg["FORMULATION"],
                  "input_size": cfg["TRAIN_SIDE"], "ensemble": current_app.predictor.n_models,
                  "parameters_millions": cfg["PERFORMANCE"]["params_m"]},
        "decision": {"detection_threshold": cfg["DETECTION_THRESHOLD"],
                     "low_confidence_flag": cfg["LOW_CONFIDENCE"],
                     "threshold_rationale": (
                         "Highest threshold at which no cracked egg was missed on the held-out "
                         "test set. The system over-flags rather than under-flags.")},
        "performance": cfg["PERFORMANCE"],
        "limitations": [
            "Single annotator; inter-rater reliability unknown.",
            "One farm and one capture setup; a session-held-out evaluation lost about 0.06 "
            "macro F1.",
            "Star is the smallest class, 11 eggs in the test set, 95% CI 0.55 to 0.94.",
            "Star and Network confusions shift the root-cause split by 40 percentage points.",
            "The root-cause rules are expert knowledge, never validated against hatch outcomes.",
            "Not validated for grading or sale decisions.",
        ],
    })


@bp.get("/rules")
def rules():
    rb = current_app.rulebase
    if not rb.ready:
        return jsonify(error="rules_unavailable", message=rb.error), 503
    out = []
    for p in rb.patterns:
        for a in rb.age_classes:
            r = rb.get(p, a)
            if r:
                out.append({"crack_pattern": p, "age_class": a, **r})
    return jsonify(rules=out, note=(
        "Deterministic expert rules from Ross 308 specifications and the crack-morphology "
        "literature. Not learned from data and not validated against hatch outcomes."))


def _diagnose(file_storage, settings, want_overlay=True):
    predictor, rb = current_app.predictor, current_app.rulebase
    name = file_storage.filename
    if not allowed_file(name):
        return {"file": name, "status": "unreadable", "error": "unsupported file type"}, None
    raw = decode_upload(file_storage.read())
    rgb = predictor.preprocess(raw) if raw is not None else None
    if rgb is None:
        return {"file": name, "status": "unreadable",
                "error": "image unreadable or no egg located"}, None
    pred, seg = predictor.predict(rgb, settings["threshold"], settings["low_confidence"])
    rule = rb.get(pred.pattern, settings["age_class"]) if pred.pattern else None
    rec = {"file": name, **pred.as_dict(),
           "physiological_pct": rule["phys"] if rule else None,
           "mechanical_pct": rule["mech"] if rule else None,
           "interpretation": rule["interpretation"] if rule else None,
           "suggested_actions": rule["actions"] if rule else None}
    return rec, (overlay_png(rgb, seg) if (want_overlay and pred.status == "Micro_cracked")
                 else None)


@bp.post("/inspect")
def api_inspect():
    f = request.files.get("image")
    if not f or not f.filename:
        return jsonify(error="bad_request", message="multipart field 'image' is required"), 400
    t, lc = read_threshold(request.form)
    settings = {"threshold": t, "low_confidence": lc,
                "age_class": valid_age_class(request.form.get("age_class"))}
    include = str(request.form.get("include_overlay", "false")).lower() in ("1", "true", "yes")
    try:
        rec, overlay = _diagnose(f, settings, want_overlay=include)
    except ModelsUnavailable as e:
        return jsonify(error="models_unavailable", message=str(e)), 503
    if rec.get("status") == "unreadable":
        return jsonify(error="unreadable_image", message=rec.get("error"), file=rec["file"]), 400
    body = {"result": rec, "settings": settings}
    if include and overlay:
        body["overlay"] = overlay
    return jsonify(body)


@bp.post("/batch")
def api_batch():
    files = [f for f in request.files.getlist("images") if f and f.filename]
    if not files:
        return jsonify(error="bad_request", message="multipart field 'images' is required"), 400
    limit = current_app.config["MAX_BATCH_FILES"]
    if len(files) > limit:
        return jsonify(error="too_many_files", message=f"limit is {limit} per request"), 400

    t, lc = read_threshold(request.form)
    settings = {"threshold": t, "low_confidence": lc,
                "age_class": valid_age_class(request.form.get("age_class"))}
    records = []
    try:
        for f in files:
            rec, _ = _diagnose(f, settings, want_overlay=False)
            records.append(rec)
    except ModelsUnavailable as e:
        return jsonify(error="models_unavailable", message=str(e)), 503

    summary = summarise(records)
    flock = current_app.rulebase.flock_assessment(records, settings["age_class"])
    return jsonify(summary=summary, flock_assessment=flock, records=records,
                   settings=settings)
