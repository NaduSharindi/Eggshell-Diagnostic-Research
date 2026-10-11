"""
routes.py
Browser-facing views: inspect one egg, analyse a batch, read the model card.

Uploaded images are decoded in memory and never written to disk. Results are returned as data
URIs, so a deployment leaves no uploaded pixel behind and needs no writable volume — which also
makes the container safe to run with a read-only filesystem.
"""
from __future__ import annotations

import logging

from flask import (Blueprint, Response, abort, current_app, flash, redirect,
                   render_template, request, url_for)

from .inference import ModelsUnavailable, decode_upload, overlay_png, to_data_uri
from .reporting import build_report, records_to_csv
from .utils import allowed_file, read_threshold, summarise, valid_age_class

bp = Blueprint("web", __name__)
log = logging.getLogger(__name__)


def _settings():
    t, lc = read_threshold(request.form)
    return {"threshold": t, "low_confidence": lc,
            "age_class": valid_age_class(request.form.get("age_class")),
            "flock_id": (request.form.get("flock_id") or "").strip()[:120],
            "operator": (request.form.get("operator") or "").strip()[:120],
            "notes": (request.form.get("notes") or "").strip()[:2000]}


@bp.route("/")
def index():
    return redirect(url_for("web.inspect"))


@bp.route("/inspect", methods=["GET", "POST"])
def inspect():
    cfg = current_app.config
    ctx = {"active": "inspect",
           "settings": {"threshold": cfg["DETECTION_THRESHOLD"],
                        "low_confidence": cfg["LOW_CONFIDENCE"],
                        "age_class": valid_age_class(None),
                        "flock_id": "", "operator": "", "notes": ""},
           "age_classes": current_app.rulebase.age_classes,
           "result": None}

    if request.method == "GET":
        return render_template("inspect.html", **ctx)

    s = _settings()
    ctx["settings"] = s
    f = request.files.get("image")
    if not f or not f.filename:
        flash("Choose a candled egg image first.", "warn")
        return render_template("inspect.html", **ctx), 400
    if not allowed_file(f.filename):
        flash("That file type is not supported. Use JPG, PNG, BMP, TIF or WebP.", "warn")
        return render_template("inspect.html", **ctx), 400

    raw = decode_upload(f.read())
    if raw is None:
        flash("That image could not be read. It may be corrupt or an unsupported format.", "warn")
        return render_template("inspect.html", **ctx), 400

    predictor = current_app.predictor
    rgb = predictor.preprocess(raw)
    if rgb is None:
        flash("No egg could be located in that image. It should show a candled egg "
              "against a dark field.", "warn")
        return render_template("inspect.html", **ctx), 400

    try:
        pred, seg = predictor.predict(rgb, s["threshold"], s["low_confidence"])
    except ModelsUnavailable as e:
        abort(503, description=str(e))

    rule = None
    if pred.status == "Micro_cracked":
        rule = current_app.rulebase.get(pred.pattern, s["age_class"])

    ctx["result"] = {
        "filename": f.filename,
        "prediction": pred,
        "image": to_data_uri(rgb),
        "overlay": overlay_png(rgb, seg) if pred.status == "Micro_cracked" else to_data_uri(rgb),
        "rule": rule,
        "probabilities": sorted(pred.probabilities.items(), key=lambda kv: -kv[1]),
    }
    return render_template("inspect.html", **ctx)


@bp.route("/batch", methods=["GET", "POST"])
def batch():
    cfg = current_app.config
    ctx = {"active": "batch",
           "settings": {"threshold": cfg["DETECTION_THRESHOLD"],
                        "low_confidence": cfg["LOW_CONFIDENCE"],
                        "age_class": valid_age_class(None),
                        "flock_id": "", "operator": "", "notes": ""},
           "age_classes": current_app.rulebase.age_classes,
           "job": None}

    if request.method == "GET":
        job_id = request.args.get("job")
        if job_id:
            payload = current_app.jobs.get(job_id)
            if payload:
                ctx["job"] = payload
                ctx["settings"] = payload["settings"]
            else:
                flash("That batch has expired. Analyse the images again.", "warn")
        return render_template("batch.html", **ctx)

    s = _settings()
    ctx["settings"] = s
    files = [f for f in request.files.getlist("images") if f and f.filename]
    if not files:
        flash("Choose at least one image.", "warn")
        return render_template("batch.html", **ctx), 400
    if len(files) > cfg["MAX_BATCH_FILES"]:
        flash(f"Too many files. The limit is {cfg['MAX_BATCH_FILES']} per batch.", "warn")
        return render_template("batch.html", **ctx), 400

    predictor = current_app.predictor
    rules = current_app.rulebase
    records, thumbs = [], []

    for f in files:
        name = f.filename
        if not allowed_file(name):
            records.append({"file": name, "status": "unreadable", "pattern": None,
                            "crack_probability": None, "confidence": None,
                            "physiological_pct": None, "mechanical_pct": None,
                            "needs_review": True})
            continue
        raw = decode_upload(f.read())
        rgb = predictor.preprocess(raw) if raw is not None else None
        if rgb is None:
            records.append({"file": name, "status": "unreadable", "pattern": None,
                            "crack_probability": None, "confidence": None,
                            "physiological_pct": None, "mechanical_pct": None,
                            "needs_review": True})
            continue
        try:
            pred, seg = predictor.predict(rgb, s["threshold"], s["low_confidence"])
        except ModelsUnavailable as e:
            abort(503, description=str(e))
        rule = rules.get(pred.pattern, s["age_class"]) if pred.pattern else None
        records.append({
            "file": name, "status": pred.status, "pattern": pred.pattern,
            "crack_probability": round(pred.crack_probability, 3),
            "confidence": round(pred.confidence, 3),
            "physiological_pct": rule["phys"] if rule else None,
            "mechanical_pct": rule["mech"] if rule else None,
            "needs_review": pred.needs_review,
        })
        if pred.status == "Micro_cracked" and len(thumbs) < cfg["MAX_THUMBNAILS"]:
            thumbs.append({"file": name, "pattern": pred.pattern,
                           "confidence": pred.confidence,
                           "needs_review": pred.needs_review,
                           "image": overlay_png(rgb, seg)})

    summary = summarise(records)
    flock = rules.flock_assessment(records, s["age_class"])
    payload = {"records": records, "summary": summary, "flock": flock,
               "thumbs": thumbs, "settings": s,
               "max_count": max(flock.get("counts", {}).values(), default=1)}
    job_id = current_app.jobs.put(payload)
    log.info("batch %s: %d files, %d cracked", job_id, summary["n_readable"],
             summary["n_cracked"])
    return redirect(url_for("web.batch", job="" + job_id))


@bp.route("/batch/<job_id>/download.csv")
def batch_csv(job_id):
    payload = current_app.jobs.get(job_id)
    if not payload:
        abort(404)
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return Response(
        records_to_csv(payload["records"]), mimetype="text/csv",
        headers={"Content-Disposition":
                 f'attachment; filename="crack_inspection_{stamp}.csv"'})


@bp.route("/batch/<job_id>/report.md")
def batch_report(job_id):
    payload = current_app.jobs.get(job_id)
    if not payload:
        abort(404)
    s = payload["settings"]
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    md = build_report(payload["records"], payload["summary"], payload["flock"],
                      s["age_class"], s["operator"], s["flock_id"], s["notes"])
    return Response(md, mimetype="text/markdown",
                    headers={"Content-Disposition":
                             f'attachment; filename="crack_report_{stamp}.md"'})


@bp.route("/model")
def model_card():
    return render_template("model.html", active="model",
                           matrix=current_app.rulebase.matrix(),
                           age_classes=current_app.rulebase.age_classes)
