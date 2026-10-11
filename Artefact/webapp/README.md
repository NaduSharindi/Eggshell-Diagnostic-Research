# Eggshell Micro-Crack Diagnostics — web application

Flask implementation of the decision-support artefact. Same model, same preprocessing and same
decision rules as the Streamlit prototype, packaged for deployment.

```
webapp/
  wsgi.py              entry point
  config.py            settings, each default traceable to the notebook that produced it
  gunicorn.conf.py     production server configuration
  Dockerfile           two-stage, CPU-only torch
  requirements.txt
  app/
    __init__.py        application factory
    inference.py       model loading and prediction
    rules.py           root-cause rule base and flock aggregation
    reporting.py       CSV and Markdown report generation
    routes.py          browser views
    api.py             JSON API under /api/v1
    jobs.py            bounded in-memory batch store
    errors.py          HTML or JSON error handling
    utils.py
    templates/  static/
  deploy/README_DEPLOY.md
```

## Run locally

```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python wsgi.py          # http://127.0.0.1:8000
```

Requires the trained checkpoints in `Artefact/models/final/` and the rule base at
`Artefact/data/processed/root_cause_rules_recovered.csv`. Set `EGG_ARTEFACT_ROOT` if the
application cannot find them.

## What it does

**Inspect** — one egg: verdict, crack probability, morphology, confidence, crack localisation from
the segmentation head, class probabilities, root-cause split, interpretation and actions.

**Batch** — a tray: crack rate, morphology mix, aggregate flock-level root cause, full action text
grouped by pattern, localisation thumbnails, per-egg table, CSV and printable report.

**Model** — the model card: performance with its reference points, the rule base, and the
limitations.

**API** — `/api/v1`, for integration with other equipment such as a future conveyor-mounted camera.

## Decisions built in

| Setting | Value | Evidence |
|---|---|---|
| Detection threshold | 0.30 | notebook 08: highest threshold at which no cracked egg was missed on the held-out test set |
| Review flag | below 0.60 confidence | notebook 08: accuracy in the 0.45–0.60 band falls under 0.5 |
| Localisation source | segmentation head, not Grad-CAM | notebook 09: pointing game 0.765 against 0.549 |

Changing the threshold is possible but logged as a warning, because it changes the error profile
the farm experiences.

## Design notes

**Preprocessing is imported, never reimplemented.** `common.preprocess` built the training cache
and is what the application calls. A divergence between training and serving preprocessing
produces confident wrong answers rather than visible errors.

**No uploaded image touches disk.** Everything is decoded in memory and returned as a data URI, so
the container needs no writable volume and leaves no pixel behind.

**One process, several threads.** Each process would hold its own copy of the ensemble, and the
batch store is per process. See `deploy/README_DEPLOY.md`.
