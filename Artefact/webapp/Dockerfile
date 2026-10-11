# syntax=docker/dockerfile:1
#
# Eggshell Micro-Crack Diagnostics - production image
#
# BUILD FROM THE REPOSITORY ROOT, not from this directory:
#     docker build -f Artefact/webapp/Dockerfile -t eggshell-diagnostics .
#
# The image needs Artefact/src (shared preprocessing and model code), the three final
# checkpoints and the rule base, all of which sit outside webapp/. The matching
# .dockerignore lives at the repository root for the same reason.
#
# Two stages, so the build toolchain never reaches the runtime image, and CPU-only torch,
# because an inference deployment has no GPU and the CUDA wheels add about 2 GB.

FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*
COPY Artefact/webapp/requirements.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt


FROM python:3.11-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    FLASK_ENV=production \
    TORCH_NUM_THREADS=2 \
    EGG_ARTEFACT_ROOT=/srv/artefact

# opencv needs libgl and libglib even in its headless build; curl is for the health check
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /srv/artefact
COPY --chown=appuser:appuser Artefact/src     /srv/artefact/src
COPY --chown=appuser:appuser Artefact/webapp  /srv/artefact/webapp
# only the deployed model, not the runner-up or the experiment checkpoints
COPY --chown=appuser:appuser \
     Artefact/models/final/final_efficientnet_b0_multitask_seg_seed*.pt \
     /srv/artefact/models/final/
COPY --chown=appuser:appuser \
     Artefact/data/processed/root_cause_rules_recovered.csv \
     /srv/artefact/data/processed/

USER appuser
WORKDIR /srv/artefact/webapp
EXPOSE 8000

# liveness only. Readiness (models loaded) is /api/v1/ready and is checked by the deploy job.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/api/v1/health || exit 1

CMD ["gunicorn", "-c", "gunicorn.conf.py", "wsgi:application"]
