"""
Eggshell Micro-Crack Diagnostics — Flask application factory.

The factory wires three long-lived objects onto the app: the model ensemble, the rule base and the
batch job store. All three are created once at start-up rather than per request, because loading
the ensemble takes seconds and allocates hundreds of megabytes.

Models are loaded eagerly so that a container with missing or corrupt checkpoints fails at
start-up, where an orchestrator will notice, rather than on the first user request. The readiness
probe reflects that state; the liveness probe does not, so a model problem is not mistaken for a
hung process.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from flask import Flask


def _bootstrap_path() -> None:
    """Make Artefact/src importable whether running from source or from a container."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        src = parent / "src"
        if (src / "common").is_dir():
            sys.path.insert(0, str(src))
            return
    env = os.environ.get("EGG_ARTEFACT_ROOT")
    if env and (Path(env) / "src" / "common").is_dir():
        sys.path.insert(0, str(Path(env) / "src"))


def create_app(config_object=None) -> Flask:
    _bootstrap_path()

    from config import get_config
    from .inference import Predictor
    from .jobs import JobStore
    from .rules import RuleBase

    app = Flask(__name__, instance_relative_config=False)
    app.config.from_object(config_object or get_config()())

    logging.basicConfig(
        level=logging.DEBUG if app.config.get("DEBUG") else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    log = logging.getLogger("eggshell")

    if abs(app.config["DETECTION_THRESHOLD"] - app.config["DEFAULT_THRESHOLD"]) > 1e-9:
        log.warning(
            "detection threshold is %.2f, not the evidenced default %.2f. The default was the "
            "highest threshold at which no cracked egg was missed on the held-out test set; "
            "raising it will start missing cracks.",
            app.config["DETECTION_THRESHOLD"], app.config["DEFAULT_THRESHOLD"])

    app.predictor = Predictor(app.config)
    app.rulebase = RuleBase(app.config)
    app.jobs = JobStore(app.config["JOB_TTL_SECONDS"], app.config["MAX_JOBS"])

    app.rulebase.load()
    if os.environ.get("EGG_LAZY_LOAD", "0") not in ("1", "true", "yes"):
        app.predictor.load()

    from .routes import bp as web_bp
    from .api import bp as api_bp
    app.register_blueprint(web_bp)
    app.register_blueprint(api_bp, url_prefix="/api/v1")

    from .errors import register_error_handlers
    register_error_handlers(app)

    @app.context_processor
    def inject_globals():
        return {
            "PERF": app.config["PERFORMANCE"],
            "cfg": app.config,
            "models_ready": app.predictor.ready,
            "rules_ready": app.rulebase.ready,
            "n_models": app.predictor.n_models,
        }

    log.info("application ready: models=%s rules=%s",
             app.predictor.ready, app.rulebase.ready)
    return app
