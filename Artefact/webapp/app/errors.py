"""
errors.py
Error handling.

Browser requests get a rendered page; anything under /api/v1 gets JSON. Internal messages are
never leaked to the user: the 500 handler logs the traceback and returns a generic message, which
matters because uploaded filenames and absolute server paths would otherwise appear in the
response.
"""
from __future__ import annotations

import logging

from flask import jsonify, render_template, request

log = logging.getLogger(__name__)


def _wants_json() -> bool:
    return request.path.startswith("/api/") or \
        request.accept_mimetypes.best == "application/json"


def register_error_handlers(app):
    @app.errorhandler(400)
    def bad_request(e):
        msg = getattr(e, "description", "Bad request.")
        if _wants_json():
            return jsonify(error="bad_request", message=msg), 400
        return render_template("error.html", code=400, title="Bad request", message=msg), 400

    @app.errorhandler(404)
    def not_found(e):
        msg = "That page does not exist."
        if _wants_json():
            return jsonify(error="not_found", message=msg), 404
        return render_template("error.html", code=404, title="Not found", message=msg), 404

    @app.errorhandler(413)
    def too_large(e):
        limit = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
        msg = f"Upload too large. The limit is {limit} MB per request."
        if _wants_json():
            return jsonify(error="payload_too_large", message=msg), 413
        return render_template("error.html", code=413, title="Upload too large",
                               message=msg), 413

    @app.errorhandler(500)
    def server_error(e):
        log.exception("unhandled error on %s", request.path)
        msg = "Something went wrong on the server. The error has been logged."
        if _wants_json():
            return jsonify(error="internal_error", message=msg), 500
        return render_template("error.html", code=500, title="Server error", message=msg), 500

    @app.errorhandler(503)
    def unavailable(e):
        msg = getattr(e, "description", "Service temporarily unavailable.")
        if _wants_json():
            return jsonify(error="unavailable", message=msg), 503
        return render_template("error.html", code=503, title="Unavailable", message=msg), 503
