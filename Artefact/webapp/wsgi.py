"""
wsgi.py
WSGI entry point.

    development:  python wsgi.py
    production:   gunicorn -c gunicorn.conf.py wsgi:application
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import create_app  # noqa: E402

application = create_app()
app = application  # convenience alias

if __name__ == "__main__":
    application.run(host=os.environ.get("HOST", "127.0.0.1"),
                    port=int(os.environ.get("PORT", "8000")),
                    debug=os.environ.get("FLASK_ENV", "").startswith("dev"))
