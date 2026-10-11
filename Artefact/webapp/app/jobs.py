"""
jobs.py
In-memory store for batch results.

Batch output is far too large for a session cookie and the application is deliberately stateless
on disk, so results live in a bounded, time-limited in-process dictionary keyed by a job id. The
browser holds only that id.

The bound matters in production. Without it a few large trays would grow the worker's resident set
until it was killed, and on a single-container deployment that takes the whole service down. Jobs
expire after a TTL and the oldest are evicted once the cap is reached.

Note for a multi-worker deployment: this store is per worker. Gunicorn is therefore configured
with threads rather than multiple worker processes, so a job created on one request is visible to
the next. If the service is ever scaled to multiple processes or instances, this must move to
Redis or ElastiCache; `deploy/README_DEPLOY.md` says so explicitly.
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Optional


class JobStore:
    def __init__(self, ttl_seconds: int = 3600, max_jobs: int = 32):
        self._ttl = ttl_seconds
        self._max = max_jobs
        self._data: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _purge(self) -> None:
        now = time.time()
        dead = [k for k, v in self._data.items() if now - v["created"] > self._ttl]
        for k in dead:
            self._data.pop(k, None)
        while len(self._data) > self._max:
            oldest = min(self._data, key=lambda k: self._data[k]["created"])
            self._data.pop(oldest, None)

    def put(self, payload: dict) -> str:
        job_id = uuid.uuid4().hex[:16]
        with self._lock:
            self._data[job_id] = {"created": time.time(), "payload": payload}
            self._purge()
        return job_id

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            self._purge()
            rec = self._data.get(job_id)
            return rec["payload"] if rec else None

    def drop(self, job_id: str) -> None:
        with self._lock:
            self._data.pop(job_id, None)

    def stats(self) -> dict:
        with self._lock:
            return {"jobs": len(self._data), "ttl_seconds": self._ttl, "max_jobs": self._max}
