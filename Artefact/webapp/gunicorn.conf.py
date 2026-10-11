"""
gunicorn.conf.py

Threads, not multiple worker processes. Two reasons, both specific to this application:

1. Each worker loads its own copy of the three-seed ensemble. Four processes would hold four
   copies of the weights, which on a small instance is the difference between fitting in memory
   and being OOM-killed.
2. Batch results live in an in-process store keyed by job id. With several processes a redirect
   after analysis could land on a worker that has never seen that job, and the user would be told
   their batch had expired seconds after creating it.

If this is ever scaled beyond one process, move the job store to Redis first. `app/jobs.py` says
so too.

The timeout is generous because a 240-image tray on CPU takes minutes, and killing a worker
mid-batch loses all of it.
"""
import multiprocessing  # noqa: F401  (kept for operators who want to re-derive the counts)
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"
workers = int(os.environ.get("GUNICORN_WORKERS", "1"))
threads = int(os.environ.get("GUNICORN_THREADS", "4"))
worker_class = "gthread"

timeout = int(os.environ.get("GUNICORN_TIMEOUT", "600"))
graceful_timeout = 60
keepalive = 5

# a worker holding the model ensemble is expensive to recreate, so recycle rarely
max_requests = int(os.environ.get("GUNICORN_MAX_REQUESTS", "2000"))
max_requests_jitter = 200

accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info")
access_log_format = '%(h)s "%(r)s" %(s)s %(b)s %(M)sms'

# load before forking so the weights are read once; harmless at workers=1 and correct if raised
preload_app = True


def on_starting(server):
    server.log.info("Eggshell diagnostics starting: %s workers, %s threads", workers, threads)
