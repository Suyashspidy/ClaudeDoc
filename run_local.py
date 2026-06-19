"""Local single-process launcher for ClaudeDoc (no Redis / no Docker required).

Runs the FastAPI backend with Celery in EAGER mode, so document jobs execute
synchronously inside the API process. This is for local testing only — the
production stack uses Redis + a separate Celery worker.

Usage:
    .venv/Scripts/python.exe run_local.py
"""
from __future__ import annotations
import os
import sys

# Active detectors are CPU-only; no GPU needed for the live pipeline.
os.environ.setdefault("USE_GPU", "false")

# Make `app.*` and `config.*` importable the same way the server expects.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
sys.path.insert(0, os.path.dirname(__file__))

from app.worker import celery_app  # noqa: E402

# Eager mode: run tasks inline, store the result in an in-process backend so the
# /status/{job_id} endpoint can read SUCCESS without Redis.
celery_app.conf.task_always_eager = True
celery_app.conf.task_eager_propagates = True
celery_app.conf.task_store_eager_result = True
celery_app.conf.broker_url = "memory://"
celery_app.conf.result_backend = "cache+memory://"

if __name__ == "__main__":
    import uvicorn
    from app.main import app  # noqa: E402

    print("ClaudeDoc local server -> http://127.0.0.1:8000  (eager mode, no Redis)")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
