"""
HuggingFace Spaces entry point.
Runs the full stack (API + worker + Streamlit) in a single process for HF Spaces.
"""
from __future__ import annotations
import os
import subprocess
import sys
import time
import threading

os.environ.setdefault("USE_GPU", "false")  # HF Spaces CPU tier by default


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def start_redis() -> subprocess.Popen:
    return subprocess.Popen(["redis-server", "--daemonize", "no"])


def start_api() -> subprocess.Popen:
    return subprocess.Popen([
        sys.executable, "-m", "uvicorn",
        "backend.app.main:app",
        "--host", "0.0.0.0", "--port", "8000",
    ])


def start_worker() -> subprocess.Popen:
    return subprocess.Popen([
        sys.executable, "-m", "celery",
        "-A", "backend.app.worker.celery_app", "worker",
        "--loglevel=warning", "--concurrency=1",
    ])


if __name__ == "__main__":
    procs = []
    procs.append(start_redis())
    time.sleep(2)
    procs.append(start_api())
    procs.append(start_worker())
    time.sleep(3)

    # Streamlit is the main process
    os.execv(sys.executable, [
        sys.executable, "-m", "streamlit", "run",
        "frontend/app.py",
        "--server.port=7860",
        "--server.address=0.0.0.0",
    ])
