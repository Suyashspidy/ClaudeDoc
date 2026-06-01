"""Celery worker — processes document QA jobs asynchronously."""
from __future__ import annotations
import json
import logging
from pathlib import Path

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

from celery import Celery
from config.settings import settings

logger = logging.getLogger(__name__)

celery_app = Celery(
    "claudedoc",
    broker=settings.celery_broker,
    backend=settings.celery_backend,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)


@celery_app.task(bind=True, name="claudedoc.process_document")
def process_document(self, job_id: str, file_path: str) -> dict:
    """Main Celery task that runs the QA pipeline on an uploaded document."""
    from app.services.pipeline import DocumentQAPipeline

    output_dir = settings.report_dir / job_id
    pipeline = DocumentQAPipeline(job_id=job_id, output_dir=output_dir)

    def _update_progress(pct: float, msg: str) -> None:
        self.update_state(
            state="PROGRESS",
            meta={"progress": pct, "message": msg},
        )

    try:
        report = pipeline.run(Path(file_path), progress_callback=_update_progress)
        report_path = output_dir / "report.json"
        report_path.write_text(json.dumps(report.model_dump(), cls=_NumpyEncoder, indent=2))
        return {"status": "completed", "job_id": job_id}
    except Exception as exc:
        logger.exception("Job %s failed: %s", job_id, exc)
        raise
