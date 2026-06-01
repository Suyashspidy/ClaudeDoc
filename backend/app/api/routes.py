"""FastAPI route definitions for the ClaudeDoc QA API."""
from __future__ import annotations
import json
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.models.schemas import JobStatus, QAReport, StatusResponse, UploadResponse
from app.worker import process_document
from config.settings import settings

router = APIRouter(prefix="/api/v1", tags=["QA"])

ALLOWED_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/tiff",
    "image/bmp",
    "image/webp",
}
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}


@router.post("/upload", response_model=UploadResponse, summary="Upload a document for QA processing")
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {suffix}")

    file_size = 0
    job_id = str(uuid.uuid4())
    upload_path = settings.upload_dir / f"{job_id}{suffix}"

    with open(upload_path, "wb") as out:
        while chunk := await file.read(1024 * 1024):  # 1 MB chunks
            file_size += len(chunk)
            if file_size > settings.max_upload_size_mb * 1024 * 1024:
                upload_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File exceeds maximum allowed size.")
            out.write(chunk)

    # Dispatch async Celery task
    process_document.apply_async(args=[job_id, str(upload_path)], task_id=job_id)

    return UploadResponse(
        job_id=job_id,
        filename=file.filename,
        status=JobStatus.pending,
        message="Document queued for processing.",
    )


@router.get("/status/{job_id}", response_model=StatusResponse, summary="Get job processing status")
async def get_status(job_id: str) -> StatusResponse:
    from celery.result import AsyncResult
    result = AsyncResult(job_id)

    if result.state == "PENDING":
        return StatusResponse(job_id=job_id, status=JobStatus.pending, progress=0, message="Waiting in queue…")
    if result.state == "PROGRESS":
        meta = result.info or {}
        return StatusResponse(job_id=job_id, status=JobStatus.processing, progress=meta.get("progress", 0), message=meta.get("message", "Processing…"))
    if result.state == "SUCCESS":
        return StatusResponse(job_id=job_id, status=JobStatus.completed, progress=100, message="Processing complete.")
    if result.state == "FAILURE":
        return StatusResponse(job_id=job_id, status=JobStatus.failed, progress=0, message="Processing failed.", error=str(result.info))

    return StatusResponse(job_id=job_id, status=JobStatus.processing, progress=0, message=result.state)


@router.get("/report/{job_id}", summary="Get the full JSON QA report")
async def get_report(job_id: str) -> JSONResponse:
    report_path = settings.report_dir / job_id / "report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found. Job may still be processing.")
    return JSONResponse(content=json.loads(report_path.read_text()))


@router.get("/report/{job_id}/pdf", summary="Download the PDF QA report")
async def download_pdf_report(job_id: str) -> FileResponse:
    pdf_path = settings.report_dir / job_id / "qa_report.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF report not found.")
    return FileResponse(str(pdf_path), media_type="application/pdf", filename=f"qa_report_{job_id}.pdf")


@router.get("/report/{job_id}/page/{page_num}", summary="Get an annotated page thumbnail")
async def get_annotated_page(job_id: str, page_num: int) -> FileResponse:
    thumb_path = settings.report_dir / job_id / "thumbnails" / f"page_{page_num}_annotated.jpg"
    if not thumb_path.exists():
        raise HTTPException(status_code=404, detail=f"Annotated page {page_num} not found.")
    return FileResponse(str(thumb_path), media_type="image/jpeg")


@router.delete("/job/{job_id}", summary="Delete a job and its files")
async def delete_job(job_id: str) -> dict:
    job_dir = settings.report_dir / job_id
    upload_files = list(settings.upload_dir.glob(f"{job_id}.*"))

    if not job_dir.exists() and not upload_files:
        raise HTTPException(status_code=404, detail="Job not found.")

    if job_dir.exists():
        shutil.rmtree(job_dir)
    for f in upload_files:
        f.unlink(missing_ok=True)

    return {"job_id": job_id, "deleted": True}
