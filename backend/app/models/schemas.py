"""Pydantic schemas for API request/response models."""
from __future__ import annotations
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class IssueType(str, Enum):
    scanner_artifact = "scanner_artifact"
    foreign_object = "foreign_object"
    folded = "folded"
    blank = "blank"
    skew = "skew"
    missing = "missing"


class Severity(str, Enum):
    critical = "critical"
    warning = "warning"
    info = "info"


class JobStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class PageIssue(BaseModel):
    type: IssueType
    severity: Severity
    description: str
    location: str
    confidence: float = Field(ge=0.0, le=1.0)
    recommended_action: str
    bounding_box: Optional[BoundingBox] = None


class PageReport(BaseModel):
    page_number: int
    issues: List[PageIssue] = []
    quality_score: float = Field(ge=0.0, le=100.0)
    annotated_image_path: Optional[str] = None


class IssuesSummary(BaseModel):
    critical: int = 0
    warnings: int = 0
    info: int = 0


class ScannerHealth(BaseModel):
    artifact_detected: bool
    affected_pages: List[int] = []
    cleaning_recommended: bool
    artifact_positions: List[dict] = []


class QAReport(BaseModel):
    job_id: str
    document_name: str
    total_pages: int
    processing_time_seconds: float
    overall_quality_score: float = Field(ge=0.0, le=100.0)
    issues_summary: IssuesSummary
    pages: List[PageReport]
    scanner_health: ScannerHealth
    ai_narrative: Optional[str] = None


class UploadResponse(BaseModel):
    job_id: str
    filename: str
    status: JobStatus
    message: str


class StatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = Field(ge=0.0, le=100.0)
    message: str
    error: Optional[str] = None
