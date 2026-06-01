# ClaudeDoc — AI Document Quality Assurance Agent

An AI-powered document QA system that automatically detects scanning artifacts,
document defects, and quality issues in scanned PDFs and images.

Built for document processing companies, archives, and enterprise scanning workflows.

---

## Features

| Detection Module | What It Finds |
|---|---|
| **Missing Pages** | Gaps in page number sequences, metadata mismatches |
| **Folded Corners** | Triangular corner anomalies using OpenCV contour analysis |
| **Blank Pages** | Near-blank pages by pixel density + std deviation |
| **Scanner Artifacts** | Vertical/horizontal dirty-scanner lines using 4-condition cross-page algorithm |
| **Foreign Objects** | Papers, bookmarks, hands blocking content (YOLOv8 + SAM) |
| **Skew / Rotation** | Pages scanned at an angle, reports exact degrees |
| **AI Report** | Professional plain-English QA narrative via Claude |

---

## Quick Start

### Option A — Docker (recommended)

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY

docker-compose up --build
```

- **Frontend:** http://localhost:8501
- **API docs:** http://localhost:8000/docs
- **API base:** http://localhost:8000/api/v1

### Option B — Local (without Docker)

**Prerequisites:** Python 3.10+, Redis running on `localhost:6379`, Tesseract OCR

```bash
# 1. Install dependencies
pip install -r requirements.txt

# For AMD GPU (ROCm):
pip install -r requirements-rocm.txt

# 2. Environment
cp .env.example .env
# Set ANTHROPIC_API_KEY in .env

# 3. Start services (4 terminals)
# Terminal 1 — API
uvicorn backend.app.main:app --reload --port 8000

# Terminal 2 — Celery worker
celery -A backend.app.worker.celery_app worker --loglevel=info

# Terminal 3 — Streamlit frontend
streamlit run frontend/app.py

# Terminal 4 — Generate sample test docs (optional)
python sample_docs/generate_samples.py
```

---

## AMD GPU Acceleration

The pipeline uses ROCm + PyTorch for GPU-accelerated inference:

- **YOLOv8** foreign object detection runs on AMD GPU
- **SAM** segmentation runs on AMD GPU
- Set `USE_GPU=true` in `.env` and use `docker-compose.yml` which mounts `/dev/kfd` and `/dev/dri`

```bash
# Verify ROCm is detected
python -c "import torch; print(torch.cuda.is_available())"  # True with ROCm
```

---

## API Reference

```
POST   /api/v1/upload                          Upload PDF/image for processing
GET    /api/v1/status/{job_id}                 Poll job progress (0-100%)
GET    /api/v1/report/{job_id}                 Full JSON QA report
GET    /api/v1/report/{job_id}/pdf             Download PDF report
GET    /api/v1/report/{job_id}/page/{page_num} Annotated page thumbnail
DELETE /api/v1/job/{job_id}                    Clean up job files
```

Interactive Swagger docs: http://localhost:8000/docs

---

## QA Report Schema

```json
{
  "document_name": "contract.pdf",
  "total_pages": 12,
  "processing_time_seconds": 18.4,
  "overall_quality_score": 72.5,
  "issues_summary": { "critical": 2, "warnings": 3, "info": 1 },
  "pages": [
    {
      "page_number": 3,
      "quality_score": 65.0,
      "issues": [
        {
          "type": "folded",
          "severity": "critical",
          "description": "Folded corner detected at bottom-right (confidence 85%)",
          "location": "bottom-right",
          "confidence": 0.85,
          "recommended_action": "Unfold the page and rescan."
        }
      ]
    }
  ],
  "scanner_health": {
    "artifact_detected": true,
    "affected_pages": [10, 11, 12],
    "cleaning_recommended": true,
    "artifact_positions": [{"orientation": "vertical", "position": 350}]
  },
  "ai_narrative": "This document presents moderate quality concerns..."
}
```

---

## Sample Test Documents

```bash
python sample_docs/generate_samples.py
```

Generates:

| File | Tests |
|---|---|
| `test_missing_pages_5_6.pdf` | Missing page detection (pages 5-6 absent) |
| `test_folded_corner_page3.pdf` | Folded bottom-right corner on page 3 |
| `test_scanner_artifact_pages10_15.pdf` | Vertical artifact on pages 10-15 |
| `test_foreign_object_page8.pdf` | Bookmark blocking 20% of page 8 |
| `test_perfect_document.pdf` | Clean document (expected score: 100) |

---

## Running Tests

```bash
pytest backend/tests/ -v
```

Test coverage:
- Blank page detection (5 tests)
- Skew detection (4 tests)
- Scanner artifact cross-page analysis (3 tests)
- Folded page detection (4 tests)
- Missing page / sequence gap analysis (6 tests)
- Quality scoring functions (7 tests)

---

## Project Structure

```
ClaudeDoc/
├── backend/
│   └── app/
│       ├── detectors/         # 6 detection modules
│       │   ├── blank_page.py
│       │   ├── folded_page.py
│       │   ├── foreign_object.py
│       │   ├── missing_page.py
│       │   ├── scanner_artifact.py
│       │   └── skew.py
│       ├── services/
│       │   ├── pipeline.py        # Main QA pipeline orchestrator
│       │   ├── pdf_processor.py   # PyMuPDF page extraction
│       │   ├── annotator.py       # Bounding box overlays
│       │   ├── quality_scorer.py  # Score calculation
│       │   └── report_generator.py # Claude API + PDF reports
│       ├── models/schemas.py      # Pydantic data models
│       ├── api/routes.py          # FastAPI endpoints
│       ├── worker.py              # Celery task definitions
│       └── main.py                # FastAPI app entry point
├── frontend/app.py                # Streamlit UI
├── config/settings.py             # Centralised configuration
├── docker/                        # Dockerfiles (API, worker, frontend)
├── docker-compose.yml
├── sample_docs/generate_samples.py
├── .github/workflows/ci.yml
├── requirements.txt
├── requirements-rocm.txt          # AMD ROCm GPU variant
└── .env.example
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | **Required.** Claude API key |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Claude model for report generation |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `USE_GPU` | `true` | Enable AMD ROCm GPU acceleration |
| `BLANK_PAGE_PIXEL_DENSITY_THRESHOLD` | `0.02` | Fraction of dark pixels below which page is blank |
| `SKEW_THRESHOLD_DEGREES` | `2.0` | Minimum skew angle to flag |
| `YOLO_CONFIDENCE_THRESHOLD` | `0.70` | YOLO detection confidence cutoff |
| `ARTIFACT_CONDITIONS_REQUIRED` | `3` | Conditions needed to classify line as artifact |

---

## License

MIT
