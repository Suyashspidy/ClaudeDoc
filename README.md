# ClaudeDoc — AI Document Quality Assurance Agent

ClaudeDoc is an AI-powered document QA system that automatically inspects scanned
PDFs and images for quality defects — missing pages, blank pages, skew, and folded
corners — and produces a structured, page-by-page quality report with an overall
quality score and an optional plain-English narrative.

It is built for **document-processing companies, archives, and enterprise scanning
workflows** (e.g. high-volume production scanning), where large batches of scanned
documents need automated quality control before being accepted into a repository.

---

## What it does

You upload a scanned PDF (or image). ClaudeDoc runs it through an asynchronous
detection pipeline and returns a JSON report containing:

- a **per-page list of detected issues** with severity, location, confidence, and a recommended action,
- a **quality score** for every page and an **overall document score** (0–100),
- a **summary** of critical / warning / info issues,
- an optional **AI-generated narrative** summarising the document's condition.

### Detection modules

| Module | Status | What it finds |
|---|---|---|
| **Missing Pages** | ✅ Active | Gaps in printed page-number sequences (via the PDF text layer) and metadata page-count mismatches |
| **Blank Pages** | ✅ Active | Near-empty pages, detected by pixel density + intensity standard deviation |
| **Skew / Rotation** | ✅ Active | Pages scanned at an angle, with the estimated rotation in degrees |
| **Folded Corners** | ✅ Active | Triangular corner folds, via OpenCV contour analysis |
| **Foreign Objects** | 🚧 In development | Cards, IDs, receipts, slips, and other papers scanned *over* the page and hiding text. A custom detector is being trained — see [Roadmap](#roadmap). |
| **Scanner Artifacts** | ⏸️ Disabled | Dirty-scanner streak lines. Present in the codebase but not wired into the pipeline (too many false positives on text). |

> **Note:** The `foreign_object.py` and `scanner_artifact.py` detector modules exist
> in the source tree but are **not** currently part of the live pipeline. Only the
> four "Active" modules above run today.

---

## Architecture

```
┌──────────────┐     upload      ┌──────────────┐    enqueue     ┌──────────────┐
│   Streamlit  │ ──────────────► │   FastAPI    │ ─────────────► │    Redis     │
│   frontend   │ ◄────────────── │   backend    │ ◄───────────── │   (broker)   │
└──────────────┘   poll status   └──────────────┘    results     └──────┬───────┘
                                                                         │
                                                                  ┌──────▼───────┐
                                                                  │    Celery    │
                                                                  │    worker    │  ← runs the QA pipeline
                                                                  └──────────────┘
```

- **Backend:** FastAPI (REST API) + Celery (async job processing) + Redis (broker/result store)
- **Document processing:** PyMuPDF (page rasterisation + text-layer extraction) and OpenCV (image analysis)
- **Detectors:** pure-Python/OpenCV modules orchestrated by `DocumentQAPipeline`
- **Report narrative:** Google Gemini API (optional; falls back to a templated summary if no key is set)
- **Frontend:** Streamlit drag-and-drop UI with an issues table, copy-paste page list, and JSON download
- **Deployment target:** optionally an AMD ROCm GPU host (e.g. DigitalOcean MI300X) for future GPU-accelerated detectors; the active detectors run fine on CPU

### Project structure

```
ClaudeDoc/
├── backend/
│   └── app/
│       ├── detectors/          # detection modules
│       │   ├── blank_page.py        ✅ active
│       │   ├── skew.py              ✅ active
│       │   ├── folded_page.py       ✅ active
│       │   ├── missing_page.py      ✅ active
│       │   ├── foreign_object.py    🚧 in development
│       │   └── scanner_artifact.py  ⏸️ disabled
│       ├── services/
│       │   ├── pipeline.py          # orchestrates the detectors → QAReport
│       │   ├── pdf_processor.py     # PyMuPDF page extraction + metadata
│       │   ├── annotator.py         # draws issue overlays on page thumbnails
│       │   ├── quality_scorer.py    # per-page and document scoring
│       │   └── report_generator.py  # Gemini narrative + (optional) PDF builder
│       ├── models/schemas.py        # Pydantic report schema
│       ├── api/routes.py            # FastAPI endpoints
│       ├── worker.py                # Celery task definitions
│       └── main.py                  # FastAPI app entry point
├── frontend/app.py                  # Streamlit UI
├── huggingface/app.py               # single-process entry point for HF Spaces
├── config/settings.py               # centralised configuration / thresholds
├── docker/                          # Dockerfiles (backend, worker, frontend)
├── docker-compose.yml
├── sample_docs/generate_samples.py  # generates synthetic test PDFs
├── requirements.txt
├── requirements-rocm.txt            # AMD ROCm GPU variant of dependencies
└── .env.example
```

---

## Installation

### Prerequisites

- **Python 3.10+**
- **Redis** (for Celery broker/result backend)
- A **Google Gemini API key** *(optional — only needed for the AI narrative; without it, a templated summary is used)*

### Option A — Docker (recommended)

```bash
git clone https://github.com/Suyashspidy/ClaudeDoc.git
cd ClaudeDoc

# 1. Create your environment file
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY (optional but recommended)

# 2. Build and start all services (backend + worker + Redis + frontend)
docker-compose up --build
```

Then open:

- **Frontend (Streamlit):** http://localhost:8501
- **API docs (Swagger):** http://localhost:8000/docs
- **API base URL:** http://localhost:8000/api/v1

### Option B — Local (without Docker)

```bash
git clone https://github.com/Suyashspidy/ClaudeDoc.git
cd ClaudeDoc

# 1. Install dependencies
pip install -r requirements.txt
#   For AMD ROCm GPU hosts, use the ROCm variant instead:
#   pip install -r requirements-rocm.txt

# 2. Environment
cp .env.example .env
#   Set GOOGLE_API_KEY in .env (optional)

# 3. Make sure Redis is running on localhost:6379
#    (e.g. `redis-server`, or via Docker: `docker run -p 6379:6379 redis`)

# 4. Start the services (each in its own terminal)

# Terminal 1 — API
uvicorn backend.app.main:app --reload --port 8000

# Terminal 2 — Celery worker
celery -A backend.app.worker.celery_app worker --loglevel=info

# Terminal 3 — Streamlit frontend
streamlit run frontend/app.py

# (optional) Terminal 4 — generate synthetic test documents
python sample_docs/generate_samples.py
```

> **Windows note:** the frontend's API base URL is set at the top of
> `frontend/app.py` (`API_BASE`). Point it at `http://localhost:8000/api/v1` for a
> local run, or at your remote backend's address.

---

## Configuration

All settings live in `config/settings.py` and can be overridden via environment
variables in `.env`. The most relevant ones:

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | — | Google Gemini API key for the AI narrative *(optional)* |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model used for report narration |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `CELERY_BROKER` | `redis://localhost:6379/0` | Celery broker URL |
| `CELERY_BACKEND` | `redis://localhost:6379/1` | Celery result backend URL |
| `USE_GPU` | `true` | Enable AMD ROCm GPU acceleration (for future GPU detectors) |
| `ROCM_VISIBLE_DEVICES` | `0` | Which ROCm device to use |
| `MAX_UPLOAD_SIZE_MB` | `200` | Maximum upload size |
| `BLANK_PAGE_PIXEL_DENSITY_THRESHOLD` | `0.02` | Dark-pixel fraction below which a page is blank |
| `SKEW_THRESHOLD_DEGREES` | `2.0` | Minimum skew angle to flag |

---

## API reference

Base path: `/api/v1`

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload a PDF/image; returns a `job_id` |
| `GET` | `/status/{job_id}` | Poll job progress (0–100%) and state |
| `GET` | `/report/{job_id}` | Full JSON QA report |
| `GET` | `/report/{job_id}/pdf` | Download a PDF report (if generated) |
| `GET` | `/report/{job_id}/page/{page_num}` | Annotated page thumbnail |
| `DELETE` | `/job/{job_id}` | Delete a job and its files |

Interactive Swagger docs: **http://localhost:8000/docs**

### QA report schema (abridged)

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
  ]
}
```

---

## Running tests

```bash
pytest backend/tests/ -v
```

Covers blank-page, skew, folded-page, missing-page, scanner-artifact, and
quality-scoring logic.

---

## Roadmap

- **Foreign-object detection** — train a custom object detector to flag cards, IDs,
  receipts, slips, bills, and other papers scanned *over* a page and hiding text.
  This targets enterprise scanning workflows where such objects are accidentally
  captured. Work in progress.
- Validate the blank-page, skew, and folded-corner detectors against real defective
  scans.
- Re-evaluate scanner-artifact detection with a more precise, lower-false-positive
  algorithm.

---

## License

MIT
