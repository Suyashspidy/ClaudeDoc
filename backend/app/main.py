"""FastAPI application entry point."""
from __future__ import annotations
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from config.settings import settings

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ClaudeDoc QA Agent starting up (version %s)", settings.app_version)
    yield
    logger.info("ClaudeDoc QA Agent shutting down")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered document quality assurance — detects scanning artifacts, defects, and quality issues.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Serve annotated thumbnails as static files
app.mount("/static", StaticFiles(directory=str(settings.report_dir)), name="static")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": settings.app_version}
