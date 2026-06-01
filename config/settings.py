"""Central configuration for all detection thresholds and application settings."""
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Application
    app_name: str = "ClaudeDoc QA Agent"
    app_version: str = "1.0.0"
    debug: bool = False

    # API
    host: str = "0.0.0.0"
    port: int = 8000

    # Google Gemini
    google_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker: str = "redis://localhost:6379/0"
    celery_backend: str = "redis://localhost:6379/1"

    # Storage
    upload_dir: Path = Path("uploads")
    report_dir: Path = Path("reports")
    cache_dir: Path = Path("cache")
    max_upload_size_mb: int = 200

    # GPU
    use_gpu: bool = True
    rocm_visible_devices: str = "0"

    # ---- Blank Page Detection ----
    blank_page_pixel_density_threshold: float = 0.02  # < 2% dark pixels = blank
    blank_page_std_threshold: float = 10.0            # std dev of pixel intensities

    # ---- Skew Detection ----
    skew_threshold_degrees: float = 2.0
    skew_max_angle_degrees: float = 45.0

    # ---- Scanner Artifact Detection ----
    artifact_line_span_ratio: float = 0.80        # must span >80% of dimension
    artifact_intensity_variance_max: float = 10.0  # uniform intensity
    artifact_min_pages_same_position: int = 3      # same pos on N+ pages
    artifact_line_width_min: int = 1
    artifact_line_width_max: int = 3
    artifact_position_tolerance_px: int = 5
    artifact_conditions_required: int = 3

    # ---- Foreign Object Detection ----
    yolo_model_path: str = "yolov8n.pt"
    yolo_confidence_threshold: float = 0.70
    foreign_object_critical_coverage: float = 0.05   # > 5% = critical
    foreign_object_warning_coverage: float = 0.01    # 1-5% = warning

    # ---- Folded Page Detection ----
    fold_triangle_area_min: int = 500               # min px² for fold triangle
    fold_corner_search_ratio: float = 0.15          # search in outer 15% of corners

    # ---- Missing Page Detection ----
    page_number_confidence_threshold: float = 0.6

    # ---- Report ----
    thumbnail_width: int = 400
    thumbnail_height: int = 566
    report_dpi: int = 150

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure storage dirs exist
for d in [settings.upload_dir, settings.report_dir, settings.cache_dir]:
    d.mkdir(parents=True, exist_ok=True)
