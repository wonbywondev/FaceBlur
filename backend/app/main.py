import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .api.routes import upload, analyze, process, debug
from .core.security import DataRetentionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info(f"Starting {settings.app_name}")

    # Create required directories
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.model_dir.mkdir(parents=True, exist_ok=True)

    # Clean up expired data
    retention_manager = DataRetentionManager(
        base_path=settings.upload_dir,
        retention_hours=settings.retention_hours
    )
    deleted = retention_manager.cleanup_expired()
    if deleted:
        logger.info(f"Cleaned up {deleted} expired items")

    yield

    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title=settings.app_name,
    description="AI-powered face blur service for privacy protection in videos",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router, prefix="/api/v1")
app.include_router(analyze.router, prefix="/api/v1")
app.include_router(process.router, prefix="/api/v1")
app.include_router(debug.router, prefix="/api/v1")  # Debug endpoints for testing


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": settings.app_name,
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/api/v1/info")
async def get_service_info():
    """Get service capabilities and limits."""
    return {
        "max_video_size_mb": settings.max_video_size_mb,
        "max_video_duration_seconds": 600,
        "max_reference_images": settings.max_reference_images,
        "supported_video_formats": list(settings.allowed_video_extensions),
        "supported_image_formats": list(settings.allowed_image_extensions),
        "blur_types": ["gaussian", "mosaic", "blackout"],
        "data_retention_hours": settings.retention_hours
    }
