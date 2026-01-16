import cv2
import base64
import logging
from pathlib import Path
from typing import List
from uuid import uuid4

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
import aiofiles

from ...config import get_settings
from ...models.schemas import (
    VideoUploadResponse,
    ReferenceUploadResponse,
    ProcessStatus,
    Resolution,
    ErrorResponse
)
from ...core.security import sanitize_filename
from ..deps import (
    sessions,
    create_session,
    get_session_dir,
    get_face_detector,
    get_face_embedder
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/upload", tags=["upload"])
settings = get_settings()


@router.post("/video", response_model=VideoUploadResponse)
async def upload_video(
    file: UploadFile = File(...)
):
    """
    Upload a video file for processing.

    - Supports MP4 and MOV formats
    - Maximum size: 500MB
    - Maximum duration: 10 minutes
    """
    # Validate file extension
    filename = sanitize_filename(file.filename or "video.mp4")
    ext = Path(filename).suffix.lower()

    if ext not in settings.allowed_video_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Allowed: {settings.allowed_video_extensions}"
        )

    # Create session
    session_id = create_session()
    session_dir = get_session_dir(session_id)

    # Save file
    video_id = str(uuid4())
    video_path = session_dir / f"{video_id}{ext}"

    try:
        # Stream file to disk
        file_size = 0
        async with aiofiles.open(video_path, 'wb') as out_file:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                file_size += len(chunk)
                if file_size > settings.max_video_size_mb * 1024 * 1024:
                    await out_file.close()
                    video_path.unlink()
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum: {settings.max_video_size_mb}MB"
                    )
                await out_file.write(chunk)

        # Get video info
        detector = get_face_detector()
        video_info = detector.get_video_info(str(video_path))

        # Validate duration (10 minutes max)
        if video_info["duration"] > 600:
            video_path.unlink()
            raise HTTPException(
                status_code=400,
                detail="Video too long. Maximum duration: 10 minutes"
            )

        # Update session
        sessions[session_id]["video_id"] = video_id
        sessions[session_id]["video_path"] = str(video_path)
        sessions[session_id]["video_info"] = video_info

        return VideoUploadResponse(
            video_id=f"{session_id}:{video_id}",
            filename=filename,
            duration=video_info["duration"],
            resolution=Resolution(
                width=video_info["width"],
                height=video_info["height"]
            ),
            status=ProcessStatus.UPLOADED
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        if video_path.exists():
            video_path.unlink()
        raise HTTPException(status_code=500, detail="Error processing video")


@router.post("/reference", response_model=ReferenceUploadResponse)
async def upload_reference(
    video_id: str,
    files: List[UploadFile] = File(...)
):
    """
    Upload reference images of the person to keep unblurred.

    - Supports JPG and PNG formats
    - 1-5 images recommended
    - Clear, front-facing photos work best
    """
    # Parse video_id to get session
    if ":" not in video_id:
        raise HTTPException(status_code=400, detail="Invalid video_id format")

    session_id = video_id.split(":")[0]

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    if len(files) > settings.max_reference_images:
        raise HTTPException(
            status_code=400,
            detail=f"Too many images. Maximum: {settings.max_reference_images}"
        )

    if len(files) == 0:
        raise HTTPException(status_code=400, detail="At least one image required")

    session_dir = get_session_dir(session_id)
    reference_id = str(uuid4())
    reference_dir = session_dir / "references"
    reference_dir.mkdir(exist_ok=True)

    saved_images = []
    embedder = get_face_embedder()

    try:
        for i, file in enumerate(files):
            filename = sanitize_filename(file.filename or f"ref_{i}.jpg")
            ext = Path(filename).suffix.lower()

            if ext not in settings.allowed_image_extensions:
                continue

            # Save image
            image_path = reference_dir / f"{reference_id}_{i}{ext}"
            content = await file.read()

            if len(content) > settings.max_image_size_mb * 1024 * 1024:
                continue

            async with aiofiles.open(image_path, 'wb') as out_file:
                await out_file.write(content)

            saved_images.append(str(image_path))

        if not saved_images:
            raise HTTPException(
                status_code=400,
                detail="No valid images uploaded"
            )

        # Extract embeddings from reference images
        import numpy as np

        images = []
        for path in saved_images:
            img = cv2.imread(path)
            if img is not None:
                images.append(img)

        reference_embedding = embedder.get_embeddings_from_images(images)

        if reference_embedding is None:
            raise HTTPException(
                status_code=400,
                detail="No faces detected in reference images"
            )

        # Store embedding in session (in-memory only for privacy)
        sessions[session_id]["reference_id"] = reference_id
        sessions[session_id]["reference_embedding"] = reference_embedding
        sessions[session_id]["reference_images"] = saved_images

        return ReferenceUploadResponse(
            reference_id=f"{session_id}:{reference_id}",
            face_count=len(images),
            embeddings_generated=True
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing reference images: {e}")
        raise HTTPException(status_code=500, detail="Error processing images")
