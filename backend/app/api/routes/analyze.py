import cv2
import base64
import logging
import asyncio
from typing import Dict, List
from uuid import uuid4

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from ...config import get_settings
from ...models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    AnalysisResult,
    DetectedFace,
    FaceAppearance,
    ProcessStatus
)
from ..deps import (
    sessions,
    analyses,
    get_face_detector,
    get_face_embedder,
    get_face_matcher,
    get_video_processor
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analyze", tags=["analyze"])
settings = get_settings()


def run_analysis(analysis_id: str, session_id: str, expected_persons: str = "10"):
    """Background task to run video analysis (sync function for thread execution)."""
    try:
        logger.info(f"Starting analysis {analysis_id} for session {session_id} (expected ~{expected_persons} people)")

        session = sessions.get(session_id)
        if not session:
            analyses[analysis_id]["status"] = ProcessStatus.FAILED
            analyses[analysis_id]["error"] = "Session not found"
            return

        video_path = session.get("video_path")
        reference_embedding = session.get("reference_embedding")

        if not video_path:
            analyses[analysis_id]["status"] = ProcessStatus.FAILED
            analyses[analysis_id]["error"] = "Video not found"
            return

        # Update progress: Loading models
        analyses[analysis_id]["progress"] = 5
        logger.info("Loading AI models...")

        detector = get_face_detector()
        embedder = get_face_embedder()
        matcher = get_face_matcher(expected_persons=expected_persons)
        processor = get_video_processor()

        analyses[analysis_id]["progress"] = 10
        logger.info("Models loaded, starting video processing...")

        all_detections = []

        # Process video frames
        def progress_callback(progress, frames):
            # Scale progress from 10-90%
            scaled_progress = 10 + (progress * 0.8)
            analyses[analysis_id]["progress"] = scaled_progress
            analyses[analysis_id]["frames_processed"] = frames
            logger.debug(f"Progress: {scaled_progress:.1f}%, Frames: {frames}")

        for frame_data in detector.process_video(
            video_path,
            sample_rate=settings.frame_sample_rate,
            progress_callback=progress_callback
        ):
            frame = frame_data["frame"]
            timestamp = frame_data["timestamp"]
            frame_number = frame_data["frame_number"]

            # Detect and embed faces in this frame
            face_results = embedder.detect_and_embed(frame)

            for face in face_results:
                all_detections.append({
                    "embedding": face["embedding"],
                    "bbox": face["bbox"],
                    "timestamp": timestamp,
                    "frame_number": frame_number,
                    "frame": frame.copy(),
                    "det_score": face["det_score"]
                })

            analyses[analysis_id]["faces_detected"] = len(all_detections)

        # Update progress: Clustering faces
        analyses[analysis_id]["progress"] = 90
        logger.info(f"Clustering {len(all_detections)} face detections...")

        # Cluster faces by identity
        clusters = matcher.cluster_faces(all_detections, reference_embedding)

        # Build result
        detected_faces = []
        for cluster in clusters:
            # Generate thumbnail
            if cluster.get("thumbnail_frame") is not None:
                thumb_bbox = cluster["appearances"][0]["bbox"]
                thumbnail = processor.generate_thumbnail(
                    cluster["thumbnail_frame"],
                    thumb_bbox
                )
                # Encode as base64
                _, buffer = cv2.imencode('.jpg', thumbnail)
                thumbnail_b64 = base64.b64encode(buffer).decode('utf-8')
            else:
                thumbnail_b64 = ""

            # Merge appearances into ranges
            merged_appearances = matcher.merge_appearances(cluster["appearances"])

            # Determine if this face should be blurred by default
            is_ref = cluster.get("is_reference", False)

            detected_faces.append(DetectedFace(
                face_id=cluster["face_id"],
                thumbnail=thumbnail_b64,
                first_appearance=cluster["appearances"][0]["timestamp"],
                appearances=[
                    FaceAppearance(
                        start=app["start"],
                        end=app["end"],
                        bbox=app["bbox"]
                    )
                    for app in merged_appearances
                ],
                appearance_count=len(cluster["appearances"]),  # Raw detection count
                similarity_to_reference=cluster.get("similarity_to_reference", 0),
                is_reference=is_ref,
                blur_enabled=not is_ref  # Blur everyone except reference by default
            ))

        # Sort faces by appearance count (most frequent first)
        detected_faces.sort(key=lambda f: f.appearance_count, reverse=True)

        # Store result
        analyses[analysis_id]["progress"] = 100
        analyses[analysis_id]["result"] = AnalysisResult(
            analysis_id=analysis_id,
            faces=detected_faces,
            total_faces=len(detected_faces),
            reference_matches=sum(1 for f in detected_faces if f.is_reference),
            status=ProcessStatus.COMPLETED
        )
        analyses[analysis_id]["status"] = ProcessStatus.COMPLETED
        analyses[analysis_id]["clusters"] = clusters  # Store for blur processing

        # Store in session for blur processing
        sessions[session_id]["analysis_id"] = analysis_id
        logger.info(f"Analysis {analysis_id} completed: {len(detected_faces)} faces found")

    except Exception as e:
        logger.error(f"Analysis error for {analysis_id}: {e}", exc_info=True)
        analyses[analysis_id]["status"] = ProcessStatus.FAILED
        analyses[analysis_id]["error"] = str(e)


@router.post("", response_model=AnalyzeResponse)
async def start_analysis(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks
):
    """
    Start video analysis to detect and identify faces.

    This is an asynchronous operation. Use the analysis_id to check progress.
    """
    # Parse video_id to get session
    if ":" not in request.video_id:
        raise HTTPException(status_code=400, detail="Invalid video_id format")

    session_id = request.video_id.split(":")[0]

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]

    if not session.get("video_path"):
        raise HTTPException(status_code=400, detail="No video uploaded")

    if not session.get("reference_embedding") is not None:
        # Reference is optional, but if provided in request, validate it
        if request.reference_id and ":" in request.reference_id:
            ref_session_id = request.reference_id.split(":")[0]
            if ref_session_id != session_id:
                raise HTTPException(status_code=400, detail="Reference ID mismatch")

    # Create analysis record
    analysis_id = str(uuid4())
    video_info = session.get("video_info", {})
    estimated_time = int(video_info.get("duration", 60) / 10)  # Rough estimate

    # Get expected persons value
    expected_persons = request.expected_persons.value if request.expected_persons else "10"

    analyses[analysis_id] = {
        "id": analysis_id,
        "session_id": session_id,
        "status": ProcessStatus.PROCESSING,
        "progress": 0,
        "faces_detected": 0,
        "frames_processed": 0,
        "expected_persons": expected_persons,
        "result": None,
        "error": None
    }

    # Start background analysis
    background_tasks.add_task(run_analysis, analysis_id, session_id, expected_persons)

    return AnalyzeResponse(
        analysis_id=analysis_id,
        status=ProcessStatus.PROCESSING,
        estimated_time=estimated_time
    )


@router.get("/{analysis_id}/status")
async def get_analysis_status(analysis_id: str):
    """Get current analysis progress."""
    if analysis_id not in analyses:
        raise HTTPException(status_code=404, detail="Analysis not found")

    analysis = analyses[analysis_id]
    status = analysis["status"]

    # Ensure status is serialized as string
    status_value = status.value if hasattr(status, 'value') else str(status)

    return {
        "analysis_id": analysis_id,
        "status": status_value,
        "progress": analysis.get("progress", 0),
        "faces_detected": analysis.get("faces_detected", 0),
        "frames_processed": analysis.get("frames_processed", 0),
        "error": analysis.get("error")
    }


@router.get("/{analysis_id}/result", response_model=AnalysisResult)
async def get_analysis_result(analysis_id: str):
    """Get completed analysis results."""
    if analysis_id not in analyses:
        raise HTTPException(status_code=404, detail="Analysis not found")

    analysis = analyses[analysis_id]

    if analysis["status"] == ProcessStatus.PROCESSING:
        raise HTTPException(status_code=202, detail="Analysis still in progress")

    if analysis["status"] == ProcessStatus.FAILED:
        raise HTTPException(
            status_code=500,
            detail=analysis.get("error", "Analysis failed")
        )

    if analysis["result"] is None:
        raise HTTPException(status_code=500, detail="No result available")

    return analysis["result"]
