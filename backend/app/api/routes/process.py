import logging
import threading
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from ...config import get_settings
from ...models.schemas import (
    ProcessRequest,
    ProcessResponse,
    ProcessStatus,
    BlurType
)
from ...services.video_processor import VideoProcessor, BlurType as VPBlurType
from ...services.face_matcher import FaceMatcher
from ..deps import sessions, analyses, processes

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/process", tags=["process"])
settings = get_settings()

# Process control events (stop and pause signals)
process_controls: dict[str, dict] = {}


def run_blur_processing(process_id: str, analysis_id: str, blur_settings: dict):
    """Background task to process video with blur (runs in separate thread)."""
    logger.info(f"[BG] Starting background blur processing: {process_id}")

    # Get control events
    controls = process_controls.get(process_id, {})
    stop_event = controls.get("stop_event")
    pause_event = controls.get("pause_event")
    try:
        analysis = analyses.get(analysis_id)
        if not analysis:
            logger.error(f"[BG] Analysis not found: {analysis_id}")
            processes[process_id]["status"] = ProcessStatus.FAILED
            processes[process_id]["error"] = "Analysis not found"
            return

        session_id = analysis["session_id"]
        logger.info(f"[BG] Session ID: {session_id}")
        session = sessions.get(session_id)

        if not session:
            logger.error(f"[BG] Session not found: {session_id}")
            processes[process_id]["status"] = ProcessStatus.FAILED
            processes[process_id]["error"] = "Session not found"
            return

        video_path = session.get("video_path")
        logger.info(f"[BG] Video path: {video_path}")
        if not video_path or not Path(video_path).exists():
            logger.error(f"[BG] Video file not found: {video_path}")
            processes[process_id]["status"] = ProcessStatus.FAILED
            processes[process_id]["error"] = "Video file not found"
            return

        # Get clusters with full appearance data
        clusters = analysis.get("clusters", [])
        logger.info(f"[BG] Clusters count: {len(clusters)}")

        # Filter to only faces that should be blurred
        face_ids_to_blur = set(blur_settings["face_ids"])
        blur_targets = []

        # Use FaceMatcher to merge appearances into start/end ranges
        matcher = FaceMatcher()

        for cluster in clusters:
            if cluster["face_id"] in face_ids_to_blur:
                # Convert timestamp-based appearances to start/end ranges
                merged_appearances = matcher.merge_appearances(cluster["appearances"])
                blur_targets.append({
                    "face_id": cluster["face_id"],
                    "appearances": merged_appearances
                })

        logger.info(f"[BG] Blur targets count: {len(blur_targets)}")

        if not blur_targets:
            logger.error("[BG] No valid faces to blur")
            processes[process_id]["status"] = ProcessStatus.FAILED
            processes[process_id]["error"] = "No valid faces to blur"
            return

        # Setup output path
        output_dir = settings.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{process_id}.mp4"
        logger.info(f"[BG] Output path: {output_path}")

        # Convert blur type
        blur_type_map = {
            "gaussian": VPBlurType.GAUSSIAN,
            "mosaic": VPBlurType.MOSAIC,
            "blackout": VPBlurType.BLACKOUT
        }
        blur_type = blur_type_map.get(blur_settings["type"], VPBlurType.GAUSSIAN)

        # Process video
        logger.info(f"[BG] Starting video processing with blur_type={blur_type}, intensity={blur_settings['intensity']}")
        processor = VideoProcessor()

        def progress_callback(progress):
            # Check for stop signal
            if stop_event and stop_event.is_set():
                logger.info(f"[BG] Stop signal received at {progress:.1f}%")
                raise InterruptedError("Processing stopped by user")

            # Check for pause signal (block until resumed)
            if pause_event and pause_event.is_set():
                logger.info(f"[BG] Paused at {progress:.1f}%")
                processes[process_id]["status"] = ProcessStatus.PAUSED
                while pause_event.is_set():
                    if stop_event and stop_event.is_set():
                        raise InterruptedError("Processing stopped by user")
                    pause_event.wait(timeout=0.5)
                processes[process_id]["status"] = ProcessStatus.PROCESSING
                logger.info(f"[BG] Resumed at {progress:.1f}%")

            processes[process_id]["progress"] = progress
            if int(progress) % 20 == 0:
                logger.info(f"[BG] Processing progress: {progress:.1f}%")

        result = processor.process_video_ffmpeg(
            input_path=video_path,
            output_path=str(output_path),
            blur_targets=blur_targets,
            blur_type=blur_type,
            intensity=blur_settings["intensity"],
            progress_callback=progress_callback
        )

        logger.info(f"[BG] Processing completed: {result}")
        processes[process_id]["status"] = ProcessStatus.COMPLETED
        processes[process_id]["output_path"] = str(output_path)
        processes[process_id]["result"] = result

    except InterruptedError as e:
        logger.info(f"[BG] Processing interrupted: {e}")
        processes[process_id]["status"] = ProcessStatus.STOPPED
        processes[process_id]["error"] = str(e)
    except Exception as e:
        logger.error(f"[BG] Processing error: {e}", exc_info=True)
        processes[process_id]["status"] = ProcessStatus.FAILED
        processes[process_id]["error"] = str(e)
    finally:
        # Clean up control events
        if process_id in process_controls:
            del process_controls[process_id]


@router.post("/blur", response_model=ProcessResponse)
async def start_blur_processing(
    request: ProcessRequest,
    background_tasks: BackgroundTasks
):
    """
    Start video blur processing.

    Apply blur to selected faces based on analysis results.
    """
    analysis_id = request.analysis_id
    logger.info(f"Starting blur processing for analysis_id: {analysis_id}")
    logger.info(f"Request blur_settings: {request.blur_settings}")

    if analysis_id not in analyses:
        logger.error(f"Analysis not found: {analysis_id}")
        logger.error(f"Available analyses: {list(analyses.keys())}")
        raise HTTPException(status_code=404, detail="Analysis not found")

    analysis = analyses[analysis_id]
    logger.info(f"Analysis status: {analysis['status']} (type: {type(analysis['status'])})")

    # Compare with string value to handle enum comparison
    status = analysis["status"]
    if hasattr(status, 'value'):
        status_value = status.value
    else:
        status_value = str(status)

    if status_value != ProcessStatus.COMPLETED.value:
        logger.error(f"Analysis not completed. Status: {status_value}")
        raise HTTPException(
            status_code=400,
            detail=f"Analysis not completed. Current status: {status_value}"
        )

    # Validate face IDs
    result = analysis.get("result")
    if not result:
        logger.error("No analysis result found")
        raise HTTPException(status_code=400, detail="No analysis result")

    logger.info(f"Result type: {type(result)}")
    logger.info(f"Result faces count: {len(result.faces)}")

    valid_face_ids = {f.face_id for f in result.faces}
    requested_ids = set(request.blur_settings.face_ids)

    logger.info(f"Valid face IDs: {valid_face_ids}")
    logger.info(f"Requested face IDs: {requested_ids}")

    invalid_ids = requested_ids - valid_face_ids
    if invalid_ids:
        logger.error(f"Invalid face IDs: {invalid_ids}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid face IDs: {invalid_ids}"
        )

    # Create process record
    process_id = str(uuid4())
    processes[process_id] = {
        "id": process_id,
        "analysis_id": analysis_id,
        "status": ProcessStatus.PROCESSING,
        "progress": 0,
        "output_path": None,
        "error": None
    }

    # Create control events for stop/pause
    process_controls[process_id] = {
        "stop_event": threading.Event(),
        "pause_event": threading.Event()
    }

    # Start background processing
    blur_settings = {
        "type": request.blur_settings.type.value,
        "intensity": request.blur_settings.intensity,
        "face_ids": request.blur_settings.face_ids
    }

    background_tasks.add_task(
        run_blur_processing,
        process_id,
        analysis_id,
        blur_settings
    )

    return ProcessResponse(
        process_id=process_id,
        status=ProcessStatus.PROCESSING
    )


@router.get("/{process_id}/status")
async def get_process_status(process_id: str):
    """Get blur processing progress."""
    logger.info(f"[STATUS] Checking status for process: {process_id}")

    if process_id not in processes:
        logger.error(f"[STATUS] Process not found: {process_id}")
        raise HTTPException(status_code=404, detail="Process not found")

    process = processes[process_id]

    # Convert enum to string value for JSON serialization
    status = process["status"]
    status_value = status.value if hasattr(status, 'value') else str(status)

    response = {
        "process_id": process_id,
        "status": status_value,
        "progress": process.get("progress", 0),
        "error": process.get("error")
    }
    logger.info(f"[STATUS] Response: {response}")

    return response


@router.get("/{process_id}/download")
async def download_processed_video(process_id: str):
    """Download the processed video with blur applied."""
    if process_id not in processes:
        raise HTTPException(status_code=404, detail="Process not found")

    process = processes[process_id]

    if process["status"] == ProcessStatus.PROCESSING:
        raise HTTPException(status_code=202, detail="Processing not complete")

    if process["status"] == ProcessStatus.FAILED:
        raise HTTPException(
            status_code=500,
            detail=process.get("error", "Processing failed")
        )

    output_path = process.get("output_path")
    if not output_path or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename=f"blurred_{process_id}.mp4"
    )


@router.post("/{process_id}/update-faces")
async def update_face_blur_settings(
    process_id: str,
    face_ids: list[str],
    blur_enabled: bool
):
    """Update which faces should be blurred (for UI toggle)."""
    if process_id not in processes:
        raise HTTPException(status_code=404, detail="Process not found")

    # This would update the session state for re-processing
    # Implementation depends on your state management approach
    return {"status": "updated", "face_ids": face_ids, "blur_enabled": blur_enabled}


@router.post("/{process_id}/stop")
async def stop_processing(process_id: str):
    """Stop the video processing."""
    if process_id not in processes:
        raise HTTPException(status_code=404, detail="Process not found")

    process = processes[process_id]
    status = process["status"]
    status_value = status.value if hasattr(status, 'value') else str(status)

    if status_value not in [ProcessStatus.PROCESSING.value, ProcessStatus.PAUSED.value]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot stop process in state: {status_value}"
        )

    controls = process_controls.get(process_id)
    if controls and "stop_event" in controls:
        controls["stop_event"].set()
        # Also clear pause to allow the loop to exit
        if "pause_event" in controls:
            controls["pause_event"].clear()
        logger.info(f"[STOP] Stop signal sent for process: {process_id}")
        return {"status": "stopping", "process_id": process_id}

    raise HTTPException(status_code=500, detail="Control events not found")


@router.post("/{process_id}/pause")
async def pause_processing(process_id: str):
    """Pause the video processing."""
    if process_id not in processes:
        raise HTTPException(status_code=404, detail="Process not found")

    process = processes[process_id]
    status = process["status"]
    status_value = status.value if hasattr(status, 'value') else str(status)

    if status_value != ProcessStatus.PROCESSING.value:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot pause process in state: {status_value}"
        )

    controls = process_controls.get(process_id)
    if controls and "pause_event" in controls:
        controls["pause_event"].set()
        logger.info(f"[PAUSE] Pause signal sent for process: {process_id}")
        return {"status": "pausing", "process_id": process_id}

    raise HTTPException(status_code=500, detail="Control events not found")


@router.post("/{process_id}/resume")
async def resume_processing(process_id: str):
    """Resume the video processing."""
    if process_id not in processes:
        raise HTTPException(status_code=404, detail="Process not found")

    process = processes[process_id]
    status = process["status"]
    status_value = status.value if hasattr(status, 'value') else str(status)

    if status_value != ProcessStatus.PAUSED.value:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot resume process in state: {status_value}"
        )

    controls = process_controls.get(process_id)
    if controls and "pause_event" in controls:
        controls["pause_event"].clear()
        logger.info(f"[RESUME] Resume signal sent for process: {process_id}")
        return {"status": "resuming", "process_id": process_id}

    raise HTTPException(status_code=500, detail="Control events not found")
