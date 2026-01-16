"""Debug endpoints for development testing."""
import logging
from fastapi import APIRouter
from ..deps import sessions, analyses, processes

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/debug", tags=["debug"])


@router.get("/sessions")
async def list_sessions():
    """List all active sessions with their IDs."""
    result = []
    for session_id, session in sessions.items():
        result.append({
            "session_id": session_id,
            "video_path": session.get("video_path"),
            "video_info": session.get("video_info"),
            "has_reference": session.get("reference_embedding") is not None,
            "analysis_id": session.get("analysis_id")
        })
    return {"sessions": result, "count": len(result)}


@router.get("/analyses")
async def list_analyses():
    """List all analyses with their IDs and status."""
    result = []
    for analysis_id, analysis in analyses.items():
        status = analysis.get("status")
        status_value = status.value if hasattr(status, 'value') else str(status)

        result.append({
            "analysis_id": analysis_id,
            "session_id": analysis.get("session_id"),
            "status": status_value,
            "progress": analysis.get("progress", 0),
            "faces_detected": analysis.get("faces_detected", 0),
            "has_result": analysis.get("result") is not None,
            "has_clusters": len(analysis.get("clusters", [])) > 0
        })
    return {"analyses": result, "count": len(result)}


@router.get("/processes")
async def list_processes():
    """List all processes with their IDs and status."""
    result = []
    for process_id, process in processes.items():
        status = process.get("status")
        status_value = status.value if hasattr(status, 'value') else str(status)

        result.append({
            "process_id": process_id,
            "analysis_id": process.get("analysis_id"),
            "status": status_value,
            "progress": process.get("progress", 0),
            "output_path": process.get("output_path"),
            "error": process.get("error")
        })
    return {"processes": result, "count": len(result)}


@router.get("/state")
async def get_full_state():
    """Get complete state for debugging."""
    return {
        "sessions_count": len(sessions),
        "analyses_count": len(analyses),
        "processes_count": len(processes),
        "sessions": await list_sessions(),
        "analyses": await list_analyses(),
        "processes": await list_processes()
    }
