import os
import shutil
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import hashlib

logger = logging.getLogger(__name__)


class DataRetentionManager:
    def __init__(self, base_path: Path, retention_hours: int = 24):
        self.base_path = Path(base_path)
        self.retention_hours = retention_hours
        self.base_path.mkdir(parents=True, exist_ok=True)

    def cleanup_expired(self) -> int:
        """Delete expired files and return count of deleted items."""
        if not self.base_path.exists():
            return 0

        cutoff = datetime.now() - timedelta(hours=self.retention_hours)
        deleted_count = 0

        for item in self.base_path.iterdir():
            try:
                if item.stat().st_mtime < cutoff.timestamp():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                    deleted_count += 1
                    logger.info(f"Deleted expired item: {item.name}")
            except Exception as e:
                logger.error(f"Error deleting {item}: {e}")

        return deleted_count

    def delete_session_data(self, session_id: str) -> bool:
        """Delete session data immediately."""
        session_path = self.base_path / session_id
        if session_path.exists():
            try:
                shutil.rmtree(session_path)
                self.log_deletion("session", session_id)
                return True
            except Exception as e:
                logger.error(f"Error deleting session {session_id}: {e}")
                return False
        return False

    def log_deletion(self, item_type: str, item_id: str) -> dict:
        """Log deletion without storing PII."""
        log_entry = {
            "event": "data_deletion",
            "type": item_type,
            "id_hash": hashlib.sha256(item_id.encode()).hexdigest()[:16],
            "timestamp": datetime.now().isoformat()
        }
        logger.info(f"Data deletion logged: {log_entry}")
        return log_entry


def generate_session_id() -> str:
    """Generate a unique session ID."""
    import uuid
    return str(uuid.uuid4())


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks."""
    # Remove path separators and null bytes
    sanitized = os.path.basename(filename)
    sanitized = sanitized.replace('\x00', '')

    # Keep only alphanumeric, dots, hyphens, underscores
    safe_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_')
    sanitized = ''.join(c if c in safe_chars else '_' for c in sanitized)

    return sanitized or 'unnamed'
