from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict


PROJECT_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))
LOGS_DIR = PROJECT_ROOT / "logs"
_LOG_LOCK = Lock()


def ensure_logs_dir() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return LOGS_DIR


def get_log_file_path(timestamp: datetime | None = None) -> Path:
    current_time = timestamp or datetime.now()
    return ensure_logs_dir() / f"costmonitoring-{current_time.strftime('%Y%m%d')}.log"


def log_event(category: str, action: str, message: str, **details: Any) -> Path:
    payload: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "category": str(category).strip() or "general",
        "action": str(action).strip() or "info",
        "message": str(message).strip(),
    }
    if details:
        payload["details"] = details

    log_path = get_log_file_path()
    with _LOG_LOCK:
        with log_path.open("a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(payload, ensure_ascii=False, default=str))
            file_obj.write("\n")
    return log_path