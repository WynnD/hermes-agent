"""Per-platform last-active conversation tracker for cron delivery.

Stores the most recent conversation (platform, chat_id, thread_id) for each
messaging platform the user has interacted with.  Used by the ``deliver:
"last_active"`` cron option to resolve at fire time to wherever the user
most recently sent a message across all connected platforms.
"""

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

from hermes_constants import get_hermes_home


def _last_active_path() -> Path:
    """Return the path to the last-active JSON file."""
    return Path(get_hermes_home()) / "cron" / "last_active.json"


def _ensure_dir() -> None:
    """Ensure the parent directory exists."""
    _last_active_path().parent.mkdir(parents=True, exist_ok=True)


def _read_raw() -> Dict[str, dict]:
    """Read the raw last-active data from disk.

    Returns {} if the file doesn't exist or is corrupt.
    """
    path = _last_active_path()
    if not path.exists():
        return {}
    try:
        raw = path.read_text("utf-8")
        if not raw.strip():
            return {}
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {}
        return data
    except (json.JSONDecodeError, OSError, ValueError):
        return {}


def write_last_active(
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
) -> None:
    """Atomically update the last-active conversation for a platform.

    Called by the gateway on every real inbound human message.  Uses a
    temp file + atomic rename to prevent partial writes on crash.

    Args:
        platform: Platform name (e.g. ``"telegram"``, ``"discord"``).
        chat_id: Chat/channel ID for this platform.
        thread_id: Optional thread/topic ID within the chat.
    """
    _ensure_dir()
    data = _read_raw()
    data[platform] = {
        "chat_id": str(chat_id),
        "thread_id": thread_id,
        "timestamp": time.time(),
    }
    path = _last_active_path()
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            "utf-8",
        )
        tmp_path.replace(path)
    except OSError:
        # Best-effort: clean up temp file on error
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def read_last_active() -> Dict[str, dict]:
    """Read all last-active entries.

    Returns a dict mapping platform name to ``{chat_id, thread_id, timestamp}``.
    Returns ``{}`` if the file doesn't exist.
    """
    return _read_raw()


def get_most_recent_active() -> Optional[dict]:
    """Return the single most recently active conversation across all platforms.

    Returns a dict with keys ``platform``, ``chat_id``, ``thread_id``,
    ``timestamp``, or ``None`` if no data exists.

    Picks the entry with the highest ``timestamp`` value.
    """
    data = _read_raw()
    if not data:
        return None
    best: Optional[dict] = None
    best_ts: float = -1.0
    for platform, entry in data.items():
        ts = entry.get("timestamp", 0)
        if ts > best_ts:
            best = {
                "platform": platform,
                "chat_id": entry["chat_id"],
                "thread_id": entry.get("thread_id"),
                "timestamp": ts,
            }
            best_ts = ts
    return best


def clear_last_active() -> None:
    """Clear all last-active data.  Mainly for tests."""
    path = _last_active_path()
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
