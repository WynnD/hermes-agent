"""Mnemosyne memory plugin — local SQLite MemoryProvider interface.

Config via $HERMES_HOME/mnemosyne.json or environment variables:
  MNEMOSYNE_DB_PATH   — SQLite path (default: $HERMES_HOME/mnemosyne.db)
  MNEMOSYNE_USER_ID   — canonical user/author id (default: hermes-user)
  MNEMOSYNE_AGENT_ID  — agent/source id (default: hermes)
  MNEMOSYNE_BANK      — optional memory bank name

Mnemosyne is local-first: no API key, no hosted service, no network dependency.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from hermes_constants import get_hermes_home
from tools.registry import tool_error

logger = logging.getLogger(__name__)

PROFILE_SCHEMA = {
    "name": "mnemosyne_profile",
    "description": (
        "Retrieve all stored Mnemosyne memories for the active user/profile. "
        "Use when you need a broad memory overview."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

SEARCH_SCHEMA = {
    "name": "mnemosyne_search",
    "description": (
        "Search local Mnemosyne memory by meaning and keywords. Returns ranked facts."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "top_k": {"type": "integer", "description": "Max results (default 10, max 50)."},
        },
        "required": ["query"],
    },
}

CONCLUDE_SCHEMA = {
    "name": "mnemosyne_conclude",
    "description": (
        "Store a durable curated fact in local Mnemosyne memory. Use for explicit "
        "preferences, corrections, stable environment facts, and decisions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "conclusion": {"type": "string", "description": "The fact to store."},
            "importance": {
                "type": "number",
                "description": "Importance from 0.0 to 1.0 (default 0.8).",
            },
            "scope": {
                "type": "string",
                "enum": ["global", "session"],
                "description": "global for cross-session facts, session for local context (default global).",
            },
        },
        "required": ["conclusion"],
    },
}


def _load_config() -> dict:
    hermes_home = get_hermes_home()
    cfg = {
        "db_path": os.environ.get("MNEMOSYNE_DB_PATH", str(hermes_home / "mnemosyne.db")),
        "user_id": os.environ.get("MNEMOSYNE_USER_ID", "hermes-user"),
        "agent_id": os.environ.get("MNEMOSYNE_AGENT_ID", "hermes"),
        "bank": os.environ.get("MNEMOSYNE_BANK", ""),
        "sync_turns": False,
    }
    config_path = hermes_home / "mnemosyne.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            cfg.update({k: v for k, v in file_cfg.items() if v is not None and v != ""})
        except Exception as exc:
            logger.debug("Failed to read mnemosyne.json: %s", exc)
    return cfg


def _memory_content(row: Dict[str, Any]) -> str:
    return str(row.get("content") or row.get("memory") or row.get("text") or "")


class MnemosyneMemoryProvider(MemoryProvider):
    """Hermes MemoryProvider backed by Mnemosyne's local SQLite store."""

    def __init__(self) -> None:
        self._config: dict = {}
        self._memory = None
        self._memory_lock = threading.Lock()
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None
        self._session_id = ""
        self._db_path = ""
        self._user_id = "hermes-user"
        self._agent_id = "hermes"
        self._platform = ""
        self._bank = ""
        self._sync_turns = False

    @property
    def name(self) -> str:
        return "mnemosyne"

    def is_available(self) -> bool:
        try:
            import mnemosyne  # noqa: F401
            return True
        except Exception:
            return False

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "db_path", "description": "SQLite database path", "default": str(get_hermes_home() / "mnemosyne.db")},
            {"key": "user_id", "description": "Canonical user/author identifier", "default": "hermes-user"},
            {"key": "agent_id", "description": "Agent/source identifier", "default": "hermes"},
            {"key": "bank", "description": "Optional Mnemosyne memory bank", "default": ""},
            {"key": "sync_turns", "description": "Automatically store completed turns verbatim", "default": "false", "choices": ["true", "false"]},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        config_path = Path(hermes_home) / "mnemosyne.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._session_id = session_id or "default"
        self._db_path = str(Path(self._config.get("db_path") or get_hermes_home() / "mnemosyne.db").expanduser())
        self._user_id = self._config.get("user_id") or kwargs.get("user_id") or "hermes-user"
        self._agent_id = self._config.get("agent_id") or kwargs.get("agent_identity") or "hermes"
        self._platform = kwargs.get("platform") or ""
        self._bank = self._config.get("bank") or None
        self._sync_turns = str(self._config.get("sync_turns", False)).lower() in {"1", "true", "yes", "on"}
        Path(self._db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._get_memory()

    def _get_memory(self):
        with self._memory_lock:
            if self._memory is None:
                from mnemosyne import Mnemosyne
                self._memory = Mnemosyne(
                    session_id=self._session_id or "default",
                    db_path=Path(self._db_path).expanduser(),
                    bank=self._bank,
                    author_id=self._user_id,
                    author_type="user",
                    channel_id=self._platform or None,
                )
            return self._memory

    @staticmethod
    def _format_results(rows: List[Dict[str, Any]], max_items: int = 10) -> str:
        lines = []
        for row in rows[:max_items]:
            content = _memory_content(row).strip()
            if not content:
                continue
            score = row.get("score")
            source = row.get("source")
            suffix = []
            if isinstance(score, (int, float)):
                suffix.append(f"score={score:.3f}")
            if source:
                suffix.append(f"source={source}")
            meta = f" ({', '.join(suffix)})" if suffix else ""
            lines.append(f"- {content}{meta}")
        return "\n".join(lines)

    def system_prompt_block(self) -> str:
        return (
            "# Mnemosyne Memory\n"
            f"Active. Local SQLite DB: {self._db_path}. User: {self._user_id}.\n"
            "Use mnemosyne_search to find memories, mnemosyne_conclude to store curated facts, "
            "mnemosyne_profile for a full overview."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=2.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## Mnemosyne Memory\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        def _run() -> None:
            try:
                rows = self._get_memory().recall(query=query or "recent context", top_k=5)
                text = self._format_results(rows, max_items=5)
                if text:
                    with self._prefetch_lock:
                        self._prefetch_result = text
            except Exception as exc:
                logger.debug("Mnemosyne prefetch failed: %s", exc)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="mnemosyne-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Optionally store whole turns. Off by default to avoid memory sludge."""
        if not self._sync_turns:
            return

        def _sync() -> None:
            try:
                content = f"User: {user_content}\nAssistant: {assistant_content}"
                self._get_memory().remember(
                    content,
                    source="conversation_turn",
                    importance=0.35,
                    scope="session",
                    metadata={"session_id": session_id or self._session_id, "agent_id": self._agent_id},
                )
            except Exception as exc:
                logger.warning("Mnemosyne sync failed: %s", exc)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=3.0)
        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="mnemosyne-sync")
        self._sync_thread.start()

    def on_memory_write(self, action: str, target: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        if action not in {"add", "replace"} or not content:
            return
        try:
            importance = 0.9 if target == "user" else 0.75
            self._get_memory().remember(
                content,
                source=f"builtin_{target}",
                importance=importance,
                scope="global",
                metadata=metadata or {},
            )
        except Exception as exc:
            logger.debug("Mnemosyne built-in memory mirror failed: %s", exc)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [PROFILE_SCHEMA, SEARCH_SCHEMA, CONCLUDE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        try:
            memory = self._get_memory()
        except Exception as exc:
            return tool_error(str(exc))

        if tool_name == "mnemosyne_profile":
            try:
                rows = memory.get_all_memories()
                rows = [r for r in rows if _memory_content(r).strip()]
                return json.dumps({"result": self._format_results(rows, max_items=50) or "No memories stored yet.", "count": len(rows)})
            except Exception as exc:
                return tool_error(f"Failed to fetch profile: {exc}")

        if tool_name == "mnemosyne_search":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            top_k = min(max(int(args.get("top_k", 10)), 1), 50)
            try:
                rows = memory.recall(query=query, top_k=top_k)
                items = [
                    {
                        "memory": _memory_content(r),
                        "score": r.get("score"),
                        "source": r.get("source"),
                        "scope": r.get("scope"),
                    }
                    for r in rows
                    if _memory_content(r).strip()
                ]
                return json.dumps({"results": items, "count": len(items)} if items else {"result": "No relevant memories found."})
            except Exception as exc:
                return tool_error(f"Search failed: {exc}")

        if tool_name == "mnemosyne_conclude":
            conclusion = args.get("conclusion", "")
            if not conclusion:
                return tool_error("Missing required parameter: conclusion")
            importance = float(args.get("importance", 0.8))
            importance = max(0.0, min(1.0, importance))
            scope = args.get("scope") or "global"
            if scope not in {"global", "session"}:
                scope = "global"
            try:
                memory_id = memory.remember(
                    conclusion,
                    source="curated_fact",
                    importance=importance,
                    scope=scope,
                    metadata={"agent_id": self._agent_id},
                )
                return json.dumps({"result": "Fact stored.", "id": memory_id})
            except Exception as exc:
                return tool_error(f"Failed to store: {exc}")

        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        for thread in (self._prefetch_thread, self._sync_thread):
            if thread and thread.is_alive():
                thread.join(timeout=5.0)


def register(ctx) -> None:
    """Register Mnemosyne as a memory provider plugin."""
    ctx.register_memory_provider(MnemosyneMemoryProvider())
