"""Tests for ``cron.last_active`` — per-platform last-active conversation tracker.

Tests cover atomic write/read, multi-platform tracking, timestamp ordering,
and the scheduler delivery resolution for ``deliver: "last_active"``.
"""

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cron.last_active import (
    _last_active_path,
    clear_last_active,
    get_most_recent_active,
    read_last_active,
    write_last_active,
)


@pytest.fixture
def hermes_home():
    """Return the isolated Hermes home set by the root conftest's autouse fixture."""
    from hermes_constants import get_hermes_home

    return get_hermes_home()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _data_path(hermes_home: Path) -> Path:
    return hermes_home / "cron" / "last_active.json"


def _write_raw(hermes_home: Path, data: dict) -> None:
    """Write raw JSON data to the last-active file (for test setup)."""
    path = _data_path(hermes_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Tests for the last_active module
# ---------------------------------------------------------------------------


class TestWriteAndRead:
    """Test the write / read / clear cycle."""

    def test_write_and_read_roundtrip(self, hermes_home: Path):
        """Write one platform, read it back correctly."""
        write_last_active("telegram", "-1001234567890", "42")
        data = read_last_active()
        assert "telegram" in data
        assert data["telegram"]["chat_id"] == "-1001234567890"
        assert data["telegram"]["thread_id"] == "42"
        assert "timestamp" in data["telegram"]

    def test_write_multiple_platforms(self, hermes_home: Path):
        """Write discord + telegram, read both back."""
        write_last_active("discord", "987654321")
        write_last_active("telegram", "-1001234567890", "42")
        data = read_last_active()
        assert set(data.keys()) == {"discord", "telegram"}
        assert data["discord"]["chat_id"] == "987654321"
        assert data["telegram"]["chat_id"] == "-1001234567890"
        assert data["telegram"]["thread_id"] == "42"

    def test_overwrite_same_platform(self, hermes_home: Path):
        """Writing twice for same platform updates the entry."""
        write_last_active("telegram", "-100111", "1")
        write_last_active("telegram", "-100222", "2")
        data = read_last_active()
        assert data["telegram"]["chat_id"] == "-100222"
        assert data["telegram"]["thread_id"] == "2"

    def test_clear_last_active(self, hermes_home: Path):
        """Write then clear → read returns {}."""
        write_last_active("telegram", "-100123")
        assert read_last_active() != {}
        clear_last_active()
        assert read_last_active() == {}

    def test_file_does_not_exist_returns_empty(self, hermes_home: Path):
        """No file on disk → read_last_active returns {}."""
        path = _data_path(hermes_home)
        if path.exists():
            path.unlink()
        assert read_last_active() == {}

    def test_corrupt_file_returns_empty(self, hermes_home: Path):
        """Corrupt JSON on disk → read_last_active returns {}."""
        path = _data_path(hermes_home)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("this is not json")
        assert read_last_active() == {}

    def test_empty_file_returns_empty(self, hermes_home: Path):
        """Empty file on disk → read_last_active returns {}."""
        path = _data_path(hermes_home)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        assert read_last_active() == {}


class TestGetMostRecent:
    """Test the ``get_most_recent_active()`` function."""

    def test_get_most_recent_single(self, hermes_home: Path):
        """One platform → returns it."""
        write_last_active("telegram", "-100123", "42")
        entry = get_most_recent_active()
        assert entry is not None
        assert entry["platform"] == "telegram"
        assert entry["chat_id"] == "-100123"
        assert entry["thread_id"] == "42"

    def test_get_recent_picks_latest(self, hermes_home: Path):
        """Multiple platforms → returns the one with the highest timestamp."""
        now = time.time()
        _write_raw(
            hermes_home,
            {
                "discord": {
                    "chat_id": "111",
                    "thread_id": None,
                    "timestamp": now - 300,  # 5 min ago
                },
                "telegram": {
                    "chat_id": "222",
                    "thread_id": "17",
                    "timestamp": now - 60,  # 1 min ago (newer)
                },
            },
        )
        entry = get_most_recent_active()
        assert entry is not None
        assert entry["platform"] == "telegram"
        assert entry["chat_id"] == "222"
        assert entry["thread_id"] == "17"

    def test_get_recent_empty(self, hermes_home: Path):
        """No data → returns None."""
        clear_last_active()
        assert get_most_recent_active() is None


class TestThreadIdHandling:
    """Test thread-id preservation and optionality."""

    def test_thread_id_preserved(self, hermes_home: Path):
        """thread_id is stored and returned correctly."""
        write_last_active("discord", "111", "thread-abc")
        entry = get_most_recent_active()
        assert entry is not None
        assert entry["thread_id"] == "thread-abc"

    def test_thread_id_none(self, hermes_home: Path):
        """Messages without thread_id work (None stored)."""
        write_last_active("telegram", "222")
        data = read_last_active()
        assert data["telegram"]["thread_id"] is None

    def test_thread_id_runtime_none(self, hermes_home: Path):
        """get_most_recent_active returns thread_id=None when stored as None."""
        write_last_active("slack", "C123")
        entry = get_most_recent_active()
        assert entry is not None
        assert entry["thread_id"] is None


class TestAtomicWrite:
    """Test atomic-write guarantees."""

    def test_atomic_write(self, hermes_home: Path):
        """File is never partially written — temp file approach works."""
        path = _data_path(hermes_home)
        write_last_active("telegram", "111")
        original = path.read_text()
        # Simulate a crash during write by leaving a temp file
        tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        tmp.write_text("corrupt")
        # Re-read: the real file should still be intact
        data = read_last_active()
        assert data["telegram"]["chat_id"] == "111"
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Tests for scheduler integration
# ---------------------------------------------------------------------------


class TestSchedulerDelivery:
    """Test that ``_resolve_single_delivery_target`` handles ``last_active``."""

    def _make_job(self, **overrides) -> dict:
        job = {
            "id": "test-job",
            "name": "Test Job",
            "prompt": "do something",
            "schedule": "30m",
            "deliver": "last_active",
        }
        job.update(overrides)
        return job

    def test_resolve_last_active_with_data(self, hermes_home: Path):
        """Last-active data exists → resolves to that target."""
        from cron.scheduler import _resolve_single_delivery_target

        write_last_active("telegram", "-100123", "42")
        job = self._make_job()
        result = _resolve_single_delivery_target(job, "last_active")
        assert result is not None
        assert result["platform"] == "telegram"
        assert result["chat_id"] == "-100123"
        assert result["thread_id"] == "42"

    def test_resolve_last_active_no_data_falls_back_to_none(self, hermes_home: Path):
        """No last-active data, no home channels → returns None."""
        from cron.scheduler import _resolve_single_delivery_target

        clear_last_active()
        job = self._make_job()
        result = _resolve_single_delivery_target(job, "last_active")
        # Without home-channel env vars set, falls through to bare platform
        # resolution (line 872+) which returns None for unknown platforms.
        assert result is None

    def test_resolve_last_active_picks_most_recent(self, hermes_home: Path):
        """Multiple platforms → most recent is chosen."""
        from cron.scheduler import _resolve_single_delivery_target

        now = time.time()
        _write_raw(
            hermes_home,
            {
                "discord": {
                    "chat_id": "111",
                    "thread_id": None,
                    "timestamp": now - 300,
                },
                "telegram": {
                    "chat_id": "222",
                    "thread_id": "17",
                    "timestamp": now - 60,
                },
            },
        )
        job = self._make_job()
        result = _resolve_single_delivery_target(job, "last_active")
        assert result is not None
        assert result["platform"] == "telegram"
        assert result["chat_id"] == "222"
        assert result["thread_id"] == "17"

    def test_resolve_last_active_returns_dict(self, hermes_home: Path):
        """Ensures return type matches the contract expected by _resolve_delivery_targets."""
        from cron.scheduler import _resolve_single_delivery_target

        write_last_active("telegram", "123")
        job = self._make_job()
        result = _resolve_single_delivery_target(job, "last_active")
        assert isinstance(result, dict)
        assert "platform" in result
        assert "chat_id" in result
        assert "thread_id" in result
