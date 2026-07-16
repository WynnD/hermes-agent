"""Tests for the Mnemosyne memory provider plugin."""

import json

from plugins.memory.mnemosyne import MnemosyneMemoryProvider


def test_mnemosyne_provider_stores_and_searches(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("MNEMOSYNE_DB_PATH", str(tmp_path / "mnemosyne.db"))
    monkeypatch.setenv("MNEMOSYNE_USER_ID", "wynn")
    monkeypatch.setenv("MNEMOSYNE_AGENT_ID", "zhu-li")

    provider = MnemosyneMemoryProvider()
    assert provider.is_available()

    provider.initialize("session-1", platform="discord")
    stored = json.loads(
        provider.handle_tool_call(
            "mnemosyne_conclude",
            {"conclusion": "Wynn prefers concise Discord replies", "importance": 0.9},
        )
    )
    assert stored["result"] == "Fact stored."
    assert stored["id"]

    result = json.loads(
        provider.handle_tool_call(
            "mnemosyne_search",
            {"query": "communication preference", "top_k": 5},
        )
    )
    assert result["count"] >= 1
    assert any("concise Discord" in item["memory"] for item in result["results"])
    provider.shutdown()


def test_mnemosyne_on_memory_write_mirrors_builtin_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("MNEMOSYNE_DB_PATH", str(tmp_path / "mnemosyne.db"))

    provider = MnemosyneMemoryProvider()
    provider.initialize("session-1", platform="cli")
    provider.on_memory_write("add", "user", "Wynn likes local-first memory systems")

    result = json.loads(
        provider.handle_tool_call(
            "mnemosyne_search",
            {"query": "local first memory", "top_k": 5},
        )
    )
    assert result["count"] >= 1
    assert any("local-first memory" in item["memory"] for item in result["results"])
    provider.shutdown()
