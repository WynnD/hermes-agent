"""Gateway-level regression: the Discord attachment rename hook must fire.

Bug history: the May 27 implementation wired a post-processing rename through
``_prepare_inbound_message_text`` via an event attribute, but the hook was
lost in a later update — voice/image/document attachment-first threads kept
their cheap placeholder titles (or the useless ``Hermes`` fallback) forever.

These tests pin the re-anchored wiring (now in ``gateway/run_inbound.py``):
after STT/vision/document enrichment changes the inbound text,
``_prepare_inbound_message_text`` must call the Discord adapter's
``rename_auto_thread_from_attachment_processing`` with the auto-created
thread ref carried on the event and the processed text.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner(discord_adapter) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="fake")},
    )
    runner.adapters = {Platform.DISCORD: discord_adapter}
    return runner


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-1",
        chat_name="Voice message from TestUser",
        chat_type="thread",
        user_name="TestUser",
    )


def _event(**kwargs) -> MessageEvent:
    return MessageEvent(
        text=kwargs.get("text", ""),
        message_type=kwargs.get("message_type", MessageType.VOICE),
        source=kwargs.get("source", _source()),
        media_urls=kwargs.get("media_urls", ["/tmp/voice-message.ogg"]),
        media_types=kwargs.get("media_types", ["audio/ogg"]),
        message_id=kwargs.get("message_id", None),
    )


@pytest.mark.asyncio
async def test_stt_rename_hook_fires_for_voice_thread(monkeypatch):
    """After STT enrichment, the voice thread must be renamed from the transcript."""
    thread = SimpleNamespace(
        id=999,
        name="Voice message from TestUser",
        edit=AsyncMock(),
    )
    discord_adapter = SimpleNamespace(
        rename_auto_thread_from_attachment_processing=AsyncMock(return_value=True),
    )
    runner = _make_runner(discord_adapter)
    source = _source()
    event = _event()
    event._discord_auto_threaded_attachment_channel = thread

    async def _fake_voice_enrich(_event, _source, message_text, audio_paths):
        return '"Can we fix the thread title bug?"'

    monkeypatch.setattr(runner, "_enrich_inbound_voice", _fake_voice_enrich)

    await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
        session_key="test-session",
    )

    discord_adapter.rename_auto_thread_from_attachment_processing.assert_awaited_once()
    args = discord_adapter.rename_auto_thread_from_attachment_processing.await_args
    assert args.args[0] is thread
    assert '"Can we fix the thread title bug?"' in args.args[1]
    assert args.kwargs.get("reason") == "Hermes auto-thread title from voice transcript"


@pytest.mark.asyncio
async def test_rename_hook_is_silent_without_thread_ref(monkeypatch):
    """No auto-thread ref on the event → the hook must not call the adapter."""
    discord_adapter = SimpleNamespace(
        rename_auto_thread_from_attachment_processing=AsyncMock(return_value=True),
    )
    runner = _make_runner(discord_adapter)
    source = _source()
    event = _event()  # no _discord_auto_threaded_attachment_channel

    async def _fake_voice_enrich(_event, _source, message_text, audio_paths):
        return '"Some transcript"'

    monkeypatch.setattr(runner, "_enrich_inbound_voice", _fake_voice_enrich)

    await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
        session_key="test-session",
    )

    discord_adapter.rename_auto_thread_from_attachment_processing.assert_not_awaited()
