from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner(adapter) -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="fake")},
    )
    runner.adapters = {Platform.DISCORD: adapter}
    runner._model = "test-model"
    runner._base_url = None
    runner._decide_image_input_mode = lambda: "text"
    return runner


@pytest.mark.asyncio
async def test_discord_auto_thread_renames_after_image_processing():
    """Any processed attachment should be able to rename a placeholder auto-thread."""
    adapter = SimpleNamespace(rename_auto_thread_from_attachment_processing=AsyncMock())
    runner = _make_runner(adapter)
    processed_text = (
        "[The user sent an image~ Here's what I can see:\n"
        "A screenshot of a router admin error page.]"
    )
    runner._enrich_message_with_vision = AsyncMock(return_value=processed_text)
    source = SessionSource(platform=Platform.DISCORD, chat_id="999", chat_type="thread")
    thread = SimpleNamespace(id=999, name="Image: router-error.png")
    event = MessageEvent(
        text="",
        message_type=MessageType.PHOTO,
        source=source,
        media_urls=["/tmp/router-error.png"],
        media_types=["image/png"],
    )
    setattr(event, "_discord_auto_threaded_attachment_channel", thread)

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result == processed_text
    adapter.rename_auto_thread_from_attachment_processing.assert_awaited_once_with(
        thread,
        processed_text,
    )
