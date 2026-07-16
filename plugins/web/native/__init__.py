"""Native local web extraction plugin — bundled, no API key required."""

from __future__ import annotations

from plugins.web.native.provider import NativeWebSearchProvider


def register(ctx) -> None:
    """Register the native extractor provider with the plugin context."""
    ctx.register_web_search_provider(NativeWebSearchProvider())
