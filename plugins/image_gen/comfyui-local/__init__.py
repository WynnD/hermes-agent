"""Local ComfyUI image generation (Krea 2 Turbo on snowball).

Shells out to the verified ``~/.hermes/scripts/gen_image.py`` (ComfyUI /prompt API
on 192.168.0.101:8188) and returns the saved PNG. Selection:
``image_gen.provider: comfyui-local`` in config.yaml.
"""
from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO, ImageGenProvider, resolve_aspect_ratio, success_response)
from plugins.image_gen._common import error_factory

logger = logging.getLogger(__name__)

GEN_SCRIPT = "/home/wynn/.hermes/scripts/gen_image.py"
COMFY_HOST = "http://192.168.0.101:8188"

_ASPECT_TO_SIZE = {
    "square": (1024, 1024),
    "landscape": (1344, 768),
    "portrait": (768, 1344),
    "wide": (1536, 640),
}

_MODELS: Dict[str, Dict[str, Any]] = {
    "krea2-turbo": {
        "display": "Krea 2 Turbo (local ComfyUI)",
        "speed": "~25s",
        "strengths": "Local Krea 2 Turbo FP8 via ComfyUI on snowball; free, private, on-LAN",
        "price": "$0.00",
    },
}
DEFAULT_MODEL = "krea2-turbo"


class ComfyUILocalImageGenProvider(ImageGenProvider):
    """ComfyUI /prompt backend served from the LAN snowball box."""

    provider_id = "comfyui-local"
    label = "Local ComfyUI (Krea 2 Turbo)"
    models = _MODELS
    default_model_id = DEFAULT_MODEL

    @property
    def name(self) -> str:
        return self.provider_id
    setup = dict(
        name="Local ComfyUI", badge="local", tag="Krea 2 Turbo on snowball :8188",
        key=None, prompt=None, url=COMFY_HOST)

    def is_available(self) -> bool:
        try:
            import urllib.request
            with urllib.request.urlopen(f"{COMFY_HOST}/system_stats", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False

    def capabilities(self) -> Dict[str, Any]:
        return {"modalities": ["text"], "max_reference_images": 0}

    def _resolve_model(self, caller_model: Optional[str]) -> Tuple[str, Dict[str, Any]]:
        model_id = caller_model or DEFAULT_MODEL
        return model_id, _MODELS.get(model_id, _MODELS[DEFAULT_MODEL])

    def generate(
        self, prompt: str, aspect_ratio: str = DEFAULT_ASPECT_RATIO, *,
        image_url: Optional[str] = None, reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        if not prompt:
            return error_factory("comfyui-local", aspect)(
                "A text prompt is required for comfyui-local.", "invalid_request")
        model_id, _meta = self._resolve_model(kwargs.get("model"))
        w, h = _ASPECT_TO_SIZE.get(aspect, _ASPECT_TO_SIZE["square"])
        fail = error_factory("comfyui-local", aspect, model=model_id, prompt=prompt)

        if not self.is_available():
            return fail(
                "ComfyUI on snowball (:8188) is not reachable. Check `systemctl status comfyui` on snowball.",
                "service_unavailable")

        out = Path(tempfile.mkstemp(prefix="krea2-", suffix=".png")[1])
        try:
            proc = subprocess.run(
                ["python3", GEN_SCRIPT, prompt, str(out), str(w), str(h)],
                capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            return fail("Krea 2 generation timed out after 300s.", "timeout")
        if proc.returncode != 0 or not out.exists():
            detail = (proc.stderr or proc.stdout or "").strip()[-400:]
            return fail(f"gen_image.py failed: {detail}", "api_error")
        try:
            info = json.loads(proc.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            info = {}
        extra: Dict[str, Any] = {"size": f"{w}x{h}", "comfy_seed": info.get("seed")}
        return success_response(
            image=str(out), model=model_id, prompt=prompt, aspect_ratio=aspect,
            provider="comfyui-local", modality="text", extra=extra)


def register(ctx) -> None:
    """Plugin entry point -- wire ComfyUILocalImageGenProvider into the registry."""
    ctx.register_image_gen_provider(ComfyUILocalImageGenProvider())
