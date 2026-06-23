"""Serving helpers: vLLM serve command + minimal OpenAI-compatible client. Next step."""

from __future__ import annotations

import logging

from .config import Config

logger = logging.getLogger(__name__)


def build_vllm_command(config: Config, model_path: str) -> list[str]:
    """Return the argv for an OpenAI-compatible vLLM server for ``model_path``."""
    raise NotImplementedError("Implemented in the export/serve step.")
