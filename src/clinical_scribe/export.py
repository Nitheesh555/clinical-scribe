"""Export: merge LoRA, push adapters + merged model + model card, GGUF. Next step."""

from __future__ import annotations

import logging

from .config import Config

logger = logging.getLogger(__name__)


def run_export(config: Config, adapter_path: str) -> None:
    """Merge the adapter, push to the Hub with a model card, and export GGUF.

    Args:
        config: Validated run configuration.
        adapter_path: Path to the trained LoRA adapter to export.
    """
    raise NotImplementedError("Implemented in the export step.")
