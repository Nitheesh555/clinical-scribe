"""Evaluation: ROUGE-L, BERTScore, structure validity, faithfulness, base-vs-FT.

Implemented in the eval step. Produces ``eval_report.md`` and runs the
ACI-Bench held-out stress test.
"""

from __future__ import annotations

import logging

from .config import Config

logger = logging.getLogger(__name__)


def run_evaluation(config: Config, adapter_path: str | None = None) -> None:
    """Evaluate base vs fine-tuned model and write the eval report.

    Args:
        config: Validated run configuration.
        adapter_path: Path to a trained LoRA adapter (None = base model only).
    """
    raise NotImplementedError("Implemented in the eval step.")
