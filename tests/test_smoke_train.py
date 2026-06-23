"""2-step smoke-train test. Skipped unless a CUDA GPU and the train stack exist.

Run on Colab/GPU with:  pytest -m "slow and gpu" -s
This guards against trainer/config wiring regressions end-to-end at tiny scale.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


_gpu_available = False
if _has("torch"):
    import torch

    _gpu_available = torch.cuda.is_available()

requires_train_stack = pytest.mark.skipif(
    not (_gpu_available and _has("transformers") and _has("trl")),
    reason="requires CUDA GPU + transformers + trl",
)


@pytest.mark.gpu
@pytest.mark.slow
@requires_train_stack
def test_smoke_train_two_steps(tmp_path: Path) -> None:
    """Run 2 optimizer steps and assert a checkpoint is produced."""
    from clinical_scribe.config import load_config
    from clinical_scribe.train import run_training

    config = load_config(CONFIGS_DIR / "phase1_smoke.yaml")
    config.general.output_dir = str(tmp_path / "smoke")

    run_training(config, resume=False)

    checkpoints = list((tmp_path / "smoke").glob("checkpoint-*"))
    assert checkpoints, "smoke train produced no checkpoint"
