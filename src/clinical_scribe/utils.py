"""Cross-cutting utilities: seeding, logging, GPU detection, version capture.

Kept import-light so it works on CPU-only CI hosts (torch is imported lazily).
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

logger = logging.getLogger(__name__)

# Packages whose versions we log for reproducibility.
_TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "trl",
    "peft",
    "bitsandbytes",
    "datasets",
    "accelerate",
    "unsloth",
    "vllm",
    "rouge_score",
    "bert_score",
    "evaluate",
    "pydantic",
)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once with a concise, timestamped format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seed(seed: int) -> None:
    """Seed ``random``, ``numpy``, and ``torch`` (incl. CUDA) for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        logger.debug("numpy not available; skipping numpy seed")
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        logger.debug("torch not available; skipping torch seed")
    logger.info("Seeded RNGs with seed=%d", seed)


def get_versions() -> dict[str, str]:
    """Return a mapping of tracked package -> installed version (or 'not installed')."""
    versions: dict[str, str] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "not installed"
    return versions


def log_versions() -> dict[str, str]:
    """Log and return tracked package versions."""
    versions = get_versions()
    logger.info("Library versions:")
    for pkg, ver in versions.items():
        logger.info("  %-14s %s", pkg, ver)
    return versions


@dataclass
class GpuInfo:
    """Detected GPU capabilities used to adapt precision/attention."""

    available: bool = False
    name: str = "cpu"
    total_memory_gb: float = 0.0
    compute_capability: tuple[int, int] = (0, 0)
    supports_bf16: bool = False
    supports_flash_attention_2: bool = False


def detect_gpu() -> GpuInfo:
    """Detect the active CUDA GPU and its bf16 / FlashAttention-2 support.

    T4 (compute capability 7.5) -> no bf16, no FA2. Ampere+ (>= 8.0) -> bf16,
    and FA2 if the ``flash_attn`` package is importable.
    """
    try:
        import torch
    except ImportError:
        logger.warning("torch not installed; assuming CPU.")
        return GpuInfo()

    if not torch.cuda.is_available():
        logger.warning("No CUDA GPU detected; running on CPU.")
        return GpuInfo()

    props = torch.cuda.get_device_properties(0)
    cc = (props.major, props.minor)
    supports_bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    has_fa2_pkg = _is_importable("flash_attn")
    supports_fa2 = supports_bf16 and cc[0] >= 8 and has_fa2_pkg

    info = GpuInfo(
        available=True,
        name=props.name,
        total_memory_gb=round(props.total_memory / (1024**3), 2),
        compute_capability=cc,
        supports_bf16=supports_bf16,
        supports_flash_attention_2=supports_fa2,
    )
    logger.info(
        "GPU: %s | %.1f GB | cc=%d.%d | bf16=%s | flash_attn2=%s",
        info.name,
        info.total_memory_gb,
        cc[0],
        cc[1],
        info.supports_bf16,
        info.supports_flash_attention_2,
    )
    return info


@dataclass
class ResolvedRuntime:
    """Concrete precision/attention chosen for this host."""

    dtype: str
    attn_implementation: str
    notes: list[str] = field(default_factory=list)


def resolve_runtime(
    requested_dtype: str,
    requested_attn: str,
    gpu: GpuInfo | None = None,
) -> ResolvedRuntime:
    """Resolve ``auto`` precision/attention against detected hardware.

    Args:
        requested_dtype: One of ``auto|fp16|bf16|fp32``.
        requested_attn: One of ``auto|sdpa|flash_attention_2|eager``.
        gpu: Detected GPU info (auto-detected if None).

    Returns:
        Concrete :class:`ResolvedRuntime` with human-readable ``notes``.
    """
    gpu = gpu or detect_gpu()
    notes: list[str] = []

    if requested_dtype == "auto":
        dtype = "bf16" if gpu.supports_bf16 else ("fp16" if gpu.available else "fp32")
        notes.append(f"dtype=auto -> {dtype}")
    else:
        dtype = requested_dtype
        if dtype == "bf16" and not gpu.supports_bf16:
            dtype = "fp16"
            notes.append("requested bf16 but unsupported (e.g. T4) -> fp16")

    if requested_attn == "auto":
        attn = "flash_attention_2" if gpu.supports_flash_attention_2 else "sdpa"
        notes.append(f"attn=auto -> {attn}")
    else:
        attn = requested_attn
        if attn == "flash_attention_2" and not gpu.supports_flash_attention_2:
            attn = "sdpa"
            notes.append("requested flash_attention_2 but unsupported -> sdpa")

    for note in notes:
        logger.info("Runtime: %s", note)
    return ResolvedRuntime(dtype=dtype, attn_implementation=attn, notes=notes)


def get_secret(name: str, required: bool = False) -> str | None:
    """Read a secret from the environment (Colab secrets export to env).

    Never logs the value. Raises if ``required`` and missing.
    """
    value = os.environ.get(name)
    if value:
        logger.info("Secret %s: present", name)
        return value
    if required:
        raise RuntimeError(f"Required secret {name} is not set in the environment.")
    logger.info("Secret %s: absent", name)
    return None


def _is_importable(module_name: str) -> bool:
    """Return True if ``module_name`` can be imported without side effects."""
    try:
        import_module(module_name)
        return True
    except ImportError:
        return False
