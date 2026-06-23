"""Model + tokenizer loading with runtime-adaptive precision/attention.

Loads Qwen3-4B-Instruct-2507 in 4-bit (QLoRA). Prefers Unsloth (fastest, lowest
VRAM on a T4); falls back to transformers + peft + bitsandbytes when Unsloth is
unavailable or does not support the model on the installed version.

All heavy imports (torch/transformers/unsloth/peft) are local to functions so
this module imports cleanly on CPU-only / CI hosts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .config import Config
from .utils import ResolvedRuntime, detect_gpu, resolve_runtime

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """Result of loading: the model, tokenizer, backend, and resolved runtime.

    For the ``unsloth`` backend the model is already LoRA-wrapped, so
    ``peft_config`` is None. For the ``transformers`` backend the base 4-bit
    model is returned and ``peft_config`` carries the LoRA config to hand to
    ``SFTTrainer(peft_config=...)``.
    """

    model: Any
    tokenizer: Any
    backend: str
    runtime: ResolvedRuntime
    peft_config: Any | None = None


def _torch_dtype(dtype: str) -> Any:
    """Map a resolved dtype string to a torch dtype object."""
    import torch

    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]


def build_lora_config(config: Config) -> Any:
    """Construct a ``peft.LoraConfig`` from the validated config (transformers path)."""
    from peft import LoraConfig

    lc = config.lora
    return LoraConfig(
        r=lc.r,
        lora_alpha=lc.alpha,
        lora_dropout=lc.dropout,
        bias=lc.bias,
        target_modules=lc.target_modules,
        task_type="CAUSAL_LM",
    )


def _load_unsloth(config: Config, runtime: ResolvedRuntime) -> LoadedModel:
    """Load via Unsloth FastLanguageModel and apply LoRA. Raises if unavailable."""
    from unsloth import FastLanguageModel  # type: ignore[import-not-found]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model.base_model_id,
        max_seq_length=config.model.max_seq_length,
        load_in_4bit=config.model.load_in_4bit,
        dtype=_torch_dtype(runtime.dtype),
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias=config.lora.bias,
        target_modules=config.lora.target_modules,
        use_gradient_checkpointing="unsloth" if config.train.gradient_checkpointing else False,
        random_state=config.general.seed,
        max_seq_length=config.model.max_seq_length,
    )
    logger.info("Loaded model via Unsloth backend.")
    return LoadedModel(model, tokenizer, "unsloth", runtime, peft_config=None)


def _load_transformers(config: Config, runtime: ResolvedRuntime) -> LoadedModel:
    """Load via transformers + bitsandbytes 4-bit; LoRA applied by the trainer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    compute_dtype = _torch_dtype(runtime.dtype)
    quant_config = None
    if config.model.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.model.base_model_id,
        quantization_config=quant_config,
        dtype=compute_dtype,
        attn_implementation=runtime.attn_implementation,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if config.model.load_in_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.train.gradient_checkpointing
        )

    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loaded model via transformers+peft backend.")
    return LoadedModel(
        model, tokenizer, "transformers", runtime, peft_config=build_lora_config(config)
    )


def load_model_and_tokenizer(config: Config) -> LoadedModel:
    """Load the 4-bit base model + tokenizer, adapted to the detected GPU.

    Tries Unsloth first when ``config.model.use_unsloth`` is set, falling back to
    transformers+peft on any import/compatibility failure.
    """
    gpu = detect_gpu()
    runtime = resolve_runtime(config.model.dtype, config.model.attn_implementation, gpu=gpu)

    if config.model.use_unsloth:
        try:
            return _load_unsloth(config, runtime)
        except Exception as exc:  # noqa: BLE001 — fall back on any Unsloth failure
            logger.warning(
                "Unsloth unavailable/incompatible (%s); falling back to transformers.",
                exc,
            )

    return _load_transformers(config, runtime)
