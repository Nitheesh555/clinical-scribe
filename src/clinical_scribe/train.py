"""QLoRA SFT training (TRL SFTTrainer) with resume, early stopping, OOM guard.

Trains on the section-level MTS-Dialog JSONL produced by :mod:`clinical_scribe.data`.
Loss is computed only on the assistant span (the section text) via TRL's
``assistant_only_loss`` (auto-patches the Qwen3 chat template).

Heavy imports are local so the module imports on CPU/CI hosts.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any

from .config import Config
from .model import LoadedModel, load_model_and_tokenizer

logger = logging.getLogger(__name__)


def _resolve_report_to(config: Config) -> str:
    """Resolve experiment-tracking backend (W&B if key present, else TensorBoard)."""
    setting = config.tracking.report_to
    if setting != "auto":
        return setting
    if os.environ.get("WANDB_API_KEY"):
        logger.info("WANDB_API_KEY found -> reporting to Weights & Biases.")
        return "wandb"
    logger.info("No WANDB_API_KEY -> reporting to TensorBoard.")
    return "tensorboard"


def _load_jsonl_datasets(config: Config) -> tuple[Any, Any]:
    """Load train/val JSONL into HF datasets, keeping only the ``messages`` column.

    Runs the data pipeline first if the processed files are missing.
    """
    from datasets import load_dataset

    processed = Path(config.data.processed_dir)
    train_path = processed / "train.jsonl"
    val_path = processed / "val.jsonl"

    if not train_path.exists() or not val_path.exists():
        logger.info("Processed JSONL missing; running data preparation.")
        from .data import prepare_dataset

        prepare_dataset(config)

    ds = load_dataset(
        "json",
        data_files={"train": str(train_path), "validation": str(val_path)},
    )
    # SFTTrainer's conversational path needs only the messages column.
    keep = "messages"
    for split in ds:
        drop = [c for c in ds[split].column_names if c != keep]
        ds[split] = ds[split].remove_columns(drop)
    logger.info(
        "Loaded datasets: train=%d, val=%d",
        ds["train"].num_rows,
        ds["validation"].num_rows,
    )
    return ds["train"], ds["validation"]


def _build_sft_config(
    config: Config,
    loaded: LoadedModel,
    report_to: str,
    batch_size: int,
    max_length: int,
) -> Any:
    """Build an SFTConfig. ``batch_size``/``max_length`` are passed explicitly so
    the OOM guard can shrink them on retry."""
    from trl import SFTConfig

    t = config.train
    # Unsloth manages gradient checkpointing internally via get_peft_model.
    gc_in_trainer = t.gradient_checkpointing and loaded.backend != "unsloth"

    # packing + assistant_only_loss are mutually exclusive on trl>=0.24 with the
    # Unsloth compiled SFTTrainer: the packing path can't apply the chat template
    # to conversational `messages` (it raises "must specify a formatting_func"),
    # while assistant-only masking *requires* that path. Assistant-only loss wins
    # (we only want loss on the section text), so drop packing when both are set.
    packing = t.packing
    if packing and t.assistant_only_loss:
        logger.warning(
            "packing is incompatible with assistant_only_loss on this trl/Unsloth "
            "stack; disabling packing (assistant-only masking takes precedence)."
        )
        packing = False

    # Unsloth ≥2026.x auto-enables padding-free batching when packing=False, which
    # requires a formatting_func.  That path returns plain text so assistant_only_loss
    # (which needs the messages-dict path) cannot be honoured simultaneously.
    assistant_only_loss = t.assistant_only_loss
    if loaded.backend == "unsloth" and not packing and assistant_only_loss:
        logger.warning(
            "Unsloth padding-free batching (packing=False) requires a formatting_func "
            "which is incompatible with assistant_only_loss; disabling assistant_only_loss."
        )
        assistant_only_loss = False

    kwargs: dict[str, Any] = {
        "output_dir": config.general.output_dir,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": config.train.per_device_eval_batch_size,
        "gradient_accumulation_steps": t.gradient_accumulation_steps,
        "learning_rate": t.learning_rate,
        "weight_decay": t.weight_decay,
        "warmup_ratio": t.warmup_ratio,
        "lr_scheduler_type": t.lr_scheduler_type,
        "max_grad_norm": t.max_grad_norm,
        "optim": t.optim,
        "max_length": max_length,
        "packing": packing,
        "assistant_only_loss": assistant_only_loss,
        "gradient_checkpointing": gc_in_trainer,
        "logging_steps": t.logging_steps,
        "eval_strategy": "steps",
        "eval_steps": t.eval_steps,
        "save_strategy": "steps",
        "save_steps": t.save_steps,
        "save_total_limit": t.save_total_limit,
        "load_best_model_at_end": t.early_stopping_patience > 0,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": report_to,
        "run_name": config.general.run_name,
        "seed": config.general.seed,
        "fp16": loaded.runtime.dtype == "fp16",
        "bf16": loaded.runtime.dtype == "bf16",
    }
    if t.max_steps and t.max_steps > 0:
        kwargs["max_steps"] = t.max_steps
    else:
        kwargs["num_train_epochs"] = t.num_train_epochs
    if gc_in_trainer:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    return SFTConfig(**kwargs)


def _build_trainer(
    config: Config,
    loaded: LoadedModel,
    train_ds: Any,
    eval_ds: Any,
    report_to: str,
    batch_size: int,
    max_length: int,
) -> Any:
    """Assemble an SFTTrainer with early stopping."""
    from transformers import EarlyStoppingCallback
    from trl import SFTTrainer

    sft_config = _build_sft_config(config, loaded, report_to, batch_size, max_length)
    callbacks = []
    if config.train.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=config.train.early_stopping_patience)
        )

    # Unsloth ≥2026.x padding-free batching (active when packing=False) raises
    # "must specify a formatting_func" without this.  We apply the tokenizer's
    # chat template here so Unsloth receives pre-formatted text strings.
    formatting_func = None
    if loaded.backend == "unsloth" and not sft_config.packing:
        tok = loaded.tokenizer

        def formatting_func(examples: dict) -> list[str]:
            # Unsloth calls this twice with different shapes:
            #   validation: single example  → examples["messages"] = [{"role":…}, …]
            #   batched map: batch          → examples["messages"] = [[{…}, …], [{…}, …]]
            # Detect by checking whether the first element is a dict (single) or list (batch).
            msgs_field = examples["messages"]
            if msgs_field and isinstance(msgs_field[0], dict):
                return [tok.apply_chat_template(msgs_field, tokenize=False, add_generation_prompt=False)]
            return [
                tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                for msgs in msgs_field
            ]

    return SFTTrainer(
        model=loaded.model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=loaded.tokenizer,
        peft_config=loaded.peft_config,  # None for Unsloth (already wrapped)
        callbacks=callbacks,
        formatting_func=formatting_func,
    )


def _mirror_to_checkpoint_dir(config: Config) -> None:
    """Copy the final adapter to a durable dir (e.g. mounted Drive), if configured."""
    if not config.general.checkpoint_dir:
        return
    src = Path(config.general.output_dir) / "adapter"
    dst = Path(config.general.checkpoint_dir) / "adapter"
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        logger.info("Mirrored adapter -> %s", dst)


def run_training(config: Config, resume: bool = False) -> None:
    """Run Phase 1 QLoRA SFT with checkpointing, eval, and early stopping.

    Args:
        config: Validated run configuration.
        resume: Resume from the latest checkpoint in ``general.output_dir``.
    """
    import torch

    report_to = _resolve_report_to(config)
    loaded = load_model_and_tokenizer(config)
    train_ds, eval_ds = _load_jsonl_datasets(config)

    batch_size = config.train.per_device_train_batch_size
    max_length = config.model.max_seq_length
    attempts = config.train.oom_max_retries + 1 if config.train.auto_oom_guard else 1

    for attempt in range(attempts):
        try:
            trainer = _build_trainer(
                config, loaded, train_ds, eval_ds, report_to, batch_size, max_length
            )
            logger.info(
                "Starting training (attempt %d/%d): batch_size=%d, max_length=%d",
                attempt + 1,
                attempts,
                batch_size,
                max_length,
            )
            trainer.train(resume_from_checkpoint=resume)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if attempt + 1 >= attempts:
                logger.error("OOM persisted after %d attempts; giving up.", attempts)
                raise
            new_batch = max(1, batch_size // 2)
            new_len = max(256, int(max_length * 0.75))
            logger.warning(
                "OOM detected. Reducing batch_size %d->%d, max_length %d->%d and retrying.",
                batch_size,
                new_batch,
                max_length,
                new_len,
            )
            batch_size, max_length = new_batch, new_len
            # Resume from any checkpoint written before the OOM.
            resume = resume or any(Path(config.general.output_dir).glob("checkpoint-*"))

    adapter_dir = Path(config.general.output_dir) / "adapter"
    trainer.save_model(str(adapter_dir))
    loaded.tokenizer.save_pretrained(str(adapter_dir))
    logger.info("Saved adapter -> %s", adapter_dir)
    _mirror_to_checkpoint_dir(config)
