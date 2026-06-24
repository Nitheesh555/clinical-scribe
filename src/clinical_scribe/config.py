"""Pydantic-validated configuration schema and loader.

All tunable values live in ``configs/*.yaml`` and are validated here. Code must
never hardcode magic numbers; it reads them from a validated :class:`Config`.

A config file may declare ``extends: <relative-path>`` to inherit from a base
config; the loader deep-merges the child over the parent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

Dtype = Literal["auto", "fp16", "bf16", "fp32"]
AttnImpl = Literal["auto", "sdpa", "flash_attention_2", "eager"]
Scheduler = Literal["linear", "cosine", "constant", "constant_with_warmup"]


class GeneralConfig(BaseModel):
    """Run-wide settings."""

    seed: int = 42
    run_name: str = "phase1-qwen3-4b"
    output_dir: str = "outputs"
    # If set (e.g. a mounted Drive path), checkpoints are mirrored here so a
    # Colab disconnect does not lose progress.
    checkpoint_dir: str | None = None


class ModelConfig(BaseModel):
    """Base model + precision/attention selection.

    ``dtype`` and ``attn_implementation`` default to ``auto``; the runtime
    resolves them from the detected GPU (T4 -> fp16 + sdpa; Ampere+ -> bf16 +
    flash_attention_2 if available). See :func:`clinical_scribe.utils.resolve_runtime`.
    """

    base_model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    dtype: Dtype = "auto"
    attn_implementation: AttnImpl = "auto"
    use_unsloth: bool = True
    # Qwen3-*-Instruct-2507 is non-thinking only; kept explicit for the chat
    # template / future hybrid models.
    enable_thinking: bool = False


class LoRAConfig(BaseModel):
    """QLoRA adapter hyperparameters."""

    r: int = Field(16, ge=1)
    alpha: int = Field(32, ge=1)
    dropout: float = Field(0.0, ge=0.0, le=1.0)
    bias: Literal["none", "all", "lora_only"] = "none"
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


class DataConfig(BaseModel):
    """Dataset sourcing, cleaning, and chat-template materialization."""

    # Canonical source: official MTS-Dialog GitHub CSVs (CC BY 4.0).
    mts_dialog_repo: str = "https://raw.githubusercontent.com/abachaa/MTS-Dialog/main"
    train_csv: str = "Main-Dataset/MTS-Dialog-TrainingSet.csv"
    val_csv: str = "Main-Dataset/MTS-Dialog-ValidationSet.csv"
    test_csv_1: str = "Main-Dataset/MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv"
    test_csv_2: str = "Main-Dataset/MTS-Dialog-TestSet-2-MEDIQA-Sum-2023.csv"

    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"

    # Expected columns in the official CSVs (schema-validated at load time).
    id_column: str = "ID"
    section_header_column: str = "section_header"
    section_text_column: str = "section_text"
    dialogue_column: str = "dialogue"

    # Token-length guardrails (logged; rows over the limit are flagged/optionally
    # dropped rather than silently truncated).
    max_prompt_tokens: int = 1536
    warn_truncation: bool = True
    drop_over_max: bool = False

    na_placeholder: str = "N/A"


class TrainConfig(BaseModel):
    """SFT trainer hyperparameters."""

    num_train_epochs: float = 3.0
    # ``max_steps`` > 0 overrides epochs (used by the smoke config).
    max_steps: int = -1
    per_device_train_batch_size: int = Field(2, ge=1)
    per_device_eval_batch_size: int = Field(2, ge=1)
    gradient_accumulation_steps: int = Field(8, ge=1)
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = Field(0.05, ge=0.0, le=1.0)
    lr_scheduler_type: Scheduler = "cosine"
    max_grad_norm: float = 1.0
    optim: str = "adamw_8bit"  # bitsandbytes 8-bit optimizer (low VRAM on T4)
    gradient_checkpointing: bool = True
    packing: bool = True
    # Compute loss only on assistant spans (the section text); TRL auto-patches
    # the Qwen3 chat template for this.
    assistant_only_loss: bool = True
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50
    save_total_limit: int = 3
    early_stopping_patience: int = Field(3, ge=0)
    # Auto-shrink batch/seq on OOM and retry, logging each reduction.
    auto_oom_guard: bool = True
    oom_max_retries: int = Field(3, ge=0)


class EvalConfig(BaseModel):
    """Evaluation metrics and faithfulness checks."""

    compute_rouge: bool = True
    compute_bertscore: bool = True
    bertscore_model: str = "microsoft/deberta-xlarge-mnli"
    bertscore_lang: str = "en"
    check_structure_validity: bool = True
    # Faithfulness: clinical-entity overlap; optional LLM-as-judge.
    faithfulness_entity_overlap: bool = True
    use_llm_judge: bool = False
    judge_model_id: str | None = None
    num_qualitative_samples: int = 10
    max_new_tokens: int = 512
    batch_size: int = Field(8, ge=1)
    # MTS-Dialog test JSONL files (relative to data.processed_dir) to evaluate.
    test_files: list[str] = Field(default_factory=lambda: ["test1.jsonl", "test2.jsonl"])
    report_path: str = "outputs/eval_report.md"
    # Success-gate thresholds (Phase 1).
    gate_min_structure_validity: float = 0.95
    # Held-out stress test on ACI-Bench (expected weaker; reported honestly).
    aci_bench_stress_test: bool = True
    aci_bench_dataset_id: str = "mkieffer/ACI-Bench"
    aci_bench_splits: list[str] = Field(default_factory=lambda: ["test1", "test2", "test3"])


class ExportConfig(BaseModel):
    """LoRA merge, Hub push, and GGUF export."""

    merge_adapter: bool = True
    # Provided by the user when we reach the export step (None = local only).
    hub_repo_id: str | None = None
    hub_private: bool = True
    push_adapter: bool = True
    push_merged: bool = True
    gguf_quant_types: list[str] = Field(default_factory=lambda: ["q4_k_m", "q8_0"])


class TrackingConfig(BaseModel):
    """Experiment tracking. Falls back to TensorBoard when no W&B key is set."""

    report_to: Literal["auto", "wandb", "tensorboard", "none"] = "auto"
    wandb_project: str = "clinical-scribe"
    wandb_entity: str | None = None


class PromptConfig(BaseModel):
    """System prompt + disclaimer text injected into every example/output."""

    disclaimer: str = (
        "DRAFT — AI-generated documentation aid. Requires clinician review. "
        "Not a diagnostic tool. Verify all content against the source encounter."
    )
    system_template: str = (
        "You are a clinical documentation assistant that drafts structured "
        "clinical note text from a doctor-patient dialogue.\n"
        "{disclaimer}\n"
        "Rules:\n"
        "- Use only information present in the dialogue; do not invent facts.\n"
        "- Write in concise clinical register.\n"
        '- If the requested section has no supporting information, output exactly "{na}".\n'
        "- Output only the section text, with no preamble or commentary."
    )
    # Section-level user instruction (Phase 1). {section} is the human-readable
    # section name resolved from the MTS-Dialog code.
    user_template: str = (
        "Dialogue:\n{dialogue}\n\n" 'Write the "{section}" section of the clinical note.'
    )

    def render_system(self, na_placeholder: str) -> str:
        """Return the system prompt with disclaimer and N/A token interpolated."""
        return self.system_template.format(disclaimer=self.disclaimer, na=na_placeholder)


class Config(BaseModel):
    """Root configuration aggregating all sub-configs."""

    general: GeneralConfig = Field(default_factory=GeneralConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    prompts: PromptConfig = Field(default_factory=PromptConfig)

    @model_validator(mode="after")
    def _check_seq_lengths(self) -> Config:
        if self.data.max_prompt_tokens >= self.model.max_seq_length:
            raise ValueError(
                f"data.max_prompt_tokens ({self.data.max_prompt_tokens}) must be "
                f"< model.max_seq_length ({self.model.max_seq_length}) to leave "
                "room for the target completion."
            )
        return self


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_yaml_with_extends(path: Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Load a YAML file, resolving a single-level chain of ``extends`` parents."""
    _seen = _seen or set()
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"Circular 'extends' detected at {path}")
    _seen.add(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    parent_rel = raw.pop("extends", None)
    if parent_rel is None:
        return raw

    parent_path = (path.parent / parent_rel).resolve()
    parent = _load_yaml_with_extends(parent_path, _seen)
    return _deep_merge(parent, raw)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load, merge (``extends``), apply overrides, and validate a config.

    Args:
        path: Path to a YAML config file.
        overrides: Optional nested dict merged last (e.g. CLI ``--set`` values).

    Returns:
        A validated :class:`Config`.

    Raises:
        FileNotFoundError: If the file (or an ``extends`` parent) is missing.
        pydantic.ValidationError: If values fail validation.
    """
    merged = _load_yaml_with_extends(Path(path))
    if overrides:
        merged = _deep_merge(merged, overrides)
    config = Config.model_validate(merged)
    logger.debug("Loaded config from %s", path)
    return config
