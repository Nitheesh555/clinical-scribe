# Clinical Scribe

Fine-tune an open LLM (Qwen3) to turn messy doctor-patient dialogue into
structured clinical note text.

> ⚠️ **This is a documentation-drafting aid, not a diagnostic tool.** Every
> generated output **requires clinician review**. Training uses **only
> de-identified public benchmark data**. Disclaimers appear in the system
> prompt, in generated outputs, and in the model card.

## Status

| Phase | Target env | Output | State |
|-------|-----------|--------|-------|
| **Phase 1** | Free Colab T4 (16 GB) | Section-level note text (MTS-Dialog style) | 🚧 In progress |
| **Phase 2** | Colab Pro/Pro+ (L4/A100) | Full SOAP note (Subjective/Objective/Assessment/Plan) | ⏳ After Phase 1 gate |

Built so far: repo scaffold, pydantic config system, and the MTS-Dialog data
pipeline (verified end-to-end on the live dataset). Training, eval, and export
are wired and being implemented incrementally.

## Architecture

```
src/clinical_scribe/
  config.py   # pydantic-validated schema + YAML loader (extends/merge). NO magic numbers in code.
  utils.py    # seeding, logging, GPU detection, runtime precision/attn resolution, version capture
  data.py     # MTS-Dialog: download -> schema-validate -> clean -> chat template -> JSONL (+ token stats)
  model.py    # 4-bit Qwen3 + LoRA load, GPU-adaptive (Unsloth -> transformers fallback)
  train.py    # QLoRA SFT (TRL SFTTrainer): checkpointing, resume, eval, early stopping, OOM guard
  eval.py     # ROUGE-L, BERTScore, structure validity, faithfulness, base-vs-FT, ACI-Bench stress test
  export.py   # merge LoRA, push adapter+merged+model card to HF Hub, GGUF (q4_k_m/q8_0)
  serve.py    # vLLM serve command + OpenAI-compatible client example
  cli.py      # argparse entrypoints (logic lives in the modules)
configs/        # base.yaml + phase1_t4.yaml + phase1_smoke.yaml (extends-merged, validated)
scripts/        # thin CLI wrappers
tests/          # pytest (config, data, 2-step smoke-train)
notebooks/      # one thin Colab notebook: install + mount Drive + load config + call src/
```

**Model:** `Qwen/Qwen3-4B-Instruct-2507` — the *non-thinking-only* Qwen3 4B
snapshot. It never emits `<think>` blocks, which is exactly what we want for
deterministic structured output (no `/no_think` suppression needed).

## Quickstart (local, CPU — data + tests)

```bash
python -m pip install -e ".[dev]"      # core + tooling
make test                              # lint-excluded pytest (CPU only)
python scripts/prepare_data.py --config configs/phase1_t4.yaml --stats-out outputs/data_stats.json
```

`prepare_data.py` downloads the official MTS-Dialog CSVs, validates the schema,
and writes `data/processed/{train,val,test1,test2}.jsonl` with logged
token-length distributions.

## Quickstart (Colab T4 — training)

A single thin notebook lives in `notebooks/` (added in the training step). It:
1. installs deps (Unsloth `colab-new` extra; torch is preinstalled on Colab),
2. mounts Google Drive for crash-resilient checkpoints,
3. loads `configs/phase1_t4.yaml`,
4. calls `src/` to prepare data, train, evaluate, and export.

Secrets (`HF_TOKEN`, `WANDB_API_KEY`) come from **Colab secrets** or env — never
hardcoded. See `.env.example`.

## Configuration

All tunables live in `configs/*.yaml`, validated by `clinical_scribe.config`.
A config may `extends: <file>` to inherit and override. Key knobs:

| Section | Keys |
|---------|------|
| `model` | `base_model_id`, `max_seq_length`, `load_in_4bit`, `dtype` (auto/fp16/bf16), `attn_implementation` (auto/sdpa/flash_attention_2), `use_unsloth` |
| `lora` | `r`, `alpha`, `dropout`, `target_modules`, `bias` |
| `train` | epochs/`max_steps`, batch sizes, `gradient_accumulation_steps`, `learning_rate`, `lr_scheduler_type`, `packing`, `gradient_checkpointing`, `early_stopping_patience` |
| `data` | CSV sources, column names, `max_prompt_tokens`, truncation policy |
| `eval` | metric toggles, `bertscore_model`, faithfulness, `aci_bench_stress_test` |
| `export` | `hub_repo_id`, `gguf_quant_types`, merge/push flags |

`dtype: auto` and `attn_implementation: auto` resolve at runtime from the
detected GPU (T4 → fp16 + SDPA; Ampere+ → bf16 + FlashAttention-2 if available).

## Reproducibility

- All RNGs (`random`, `numpy`, `torch`/CUDA) seeded from `general.seed`.
- Resolved config + all library versions are logged at the start of every run.
- `make lock` (run on the target Colab runtime) freezes resolved versions to
  `requirements.lock` — the source of truth for reproducing a run, since the
  Unsloth/TRL/transformers stack moves weekly.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `KeyError: 'qwen3'` | `transformers < 4.51` — upgrade. |
| **OOM on T4** | Lower `train.per_device_train_batch_size` (raise `gradient_accumulation_steps` to keep effective batch), or lower `model.max_seq_length`. `auto_oom_guard` reduces these automatically and logs it. |
| **Colab disconnect** | Set `general.checkpoint_dir` to a mounted Drive path; rerun with `--resume`. |
| **HF auth / gated model** | Set `HF_TOKEN` as a Colab secret (read scope to pull, write to push at export). Never printed/committed. |
| **bf16/FlashAttention errors on T4** | Expected — T4 (cc 7.5) has neither. Keep `dtype: auto`/`attn: auto` (or `fp16`/`sdpa`); the runtime forces these on T4. |
| **Symlink warning on Windows** | Harmless HF cache notice; set `HF_HUB_DISABLE_SYMLINKS_WARNING=1`. |

## Data & licenses

- **MTS-Dialog** — 1,201 train / 100 val / 200×2 test; short dialogue + a single
  section (header + content). **License: CC BY 4.0.** Ben Abacha et al., *An
  Empirical Study of Clinical Note Generation from Doctor-Patient Encounters*,
  EACL 2023.
- **ACI-Bench** — small but full multi-section notes; **held-out eval in Phase
  1**, a training source in Phase 2. Yim et al., *ACI-Bench…*, Nature Scientific
  Data 2023.
- **Qwen3-4B-Instruct-2507** — Apache-2.0 (Qwen team, Alibaba Cloud).

Source code: Apache-2.0 (see `LICENSE`).
