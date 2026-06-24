# Clinical Scribe — Project Overview

> A plain-language guide to what we're building, why, and how it's put together.
> Written to be read by a manager or stakeholder, not just engineers.

---

## TL;DR

We are fine-tuning a small open-source language model (**Qwen3-4B**) to turn a
messy doctor–patient **conversation** into **structured clinical-note text**
(History of Present Illness, Assessment, Plan, etc.).

**It is a documentation-drafting aid, not a diagnostic tool.** Every output
requires clinician review. We train **only** on de-identified, public research
data — no real patient data ever touches this system. The disclaimer appears in
the model's instructions, in every generated output, and in the model card.

---

## 1. What are we building, and why?

A doctor and patient talk. Today, someone has to *listen and type up* a
structured clinical note afterwards — slow, expensive, and a major cause of
clinician burnout.

Our system takes the **raw dialogue** and **drafts the structured note** for
them, so the clinician's job becomes a quick review-and-edit instead of typing
from scratch.

- **Purpose (why it matters):** reduce clinician documentation burden — turn a
  10-minute typing chore into a 30-second review.
- **Aim (what we're doing technically):** fine-tune a *small, specialized*
  open-source model so it is genuinely good at this one narrow task. A
  specialized small model is cheaper to run, can be self-hosted (important near
  healthcare data), and is more reliable on the exact format we need than a
  giant general-purpose model.

---

## 2. Datasets — what and from where

Two **public, de-identified research datasets**. No real patient data, ever.

| Dataset | Role | Source | License |
|---|---|---|---|
| **MTS-Dialog** | Training + validation + test (Phase 1). ~1,700 dialogues paired with their clinical-note sections. | Official GitHub repo: `github.com/abachaa/MTS-Dialog` | **CC BY 4.0** |
| **ACI-Bench** | Held-out **stress test only** — never used for training. Harder, full clinical notes. | Hugging Face Hub: `mkieffer/ACI-Bench` | Public research |

We use the dataset's **official splits** (1,201 train / 100 val / 200 + 200
test) and never let test rows leak into training — a data-integrity safeguard.

**Citations:**
- MTS-Dialog — Ben Abacha et al., *"An Empirical Study of Clinical Note
  Generation from Doctor-Patient Encounters,"* EACL 2023.
- ACI-Bench — Yim et al., *Nature Scientific Data*, 2023.

---

## 3. Which model, and why?

We use **Qwen3-4B-Instruct-2507**, an open-weights model from Alibaba.

- **"4B"** = 4 billion parameters — small enough to train on a cheap GPU, big
  enough to write coherent clinical prose.
- **"Instruct"** = already tuned to follow instructions; we refine behavior
  rather than teach language from scratch.
- **"2507" / non-thinking** = this variant does **not** emit a visible
  "reasoning out loud" stream. For us that is a feature — we want clean,
  deterministic note text, not the model musing about its logic.

We do **not** retrain the whole model. We use **QLoRA**: bolt a small trainable
"adapter" onto a frozen, 4-bit-compressed version of the model and train *only*
the adapter (a few million parameters instead of 4 billion). This is the
standard industry technique for fine-tuning on a budget.

### Why not DeepSeek?

DeepSeek is another respected open-source LLM family, strong on **reasoning**
(e.g. DeepSeek-R1). We didn't use it here because:

1. **Size fits our budget.** DeepSeek's strong models are very large (R1 is
   671B parameters — impossible on a cheap GPU). Qwen offers a clean, supported
   **4B** sized for a single consumer GPU.
2. **We don't want reasoning.** R1's selling point is "thinking out loud"; for
   note-drafting we want the opposite — clean, deterministic, formatted output.
   Qwen3-Instruct-2507 is explicitly a **non-thinking** variant.
3. **Tooling.** Qwen3 is first-class in the libraries we use (Unsloth, TRL,
   transformers) — fewer surprises on a fixed timeline.

DeepSeek isn't bad; it's the wrong tool for a small, deterministic,
single-GPU drafting task.

---

## 4. Local vs. Colab — who does what

**All the code is local; only the GPU work runs on Colab.** The reason is
simple: the development laptop has **no NVIDIA GPU**, and training a neural
network without one is not feasible (a CPU attempt died loading the model).
This split is standard industry practice, not a workaround.

| | **Local (VS Code / laptop)** | **Colab (cloud GPU)** |
|---|---|---|
| Entire Python package (`src/clinical_scribe/`) | written & lives here | pulled from GitHub |
| Config files, tests | here | pulled |
| Code editing, linting, unit tests | here | — |
| Git / version control (source of truth) | here → GitHub | clones from GitHub |
| Data download + cleaning | code here; can run here (CPU OK) | runs here in practice |
| Tokenization | code here | **runs here** |
| Training (QLoRA) | code here, *cannot run* (no GPU) | **runs here** |
| Evaluation | pure-metric tests run here; full eval needs GPU | **runs here** |
| Inference / generation | code here, can't run | **runs here** |

**The discipline:**
1. Write and test all logic locally as a proper, version-controlled package.
2. Keep the Colab notebook **deliberately thin** — it only clones the repo,
   installs deps, and calls our package. **No real logic lives in the notebook.**
3. Push code to GitHub; the GPU machine pulls it and runs it.

> *"The laptop is the workshop where we build and test the tool; Colab is the
> rented machine with the heavy equipment we run it on."*

### Why "Colab Pro" came up (it wasn't in the original plan)

The plan didn't change — our understanding of the free tier's limits got
sharper.

- Phase 1 was scoped for a **free Colab T4 GPU**, which is technically capable.
- The friction is **practical, not a design flaw**: free Colab disconnects
  after idle/time limits and doesn't guarantee a GPU, so a multi-hour training
  run can be cut off mid-way. (We built crash-resilient checkpointing to Google
  Drive to survive this — but it's still painful.)
- The laptop can't be a fallback (no GPU). Kaggle's free tier (30 GPU-hrs/week)
  was the other option being evaluated.

**Colab Pro (~$10/month)** buys priority GPU access and longer sessions — i.e.
reliability, not new capability.

> *"Phase 1 was scoped for a free GPU and still works on one. We're considering
> ~$10 of Colab Pro purely to avoid free-tier disconnects — convenience, not a
> blocker."*

---

## 5. Pipeline workflow

```
                         ┌─────────────────────────────────────────────┐
                         │   LOCAL  (VS Code)  —  the code & tests       │
                         │   src/clinical_scribe/  +  configs/*.yaml     │
                         └───────────────────────┬─────────────────────┘
                                                 │  git push
                                                 ▼
                                        ┌──────────────────┐
                                        │   GitHub (repo)   │  ← single source of truth
                                        └────────┬─────────┘
                                                 │  git clone / pull
                                                 ▼
   ┌──────────────────────────── COLAB  (GPU) ───────────────────────────────┐
   │                                                                          │
   │  ① DATA          data.py                                                 │
   │     MTS-Dialog CSV ──► clean ──► chat-format JSONL  (train/val/test)     │
   │                                   │                                      │
   │  ② TOKENIZE      train.py (TRL)   ▼                                      │
   │     text ──► token IDs ──► packed batches                                │
   │                                   │                                      │
   │  ③ TRAIN         train.py + model.py                                     │
   │     Qwen3-4B (frozen, 4-bit)  +  LoRA adapter (trainable)                │
   │     QLoRA SFT ──────────────► outputs/adapter/   (+ mirror to Drive)     │
   │                                   │                                      │
   │  ④ EVALUATE      eval.py          ▼                                      │
   │     base  vs  base+adapter:  ROUGE-L · BERTScore · structure · faithful  │
   │     ──► eval_report.md  ──►  ✅/❌ SUCCESS GATE  ── STOP for approval     │
   │                                   │                                      │
   │  ⑤ EXPORT (todo) export.py        ▼                                      │
   │     merge LoRA ──► push to HF Hub + model card ──► GGUF                   │
   │                                   │                                      │
   │  ⑥ SERVE  (todo) serve.py         ▼                                      │
   │     vLLM OpenAI-compatible API  ◄── inference for real use               │
   └──────────────────────────────────────────────────────────────────────────┘
```

The disclaimer ("draft, needs clinician review") rides along the **entire**
pipeline — in the prompt, in every output, and in the model card.

---

## 6. Where each stage is written

**Every stage is written locally** in the package. Colab only *runs* it.

| Stage | File (in `src/clinical_scribe/`) | Written | Runs |
|---|---|---|---|
| Load dataset | `data.py` | local | Colab (or local) |
| Tokenization | inside `train.py` (via TRL trainer) | local | Colab |
| Training | `train.py` + `model.py` | local | **Colab only** |
| Evaluation | `eval.py` | local | Colab (pure metrics tested locally) |
| Inference | `eval.py` (`_generate`) now; `serve.py` later | local | **Colab only** |

> *"There is no logic in Colab. 100% of the code is in our version-controlled,
> tested package. The notebook is a 16-cell launcher — clone, install, call."*

---

## 7. Is it "just Python scripts," or real engineering?

It is a properly structured, installable Python **package** built on
industry-standard frameworks — not loose scripts.

**Structure:**
- `src/clinical_scribe/` — 11 modules, each with one job (`data`, `model`,
  `train`, `eval`, `export`, `serve`, `config`, `utils`, `cli`).
- `tests/` — **26 passing unit tests** (pytest).
- `configs/` — all settings in validated YAML; **zero hardcoded magic numbers**.
- `scripts/` — command-line entry points.

**Frameworks & tools:**

| Tool | Role |
|---|---|
| **PyTorch** | Deep-learning engine |
| **Hugging Face Transformers** | Loads the Qwen3 model |
| **TRL** | Supervised fine-tuning trainer |
| **PEFT** + **bitsandbytes** | QLoRA (adapter + 4-bit) machinery |
| **Unsloth** | Training speed/memory optimization |
| **Pydantic** | Validates every config value at load time |
| **datasets / rouge-score / bert-score** | Data handling and evaluation |
| **vLLM** | Serving (later) |

**Engineering discipline:**
- Type hints + docstrings throughout; **mypy** type-checking clean.
- **ruff** + **black** linting/formatting; **pre-commit** hooks.
- **GitHub Actions CI** runs tests on every push.
- Fixed random seeds (reproducibility), structured logging (no stray prints),
  Apache-2.0 license, README, Makefile, `.env.example`.

> *"It's a tested, type-checked, CI-backed Python package built on the standard
> fine-tuning stack — Transformers, TRL, PEFT, PyTorch — not a pile of notebook
> cells."*

---

## 8. Current status

| Module | Status |
|---|---|
| `config`, `utils`, `data` | ✅ done & verified |
| `model`, `train`, notebook | 🟡 built, GPU-unverified |
| `eval`, `export`, `serve` | ✅ built & tested (GPU run pending) |

- **39 tests passing**, ruff + mypy clean. **All Phase 1 modules are built.**
- **Next milestone:** run training on a real GPU (Colab) and check the Phase 1
  **success gate** (is the fine-tuned model measurably better than the base
  model on ROUGE-L, BERTScore, structure validity, and faithfulness?).
- **Phase 2** (bigger model, full SOAP notes) is proposed **only after** the
  Phase 1 gate passes and is approved.
