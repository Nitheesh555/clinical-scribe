"""Export: merge LoRA, write a clinical model card, push to the Hub, GGUF.

The model card builder (:func:`build_model_card`) is a pure, CPU-testable
function. The merge / Hub-push / GGUF steps load the model and shell out to
external tools, so their heavy imports are local to the functions and they are
only exercised on a GPU host.

Disclaimers ride along the whole pipeline: the card leads with the same
clinician-review banner that is injected into every prompt and output
(``config.prompts.disclaimer``).
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from .config import Config
from .utils import get_secret

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Pure, CPU-testable: model card                                              #
# --------------------------------------------------------------------------- #
def _read_eval_report(report_path: str | Path) -> str | None:
    """Return the eval report markdown if it exists, else None."""
    path = Path(report_path)
    if not path.exists():
        logger.info("No eval report at %s; card will omit metrics.", path)
        return None
    return path.read_text(encoding="utf-8")


def _lora_summary(config: Config) -> str:
    """One-line LoRA hyperparameter summary for the card."""
    lc = config.lora
    return (
        f"QLoRA (4-bit NF4, double-quant); r={lc.r}, alpha={lc.alpha}, "
        f"dropout={lc.dropout}, target_modules={', '.join(lc.target_modules)}"
    )


def build_model_card(
    config: Config,
    *,
    repo_id: str,
    is_adapter: bool,
    eval_report: str | None = None,
) -> str:
    """Render a Hugging Face model card (README.md) for the export.

    Args:
        config: Validated run configuration.
        repo_id: Target Hub repo id (used in the title only).
        is_adapter: True for the standalone LoRA adapter card; False for the
            merged full model.
        eval_report: Optional contents of ``eval_report.md`` to embed verbatim.

    Returns:
        Markdown text including YAML frontmatter required by the Hub.
    """
    disclaimer = config.prompts.disclaimer
    base = config.model.base_model_id
    kind = "LoRA adapter" if is_adapter else "merged model"

    frontmatter_lines = [
        "---",
        f"license: {config.export.model_license}",
        f"base_model: {base}",
        "library_name: peft" if is_adapter else "library_name: transformers",
        "tags:",
        "  - clinical-nlp",
        "  - qlora",
        "  - qwen3",
        "  - documentation-aid",
        "  - not-for-diagnosis",
        "datasets:",
        "  - mts-dialog",
        "language:",
        "  - en",
        "pipeline_tag: text-generation",
        "---",
    ]

    eval_section = (
        f"\n## Evaluation\n\n{eval_report.strip()}\n"
        if eval_report
        else "\n## Evaluation\n\nSee `eval_report.md` in the training run "
        "(base vs. fine-tuned ROUGE-L, BERTScore, structure validity, "
        "faithfulness). Not yet attached.\n"
    )

    body = f"""# Clinical Scribe — {kind} (`{repo_id}`)

> ⚠️ **DRAFT — documentation aid, not a diagnostic tool.**
> {disclaimer}

## Intended use

Drafts **section-level clinical-note text** (e.g. History of Present Illness,
Assessment, Plan) from a de-identified doctor–patient dialogue, to reduce
clinician documentation burden. **Every output requires clinician review and
sign-off before use.**

### Out of scope

- **Not** a diagnostic, triage, or clinical-decision tool.
- **Not** for use on identifiable patient data without appropriate governance.
- **Not** validated for autonomous (un-reviewed) deployment.

## Model details

- **Base model:** `{base}`
- **Adaptation:** {_lora_summary(config)}
- **Format:** {"LoRA adapter weights — load on top of the base model with PEFT."
  if is_adapter else "Merged full-precision weights (adapter folded into base)."}
- **Thinking mode:** disabled (deterministic, formatted output).

## Training data

- **MTS-Dialog** (CC BY 4.0) — Ben Abacha et al., *"An Empirical Study of
  Clinical Note Generation from Doctor-Patient Encounters,"* EACL 2023.
- Official splits; test rows are never mixed into training.
- **ACI-Bench** (Yim et al., *Nature Scientific Data*, 2023) is used **only**
  as a held-out stress test, never for training.

Training used **only de-identified, public benchmark data**. No real patient
data was used.
{eval_section}
## Limitations & risks

- May omit or hallucinate clinical details; outputs are unverified drafts.
- Trained on a small benchmark; performance on real-world dialogue and unseen
  section types is not guaranteed.
- The faithfulness metric is a clinical-term-overlap proxy, not medical NER or
  clinical judgement.

## How to use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
{"from peft import PeftModel" if is_adapter else ""}

tok = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForCausalLM.from_pretrained("{repo_id if not is_adapter else base}")
{f'model = PeftModel.from_pretrained(model, "{repo_id}")' if is_adapter else ""}
```

## License

{config.export.model_license}. Derived from `{base}` and MTS-Dialog (CC BY 4.0);
review upstream license terms before redistribution.
"""
    return "\n".join(frontmatter_lines) + "\n\n" + body


# --------------------------------------------------------------------------- #
# Heavy (GPU host): merge, push, GGUF                                         #
# --------------------------------------------------------------------------- #
def merge_adapter(config: Config, adapter_path: str, output_dir: Path) -> Path:
    """Fold the LoRA adapter into full-precision base weights and save.

    The base is loaded in full precision (not 4-bit): merging into a quantized
    model is lossy/unsupported. Returns the directory of the merged model.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .utils import resolve_runtime

    runtime = resolve_runtime(config.model.dtype, config.model.attn_implementation)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[runtime.dtype]

    logger.info("Loading base %s in %s for merge.", config.model.base_model_id, runtime.dtype)
    base = AutoModelForCausalLM.from_pretrained(
        config.model.base_model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    logger.info("Merging adapter from %s ...", adapter_path)
    model = model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), safe_serialization=True)

    # The tokenizer was saved alongside the adapter during training.
    tok_source = adapter_path if (Path(adapter_path) / "tokenizer_config.json").exists() else None
    tokenizer = AutoTokenizer.from_pretrained(tok_source or config.model.base_model_id)
    tokenizer.save_pretrained(str(output_dir))

    logger.info("Saved merged model -> %s", output_dir)
    return output_dir


def push_folder_to_hub(repo_id: str, folder: Path, *, private: bool, token: str) -> None:
    """Create the repo (if needed) and upload a folder to the Hub."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True, repo_type="model")
    logger.info("Uploading %s -> hf.co/%s (private=%s)", folder, repo_id, private)
    api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type="model")
    logger.info("Pushed %s", repo_id)


def export_gguf(config: Config, source_dir: Path, output_dir: Path) -> list[Path]:
    """Convert a merged HF model to GGUF and quantize, via a local llama.cpp.

    Requires ``config.export.llama_cpp_dir`` pointing at a built llama.cpp
    checkout (with ``convert_hf_to_gguf.py`` and the ``llama-quantize`` binary).
    Returns the list of produced GGUF files; logs and returns ``[]`` if the
    toolchain is unavailable rather than crashing the whole export.
    """
    llama_dir = config.export.llama_cpp_dir
    if not llama_dir:
        logger.warning("export.llama_cpp_dir not set; skipping GGUF export.")
        return []

    llama = Path(llama_dir)
    convert = llama / "convert_hf_to_gguf.py"
    if not convert.exists():
        logger.warning("convert_hf_to_gguf.py not found in %s; skipping GGUF.", llama)
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    f16_path = output_dir / "model-f16.gguf"

    logger.info("Converting %s -> %s (f16).", source_dir, f16_path)
    subprocess.run(  # noqa: S603 — args are config-derived, not user web input
        ["python", str(convert), str(source_dir), "--outfile", str(f16_path), "--outtype", "f16"],
        check=True,
    )

    produced: list[Path] = [f16_path]
    quantize_bin = _find_quantize_binary(llama)
    if quantize_bin is None:
        logger.warning("llama-quantize binary not found in %s; kept f16 GGUF only.", llama)
        return produced

    for quant in config.export.gguf_quant_types:
        out = output_dir / f"model-{quant}.gguf"
        logger.info("Quantizing -> %s (%s).", out, quant)
        subprocess.run(  # noqa: S603 — config-derived args
            [str(quantize_bin), str(f16_path), str(out), quant],
            check=True,
        )
        produced.append(out)

    logger.info("GGUF export produced: %s", ", ".join(p.name for p in produced))
    return produced


def _find_quantize_binary(llama_dir: Path) -> Path | None:
    """Locate the llama.cpp quantize binary across common build layouts."""
    for name in ("llama-quantize", "llama-quantize.exe", "quantize", "quantize.exe"):
        for candidate in (llama_dir / name, llama_dir / "build" / "bin" / name):
            if candidate.exists():
                return candidate
    return None


# --------------------------------------------------------------------------- #
# Orchestrator                                                                #
# --------------------------------------------------------------------------- #
def run_export(config: Config, adapter_path: str) -> None:
    """Merge the adapter, write the model card, push to the Hub, export GGUF.

    Args:
        config: Validated run configuration.
        adapter_path: Path to the trained LoRA adapter to export.
    """
    if not Path(adapter_path).exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    exp = config.export
    out_root = Path(config.general.output_dir)
    eval_report = _read_eval_report(config.eval.report_path)

    merged_dir: Path | None = None
    if exp.merge_adapter:
        merged_dir = merge_adapter(config, adapter_path, out_root / exp.merged_subdir)

        if exp.export_gguf and merged_dir is not None:
            export_gguf(config, merged_dir, out_root / exp.gguf_subdir)

    # Hub push (no-op when hub_repo_id is unset -> local-only export).
    if exp.hub_repo_id is None:
        logger.info("export.hub_repo_id not set; local export only (no Hub push).")
        # Still write a card next to local artifacts for review.
        if merged_dir is not None:
            card = build_model_card(
                config, repo_id="(local)", is_adapter=False, eval_report=eval_report
            )
            (merged_dir / "README.md").write_text(card, encoding="utf-8")
            logger.info("Wrote local model card -> %s", merged_dir / "README.md")
        return

    token = get_secret("HF_TOKEN", required=True)
    assert token is not None  # get_secret(required=True) raises otherwise

    if exp.push_merged and merged_dir is not None:
        card = build_model_card(
            config, repo_id=exp.hub_repo_id, is_adapter=False, eval_report=eval_report
        )
        (merged_dir / "README.md").write_text(card, encoding="utf-8")
        push_folder_to_hub(exp.hub_repo_id, merged_dir, private=exp.hub_private, token=token)

    if exp.push_adapter:
        adapter_repo = (
            exp.hub_repo_id + exp.adapter_repo_suffix if exp.push_merged else exp.hub_repo_id
        )
        card = build_model_card(
            config, repo_id=adapter_repo, is_adapter=True, eval_report=eval_report
        )
        (Path(adapter_path) / "README.md").write_text(card, encoding="utf-8")
        push_folder_to_hub(adapter_repo, Path(adapter_path), private=exp.hub_private, token=token)

    logger.info("Export complete.")
