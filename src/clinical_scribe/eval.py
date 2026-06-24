"""Evaluation: ROUGE-L, BERTScore, structure validity, faithfulness, base-vs-FT.

Design: the metric functions (structure validity, clinical-term overlap) are
pure and stdlib-only, so they are unit-testable on CPU. ROUGE/BERTScore and all
model inference are lazily imported (GPU/heavy) and only run inside
:func:`run_evaluation`.

Produces ``eval_report.md`` and runs the ACI-Bench held-out stress test. The
faithfulness numbers are deterministic *proxies* (clinical-term overlap), not a
substitute for clinical review; an optional LLM-as-judge can be added later.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Config

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Pure metric helpers (stdlib only — unit-testable without a GPU)
# --------------------------------------------------------------------------- #

# Chat-template / reasoning artifacts that must never appear in a clean section.
_FORBIDDEN_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<|endoftext|>",
)

# Small English stoplist for the clinical-term proxy (kept intentionally short).
_STOPWORDS = frozenset(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "then",
        "this",
        "that",
        "these",
        "those",
        "with",
        "without",
        "within",
        "for",
        "from",
        "into",
        "onto",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "shall",
        "should",
        "can",
        "could",
        "may",
        "might",
        "must",
        "not",
        "no",
        "nor",
        "so",
        "than",
        "too",
        "very",
        "your",
        "you",
        "yours",
        "their",
        "there",
        "here",
        "what",
        "which",
        "who",
        "whom",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "other",
        "patient",
        "reports",
        "denies",
        "states",
        "history",
        "section",
        "note",
        "please",
        "write",
        "following",
        "also",
        "given",
        "about",
        "over",
        "under",
        "again",
    ]
)

_WORD_RE = re.compile(r"[a-z]+|\d+(?:\.\d+)?")


def extract_clinical_terms(text: str) -> set[str]:
    """Heuristic content-term extractor used for the faithfulness proxy.

    Keeps numbers/measurements and alphabetic tokens of length >= 4 that are not
    common stopwords. This is a deterministic approximation of clinical-entity
    overlap, not a medical NER.
    """
    terms: set[str] = set()
    for tok in _WORD_RE.findall(text.lower()):
        if any(c.isdigit() for c in tok) or (len(tok) >= 4 and tok not in _STOPWORDS):
            terms.add(tok)
    return terms


def is_structurally_valid(text: str, na_placeholder: str = "N/A") -> bool:
    """Whether a generated section parses cleanly (the structure-validity gate).

    Invalid when: empty, contains chat-template/reasoning artifacts, or is
    degenerately repetitive. An exact ``N/A`` is valid.
    """
    t = text.strip()
    if not t:
        return False
    if t == na_placeholder:
        return True
    if any(tok in t for tok in _FORBIDDEN_TOKENS):
        return False
    words = t.split()
    if len(words) >= 10:
        most_common = max((words.count(w) for w in set(words)), default=0)
        if most_common / len(words) > 0.6:  # one token dominates -> degenerate
            return False
    return True


def structure_validity_rate(predictions: list[str], na_placeholder: str = "N/A") -> float:
    """Fraction of predictions that pass :func:`is_structurally_valid`."""
    if not predictions:
        return 0.0
    valid = sum(is_structurally_valid(p, na_placeholder) for p in predictions)
    return valid / len(predictions)


def grounding_score(prediction: str, source: str) -> float:
    """Share of the prediction's clinical terms that are supported by the source.

    1.0 means fully grounded; ``1 - grounding`` is the hallucination proxy.
    """
    pred_terms = extract_clinical_terms(prediction)
    if not pred_terms:
        return 1.0  # nothing asserted -> nothing hallucinated
    source_terms = extract_clinical_terms(source)
    return len(pred_terms & source_terms) / len(pred_terms)


def completeness_score(prediction: str, reference: str) -> float:
    """Share of the reference's clinical terms recovered by the prediction.

    ``1 - completeness`` is the omission proxy.
    """
    ref_terms = extract_clinical_terms(reference)
    if not ref_terms:
        return 1.0
    pred_terms = extract_clinical_terms(prediction)
    return len(ref_terms & pred_terms) / len(ref_terms)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# --------------------------------------------------------------------------- #
# Heavy metrics (lazy imports)
# --------------------------------------------------------------------------- #


def compute_rouge_l(predictions: list[str], references: list[str]) -> float:
    """Mean ROUGE-L F-measure (stemmed)."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(ref, pred)["rougeL"].fmeasure
        for pred, ref in zip(predictions, references, strict=True)
    ]
    return _mean(scores)


def compute_bertscore(
    predictions: list[str], references: list[str], model_type: str, lang: str
) -> float:
    """Mean BERTScore F1."""
    from bert_score import score as bert_score_fn

    _, _, f1 = bert_score_fn(
        predictions, references, model_type=model_type, lang=lang, verbose=False
    )
    return float(f1.mean().item())


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class EvalExample:
    """One evaluation item: the prompt, the reference, and the source text."""

    prompt_messages: list[dict[str, str]]
    reference: str
    source: str
    meta: dict[str, str] = field(default_factory=dict)


@dataclass
class ModelScores:
    """Aggregated metrics for one model on one dataset."""

    name: str
    n: int
    rouge_l: float = 0.0
    bertscore_f: float = 0.0
    structure_validity: float = 0.0
    hallucination_rate: float = 0.0
    omission_rate: float = 0.0


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #


def load_mts_eval_examples(config: Config) -> list[EvalExample]:
    """Load MTS-Dialog test JSONL into eval examples (prepares data if missing)."""
    processed = Path(config.data.processed_dir)
    if not all((processed / f).exists() for f in config.eval.test_files):
        logger.info("Test JSONL missing; running data preparation.")
        from .data import prepare_dataset

        prepare_dataset(config)

    examples: list[EvalExample] = []
    for fname in config.eval.test_files:
        path = processed / fname
        if not path.exists():
            logger.warning("Missing test file: %s", path)
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            msgs = row["messages"]
            prompt = [m for m in msgs if m["role"] in ("system", "user")]
            reference = next(m["content"] for m in msgs if m["role"] == "assistant")
            source = next(m["content"] for m in msgs if m["role"] == "user")
            examples.append(
                EvalExample(
                    prompt_messages=prompt,
                    reference=reference,
                    source=source,
                    meta={"section": row.get("section_name", ""), "dataset": "mts"},
                )
            )
    logger.info("Loaded %d MTS-Dialog eval examples.", len(examples))
    return examples


def load_aci_bench_examples(config: Config) -> list[EvalExample]:
    """Load ACI-Bench held-out splits as full-note eval examples."""
    from datasets import load_dataset

    disclaimer = config.prompts.disclaimer
    system = (
        "You are a clinical documentation assistant. From the doctor-patient "
        "dialogue, draft a structured clinical note.\n" + disclaimer + "\n"
        "Use only information present in the dialogue; do not invent facts."
    )
    ds = load_dataset(config.eval.aci_bench_dataset_id)
    examples: list[EvalExample] = []
    for split in config.eval.aci_bench_splits:
        if split not in ds:
            logger.warning("ACI-Bench split not found: %s", split)
            continue
        for row in ds[split]:
            dialogue = row["dialogue"]
            user = f"Dialogue:\n{dialogue}\n\nWrite the full clinical note."
            examples.append(
                EvalExample(
                    prompt_messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    reference=row["note"],
                    source=dialogue,
                    meta={"dataset": "aci", "split": split},
                )
            )
    logger.info("Loaded %d ACI-Bench eval examples.", len(examples))
    return examples


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #


def _load_inference_model(config: Config, adapter_path: str | None) -> tuple[Any, Any]:
    """Load the base (optionally adapter-wrapped) model + tokenizer for generation."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from .utils import resolve_runtime

    runtime = resolve_runtime(config.model.dtype, config.model.attn_implementation)
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    compute_dtype = dtype_map[runtime.dtype]

    quant = None
    if config.model.load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model_id)
    tokenizer.padding_side = "left"  # decoder-only batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model.base_model_id,
        quantization_config=quant,
        dtype=compute_dtype,
        attn_implementation=runtime.attn_implementation,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        logger.info("Attached adapter from %s", adapter_path)
    model.eval()
    return model, tokenizer


def _generate(model: Any, tokenizer: Any, examples: list[EvalExample], config: Config) -> list[str]:
    """Deterministically generate completions for each example's prompt."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    preds: list[str] = []
    bs = config.eval.batch_size
    for start in range(0, len(examples), bs):
        batch = examples[start : start + bs]
        texts = [
            tokenizer.apply_chat_template(
                ex.prompt_messages, tokenize=False, add_generation_prompt=True
            )
            for ex in batch
        ]
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.model.max_seq_length,
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=config.eval.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_len = enc["input_ids"].shape[1]
        for j in range(len(batch)):
            gen = out[j][prompt_len:]
            preds.append(tokenizer.decode(gen, skip_special_tokens=True).strip())
        logger.info("Generated %d/%d", min(start + bs, len(examples)), len(examples))
    return preds


# --------------------------------------------------------------------------- #
# Scoring & orchestration
# --------------------------------------------------------------------------- #


def score_predictions(
    name: str, predictions: list[str], examples: list[EvalExample], config: Config
) -> ModelScores:
    """Compute all configured metrics for a set of predictions."""
    refs = [ex.reference for ex in examples]
    srcs = [ex.source for ex in examples]
    na = config.data.na_placeholder

    scores = ModelScores(name=name, n=len(predictions))
    if config.eval.compute_rouge:
        scores.rouge_l = compute_rouge_l(predictions, refs)
    if config.eval.compute_bertscore:
        scores.bertscore_f = compute_bertscore(
            predictions, refs, config.eval.bertscore_model, config.eval.bertscore_lang
        )
    if config.eval.check_structure_validity:
        scores.structure_validity = structure_validity_rate(predictions, na)
    if config.eval.faithfulness_entity_overlap:
        scores.hallucination_rate = _mean(
            [1.0 - grounding_score(p, s) for p, s in zip(predictions, srcs, strict=True)]
        )
        scores.omission_rate = _mean(
            [1.0 - completeness_score(p, r) for p, r in zip(predictions, refs, strict=True)]
        )
    logger.info(
        "%s: ROUGE-L=%.4f BERTScore=%.4f valid=%.3f halluc=%.3f omit=%.3f (n=%d)",
        name,
        scores.rouge_l,
        scores.bertscore_f,
        scores.structure_validity,
        scores.hallucination_rate,
        scores.omission_rate,
        scores.n,
    )
    return scores


def evaluate_model(
    config: Config, examples: list[EvalExample], adapter_path: str | None, label: str
) -> tuple[ModelScores, list[str]]:
    """Load a model, generate, score, and free GPU memory."""
    import gc

    import torch

    model, tokenizer = _load_inference_model(config, adapter_path)
    try:
        preds = _generate(model, tokenizer, examples, config)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return score_predictions(label, preds, examples, config), preds


def check_success_gate(base: ModelScores, ft: ModelScores, config: Config) -> dict[str, bool]:
    """Evaluate the Phase 1 success-gate criteria for fine-tuned vs base."""
    checks = {
        "rougeL_beats_base": ft.rouge_l > base.rouge_l,
        "bertscore_beats_base": ft.bertscore_f > base.bertscore_f,
        "structure_validity_ok": ft.structure_validity >= config.eval.gate_min_structure_validity,
        "faithfulness_no_worse": ft.hallucination_rate <= base.hallucination_rate + 1e-9,
    }
    checks["passed"] = all(checks.values())
    return checks


def _scores_table(results: list[ModelScores]) -> str:
    """Render a markdown metrics table."""
    head = (
        "| Model | n | ROUGE-L | BERTScore-F | Struct-valid | Halluc.↓ | Omission↓ |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    rows = "".join(
        f"| {s.name} | {s.n} | {s.rouge_l:.4f} | {s.bertscore_f:.4f} | "
        f"{s.structure_validity:.3f} | {s.hallucination_rate:.3f} | {s.omission_rate:.3f} |\n"
        for s in results
    )
    return head + rows


def write_eval_report(
    config: Config,
    mts_results: list[ModelScores],
    aci_results: list[ModelScores],
    gate: dict[str, bool] | None,
    samples: list[dict[str, str]],
) -> None:
    """Write the markdown evaluation report."""
    path = Path(config.eval.report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Clinical Scribe — Phase 1 Evaluation Report\n")
    lines.append(f"> {config.prompts.disclaimer}\n")
    lines.append(f"\n**Base model:** `{config.model.base_model_id}`\n")
    lines.append("\n## MTS-Dialog test (section-level)\n")
    lines.append(_scores_table(mts_results))
    lines.append(
        "\n*Faithfulness is a deterministic clinical-term-overlap proxy "
        "(hallucination = unsupported terms; omission = missed reference terms), "
        "not a clinical judgment.*\n"
    )

    if gate is not None:
        lines.append("\n## Phase 1 success gate\n")
        emoji = {True: "✅", False: "❌"}
        for k, v in gate.items():
            if k == "passed":
                continue
            lines.append(f"- {emoji[v]} `{k}`\n")
        lines.append(f"\n**Gate {'PASSED ✅' if gate['passed'] else 'NOT passed ❌'}**\n")

    if aci_results:
        lines.append("\n## ACI-Bench held-out stress test (full-note)\n")
        lines.append(
            "*Expected to be weaker — the Phase 1 model is trained for single "
            "sections, not full notes. Reported for honesty.*\n\n"
        )
        lines.append(_scores_table(aci_results))

    if samples:
        lines.append("\n## Qualitative samples\n")
        for i, s in enumerate(samples, 1):
            lines.append(f"\n### Example {i} — {s.get('section', '')}\n")
            lines.append(f"**Dialogue (truncated):** {s['source'][:400]}\n\n")
            lines.append(f"**Reference:** {s['reference']}\n\n")
            lines.append(f"**Base:** {s['base']}\n\n")
            if "finetuned" in s:
                lines.append(f"**Fine-tuned:** {s['finetuned']}\n")

    path.write_text("".join(lines), encoding="utf-8")
    logger.info("Wrote eval report -> %s", path)


def run_evaluation(config: Config, adapter_path: str | None = None) -> None:
    """Evaluate base vs fine-tuned model and write the eval report.

    Args:
        config: Validated run configuration.
        adapter_path: Path to a trained LoRA adapter (None = base model only).
    """
    examples = load_mts_eval_examples(config)
    base_scores, base_preds = evaluate_model(config, examples, None, "MTS base")
    mts_results = [base_scores]
    ft_preds: list[str] | None = None
    gate: dict[str, bool] | None = None

    if adapter_path:
        ft_scores, ft_preds = evaluate_model(config, examples, adapter_path, "MTS fine-tuned")
        mts_results.append(ft_scores)
        gate = check_success_gate(base_scores, ft_scores, config)
        logger.info("Success gate: %s", gate)

    # Qualitative samples.
    samples: list[dict[str, str]] = []
    for i in range(min(config.eval.num_qualitative_samples, len(examples))):
        s = {
            "section": examples[i].meta.get("section", ""),
            "source": examples[i].source,
            "reference": examples[i].reference,
            "base": base_preds[i],
        }
        if ft_preds is not None:
            s["finetuned"] = ft_preds[i]
        samples.append(s)

    # ACI-Bench held-out stress test.
    aci_results: list[ModelScores] = []
    if config.eval.aci_bench_stress_test:
        try:
            aci_examples = load_aci_bench_examples(config)
            aci_results.append(evaluate_model(config, aci_examples, None, "ACI base")[0])
            if adapter_path:
                aci_results.append(
                    evaluate_model(config, aci_examples, adapter_path, "ACI fine-tuned")[0]
                )
        except Exception as exc:  # noqa: BLE001 — never let the stress test break the report
            logger.warning("ACI-Bench stress test skipped: %s", exc)

    write_eval_report(config, mts_results, aci_results, gate, samples)
