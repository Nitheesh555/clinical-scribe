"""Serving: launch an OpenAI-compatible vLLM server + a thin client.

Split the same way as the rest of the package: the request/command builders are
pure and CPU-testable; the actual server launch and HTTP call are thin wrappers
with their heavy work behind a subprocess / the stdlib ``urllib``.

Every served output is decorated with the clinician-review disclaimer
(:func:`decorate_draft`) so the "this is a draft" warning travels with the text
itself, not just the system prompt.
"""

from __future__ import annotations

import json
import logging
import subprocess
import urllib.request
from typing import Any

from .config import Config
from .utils import get_secret

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Pure, CPU-testable: command + request builders                             #
# --------------------------------------------------------------------------- #
def build_vllm_command(config: Config, model_path: str) -> list[str]:
    """Return the argv for an OpenAI-compatible vLLM server for ``model_path``.

    Serves the *merged* model (full precision); no 4-bit flag is passed.
    ``serve.max_model_len`` falls back to ``model.max_seq_length`` when unset.
    """
    s = config.serve
    max_len = s.max_model_len or config.model.max_seq_length
    return [
        "vllm",
        "serve",
        model_path,
        "--served-model-name",
        s.served_model_name,
        "--host",
        s.host,
        "--port",
        str(s.port),
        "--max-model-len",
        str(max_len),
        "--gpu-memory-utilization",
        str(s.gpu_memory_utilization),
        "--dtype",
        s.dtype,
    ]


def build_messages(config: Config, dialogue: str, section: str) -> list[dict[str, str]]:
    """Render the system+user chat messages for one section request.

    Mirrors the training format (:func:`clinical_scribe.data.build_example`) so
    inference matches what the model was trained on.
    """
    system = config.prompts.render_system(config.data.na_placeholder)
    user = config.prompts.user_template.format(dialogue=dialogue, section=section)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_chat_payload(config: Config, messages: list[dict[str, str]]) -> dict[str, Any]:
    """Build the OpenAI ``/v1/chat/completions`` request body (deterministic)."""
    s = config.serve
    return {
        "model": s.served_model_name,
        "messages": messages,
        "max_tokens": s.max_new_tokens,
        "temperature": s.temperature,
        "top_p": s.top_p,
    }


def decorate_draft(text: str, config: Config) -> str:
    """Prefix the model output with the clinician-review disclaimer banner."""
    return f"[DRAFT — {config.prompts.disclaimer}]\n\n{text.strip()}"


def base_url(config: Config, host: str | None = None) -> str:
    """Return the OpenAI-compatible base URL for the client.

    Uses ``localhost`` for client calls by default (the server may bind
    ``0.0.0.0``, which is not a valid connect address).
    """
    connect_host = host or ("localhost" if config.serve.host == "0.0.0.0" else config.serve.host)
    return f"http://{connect_host}:{config.serve.port}/v1"


# --------------------------------------------------------------------------- #
# Thin runtime wrappers: launch server + HTTP client                         #
# --------------------------------------------------------------------------- #
def run_server(config: Config, model_path: str) -> int:
    """Launch the vLLM OpenAI-compatible server (blocking). Returns its exit code."""
    cmd = build_vllm_command(config, model_path)
    logger.info("Launching vLLM server: %s", " ".join(cmd))
    return subprocess.call(cmd)  # noqa: S603 — config-derived argv, not web input


def _post_chat_completion(
    config: Config, payload: dict[str, Any], *, url: str, api_key: str | None
) -> str:
    """POST a chat-completion request and return the assistant content."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310 — fixed http scheme to local server
        f"{url}/chat/completions", data=data, headers=headers, method="POST"
    )
    with urllib.request.urlopen(
        request, timeout=config.serve.request_timeout_s
    ) as resp:  # noqa: S310
        body = json.loads(resp.read().decode("utf-8"))
    content: str = body["choices"][0]["message"]["content"]
    return content


def generate_section(
    config: Config,
    dialogue: str,
    section: str,
    *,
    url: str | None = None,
    api_key: str | None = None,
) -> str:
    """Request one clinical-note section from a running server (disclaimer-decorated).

    Args:
        config: Validated run configuration.
        dialogue: The doctor-patient dialogue.
        section: Human-readable target section (e.g. "Assessment").
        url: Override base URL (defaults to the configured local server).
        api_key: Optional bearer token (else read from ``VLLM_API_KEY`` env).

    Returns:
        The drafted section text, prefixed with the clinician-review banner.
    """
    messages = build_messages(config, dialogue, section)
    payload = build_chat_payload(config, messages)
    raw = _post_chat_completion(
        config,
        payload,
        url=url or base_url(config),
        api_key=api_key or get_secret("VLLM_API_KEY"),
    )
    return decorate_draft(raw, config)
