"""Tests for the pure (CPU-only) serving helpers: command + request builders."""

from __future__ import annotations

from clinical_scribe.config import Config
from clinical_scribe.serve import (
    base_url,
    build_chat_payload,
    build_messages,
    build_vllm_command,
    decorate_draft,
)


def test_build_vllm_command_uses_config() -> None:
    config = Config()
    cmd = build_vllm_command(config, "outputs/merged")
    assert cmd[:3] == ["vllm", "serve", "outputs/merged"]
    assert "--served-model-name" in cmd
    assert config.serve.served_model_name in cmd
    assert "--port" in cmd and str(config.serve.port) in cmd
    # max_model_len falls back to model.max_seq_length when unset.
    assert str(config.model.max_seq_length) in cmd


def test_build_vllm_command_respects_explicit_max_len() -> None:
    config = Config()
    config.serve.max_model_len = 4096
    cmd = build_vllm_command(config, "m")
    assert "4096" in cmd
    assert str(config.model.max_seq_length) not in cmd or config.model.max_seq_length == 4096


def test_build_messages_matches_training_format() -> None:
    config = Config()
    messages = build_messages(config, "Doctor: hello\nPatient: hi", "Assessment")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert config.prompts.disclaimer in messages[0]["content"]
    assert "Assessment" in messages[1]["content"]
    assert "Doctor: hello" in messages[1]["content"]


def test_build_chat_payload_is_deterministic() -> None:
    config = Config()
    payload = build_chat_payload(config, build_messages(config, "d", "Plan"))
    assert payload["model"] == config.serve.served_model_name
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == config.serve.max_new_tokens


def test_decorate_draft_prepends_disclaimer() -> None:
    config = Config()
    out = decorate_draft("  Three-day cough.  ", config)
    assert out.startswith("[DRAFT")
    assert config.prompts.disclaimer in out
    assert out.rstrip().endswith("Three-day cough.")


def test_base_url_rewrites_bind_all_to_localhost() -> None:
    config = Config()  # host defaults to 0.0.0.0
    assert base_url(config) == f"http://localhost:{config.serve.port}/v1"
    config.serve.host = "example.internal"
    assert base_url(config) == f"http://example.internal:{config.serve.port}/v1"
