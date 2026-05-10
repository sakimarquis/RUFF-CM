from __future__ import annotations

from ruff_cm.llm.extract_hiddens.sglang import normalize_sglang_url


def test_normalize_sglang_url_appends_generate():
    assert normalize_sglang_url("http://x:8080") == "http://x:8080/generate"


def test_normalize_sglang_url_is_idempotent():
    assert normalize_sglang_url("http://x:8080/generate") == "http://x:8080/generate"


def test_normalize_sglang_url_replaces_openai_compatible_path():
    assert normalize_sglang_url("http://x:8080/v1/chat/completions") == "http://x:8080/generate"
