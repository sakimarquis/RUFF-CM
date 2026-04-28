from __future__ import annotations

from pathlib import Path

from ruff_cm.llm.reasoning import ThinkingConfig, resolve_thinking


def alias_file(tmp_path: Path) -> Path:
    path = tmp_path / "aliases.yml"
    path.write_text(
        "\n".join(
            [
                "qwen:",
                "  backend: hf",
                "  model_id: Qwen/Qwen3-4B",
                "gpt:",
                "  backend: api",
                "  provider: openai",
                "  model: gpt-5-mini",
                "gemini-pro:",
                "  backend: google_cloud",
                "  model: gemini-3.1-pro",
                "gemini-lite:",
                "  backend: google_cloud",
                "  model: gemini-3.1-flash-lite",
                "api-google:",
                "  backend: api",
                "  provider: google_cloud",
                "  model: gemini-3.1-pro",
                "custom-openai:",
                "  backend: custom",
                "  provider: openai",
                "  model: custom-openai-model",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_hf_thinking_enabled_by_cli(tmp_path):
    cfg = resolve_thinking("qwen", {}, cli_thinking=True, aliases_path=alias_file(tmp_path))
    assert isinstance(cfg, ThinkingConfig)
    assert cfg.enable_thinking is True
    assert cfg.display_suffix == "_thinking"


def test_openai_reasoning_effort_from_config(tmp_path):
    cfg = resolve_thinking("gpt", {"REASONING_EFFORT": "medium"}, aliases_path=alias_file(tmp_path))
    assert cfg.reasoning_effort == "medium"
    assert cfg.display_suffix == "_thinking"


def test_gemini_flash_lite_defaults_to_minimal(tmp_path):
    cfg = resolve_thinking("gemini-lite", {}, aliases_path=alias_file(tmp_path))
    assert cfg.google_thinking_level == "MINIMAL"
    assert cfg.display_suffix == ""


def test_gemini_flash_lite_cli_upgrades_to_medium(tmp_path):
    cfg = resolve_thinking("gemini-lite", {}, cli_thinking=True, aliases_path=alias_file(tmp_path))
    assert cfg.google_thinking_level == "MEDIUM"
    assert cfg.display_suffix == "_thinking"


def test_gemini_pro_defaults_high(tmp_path):
    cfg = resolve_thinking("gemini-pro", {}, aliases_path=alias_file(tmp_path))
    assert cfg.google_thinking_level == "HIGH"


def test_api_google_cloud_does_not_resolve_gemini_thinking(tmp_path):
    cfg = resolve_thinking("api-google", {}, aliases_path=alias_file(tmp_path))
    assert cfg == ThinkingConfig(
        enable_thinking=False,
        thinking_budget=0,
        reasoning_effort=None,
        reasoning_budget=0,
        google_thinking_level=None,
        google_reasoning_budget=0,
        display_suffix="",
    )


def test_unknown_backend_ignores_provider_reasoning(tmp_path):
    cfg = resolve_thinking(
        "custom-openai",
        {"REASONING_EFFORT": "medium"},
        aliases_path=alias_file(tmp_path),
    )
    assert cfg == ThinkingConfig(
        enable_thinking=False,
        thinking_budget=0,
        reasoning_effort=None,
        reasoning_budget=0,
        google_thinking_level=None,
        google_reasoning_budget=0,
        display_suffix="",
    )
