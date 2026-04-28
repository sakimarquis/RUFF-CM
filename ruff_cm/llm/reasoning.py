from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .backends.registry import DEFAULT_ALIASES_PATH, load_aliases


@dataclass(frozen=True)
class ThinkingConfig:
    enable_thinking: bool
    thinking_budget: int
    reasoning_effort: str | None
    reasoning_budget: int
    google_thinking_level: str | None
    google_reasoning_budget: int
    display_suffix: str


def _disabled() -> ThinkingConfig:
    return ThinkingConfig(
        enable_thinking=False,
        thinking_budget=0,
        reasoning_effort=None,
        reasoning_budget=0,
        google_thinking_level=None,
        google_reasoning_budget=0,
        display_suffix="",
    )


def resolve_thinking(
    model_alias: str,
    config: dict,
    cli_thinking: bool = False,
    *,
    aliases_path: Path = DEFAULT_ALIASES_PATH,
) -> ThinkingConfig:
    aliases = load_aliases(aliases_path)
    alias_cfg = aliases[model_alias]
    backend = alias_cfg["backend"]

    if backend == "hf":
        enabled = cli_thinking or bool(config.get("ENABLE_THINKING"))
        return ThinkingConfig(
            enable_thinking=enabled,
            thinking_budget=int(config.get("THINKING_BUDGET", 0)),
            reasoning_effort=None,
            reasoning_budget=0,
            google_thinking_level=None,
            google_reasoning_budget=0,
            display_suffix="_thinking" if enabled else "",
        )

    if backend == "google_cloud":
        model = alias_cfg["model"]
        default_level = "MINIMAL" if "flash-lite" in model else "HIGH"
        google_thinking_level = "MEDIUM" if cli_thinking and default_level == "MINIMAL" else default_level
        if not cli_thinking:
            google_thinking_level = config.get("GOOGLE_THINKING_LEVEL") or google_thinking_level
        enabled = google_thinking_level != "MINIMAL"
        return ThinkingConfig(
            enable_thinking=enabled,
            thinking_budget=0,
            reasoning_effort=None,
            reasoning_budget=0,
            google_thinking_level=google_thinking_level,
            google_reasoning_budget=int(config.get("GOOGLE_REASONING_BUDGET", 0)),
            display_suffix="_thinking" if enabled else "",
        )

    if backend == "api":
        provider = alias_cfg.get("provider")
        if provider == "openai":
            reasoning_effort = config.get("REASONING_EFFORT") or ("medium" if cli_thinking else None)
            enabled = cli_thinking or reasoning_effort not in (None, "none")
            return ThinkingConfig(
                enable_thinking=enabled,
                thinking_budget=0,
                reasoning_effort=reasoning_effort,
                reasoning_budget=int(config.get("REASONING_BUDGET", 0)),
                google_thinking_level=None,
                google_reasoning_budget=0,
                display_suffix="_thinking" if enabled else "",
            )

    return _disabled()
