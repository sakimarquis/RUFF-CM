from __future__ import annotations

from pathlib import Path

import pytest

from ruff_cm.llm.reasoning import resolve_thinking


def old_number_game_display_suffix(enable_thinking: bool) -> str:
    return "_thinking" if enable_thinking else ""


@pytest.mark.parity
def test_resolve_thinking_display_suffix_matches_number_game_fixture(tmp_path: Path):
    aliases = tmp_path / "aliases.yml"
    aliases.write_text("qwen:\n  backend: hf\n  model_id: Qwen/Qwen3-4B\n", encoding="utf-8")
    cfg = resolve_thinking("qwen", {}, cli_thinking=True, aliases_path=aliases)
    assert cfg.display_suffix == old_number_game_display_suffix(True)
