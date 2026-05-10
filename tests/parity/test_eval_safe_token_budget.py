from __future__ import annotations

from types import SimpleNamespace

import torch

from ruff_cm.eval import _safe_token_budget, auto_max_chars, short_answer_match


class CpuModel:
    config = SimpleNamespace(max_position_embeddings=1024)

    def parameters(self):
        yield torch.zeros(1)


def test_safe_token_budget_matches_pr_cpu_path():
    assert _safe_token_budget(CpuModel(), 128) == 896


def test_eval_small_utilities_match_pr_contract():
    tokenizer = SimpleNamespace(model_max_length=256)
    assert auto_max_chars(tokenizer) == 1024
    assert short_answer_match("```text\nAnswer\n```", "answer")
