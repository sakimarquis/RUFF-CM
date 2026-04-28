from __future__ import annotations

import pytest

from ruff_cm.llm.backends.base import BackendCapabilityError, CaptureResult, ChoiceScores, GenerateResult, Generator, HiddenReader, Message, Scorer


def test_message_is_frozen():
    message = Message(role="user", content="hello")
    with pytest.raises(Exception):
        message.content = "changed"  # type: ignore[misc]


def test_choice_scores_exact_invariant_shape():
    result = ChoiceScores(method="exact", scores={"A": -0.1, "B": -2.0}, complete=True, missing=[], fallback_count=0)
    assert result.method == "exact"
    assert result.complete is True
    assert result.missing == []
    assert result.fallback_count == 0


def test_choice_scores_partial_has_missing_candidates():
    result = ChoiceScores(method="partial", scores={"A": -0.1}, complete=False, missing=["B"], fallback_count=0)
    assert result.method == "partial"
    assert result.complete is False
    assert result.missing == ["B"]


def test_protocols_are_runtime_checkable():
    class Backend:
        name = "fake"
        capabilities = frozenset({"generate", "score_exact", "hidden_prefill"})

        def generate(self, messages, *, temperature=0.0, max_tokens=256, stop=None, seed=None):
            return GenerateResult(text="ok", finish_reason="stop")

        def score_choices(self, messages, choice_set):
            return ChoiceScores(method="exact", scores={}, complete=True, missing=[], fallback_count=0)

        def capture(self, messages, spec):
            return CaptureResult(hiddens={}, logits=None, token_ids=None, spec=spec, valid_mask=None)

    backend = Backend()
    assert isinstance(backend, Generator)
    assert isinstance(backend, Scorer)
    assert isinstance(backend, HiddenReader)


def test_capability_error_is_exception():
    assert issubclass(BackendCapabilityError, Exception)
