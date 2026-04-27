from __future__ import annotations

import math

import pytest

from ruff_cm.llm.backends.base import BackendCapabilityError, Message
from ruff_cm.llm.backends.hf import HfBackend
from ruff_cm.llm.choice import ChoiceSet
from ruff_cm.llm.hooks import CaptureMode, CaptureSpec


@pytest.fixture
def backend(tiny_hf):
    hf = HfBackend(tiny_hf["model_id"], dtype="float32", device="cpu")
    hf._model = tiny_hf["model"]
    hf._tokenizer = tiny_hf["tokenizer"]
    return hf


@pytest.mark.hf
def test_hf_backend_capabilities(backend):
    assert backend.capabilities == frozenset({"generate", "score_exact", "hidden_prefill", "hidden_teacher_forcing_sparse"})


@pytest.mark.hf
def test_hf_score_choices_exact_complete(backend):
    choice_set = ChoiceSet(backend._tokenizer, ["A", "B"], variants=["raw", "with_space"])
    scores = backend.score_choices([Message("user", "Choose A or B:")], choice_set)
    assert scores.method == "exact"
    assert scores.complete is True
    assert scores.missing == []
    assert math.isclose(sum(math.exp(value) for value in scores.scores.values()), 1.0, abs_tol=1e-4)


@pytest.mark.hf
def test_hf_capture_prefill(backend):
    result = backend.capture([Message("user", "hello world")], CaptureSpec(mode=CaptureMode.PREFILL, layers=[0], positions="last", with_logits=True))
    assert set(result.hiddens) == {0}
    assert result.hiddens[0].shape[1] == 1
    assert result.logits.shape[1] == 1


@pytest.mark.hf
def test_hf_capture_teacher_forcing_sparse(backend):
    result = backend.capture(
        [[Message("user", "hello")], [Message("user", "world")]],
        CaptureSpec(mode=CaptureMode.TEACHER_FORCING_SPARSE, layers=[0], positions=[[0, 1], [0]], target_text=[" A", " B"]),
    )
    assert result.hiddens[0].shape[:2] == (2, 2)
    assert result.valid_mask.tolist() == [[True, True], [True, False]]


@pytest.mark.hf
def test_hf_capture_generate_steps_rejected(backend):
    with pytest.raises(BackendCapabilityError):
        backend.capture([Message("user", "hello")], CaptureSpec(mode=CaptureMode.GENERATE_STEPS))
