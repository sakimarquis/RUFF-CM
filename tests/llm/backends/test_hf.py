from __future__ import annotations

import math
from types import SimpleNamespace

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
def test_hf_capture_batched_last_uses_non_pad_positions(backend):
    backend._tokenizer.padding_side = "right"
    messages = [[Message("user", "hi")], [Message("user", "hello world today")]]
    prompts = [backend._render_chat(sample) for sample in messages]
    encoded = backend._tokenizer(prompts, return_tensors="pt", padding=True)
    expected_positions = [[int(length.item()) - 1] for length in encoded.attention_mask.sum(dim=1)]

    result = backend.capture(messages, CaptureSpec(mode=CaptureMode.PREFILL, layers=[0], positions="last"))

    assert result.spec.positions == expected_positions
    assert expected_positions[0][0] < encoded.input_ids.shape[1] - 1


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


def test_hf_generate_stop_trim_sets_stop_finish_reason(monkeypatch):
    torch = pytest.importorskip("torch")
    backend = HfBackend("fake", device="cpu")
    backend._model = SimpleNamespace(generate=lambda *args, **kwargs: torch.tensor([[10, 20, 30, 31]]))
    backend._tokenizer = SimpleNamespace(pad_token_id=0, decode=lambda token_ids, skip_special_tokens: "answer STOP")
    monkeypatch.setattr(backend, "_ensure_loaded", lambda: None)
    monkeypatch.setattr(backend, "_encode_batch", lambda messages: (torch.tensor([[10, 20]]), torch.tensor([[1, 1]])))

    result = backend.generate([Message("user", "hello")], max_tokens=2, stop=[" STOP"])

    assert result.text == "answer"
    assert result.finish_reason == "stop"
