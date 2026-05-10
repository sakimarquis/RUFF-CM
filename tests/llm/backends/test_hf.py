from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from ruff_cm.llm.backends.base import BackendCapabilityError, Message
from ruff_cm.llm.backends.hf import HfBackend, _spec_with_non_pad_last_positions
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


def test_hf_last_position_resolution_handles_left_padding():
    torch = pytest.importorskip("torch")
    spec = CaptureSpec(mode=CaptureMode.PREFILL, layers=[0], positions="last")

    resolved = _spec_with_non_pad_last_positions(spec, torch.tensor([[0, 0, 1, 1]]))

    assert resolved.positions == [[3]]


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


class FakeThinkingTokenizer:
    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "<eos>"
    eos_token_id = 99
    chat_template = "thinking-template"
    name_or_path = "Qwen/Qwen3-0.6B"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        if messages and "reasoning_content" in messages[-1]:
            assistant = messages[-1]
            return f"user ___RUFF_PROMPT_PROBE___<think>{assistant['reasoning_content']}</think>{assistant['content']}"
        return "prompt <think>" if enable_thinking else "prompt"

    def encode(self, text, add_special_tokens=False):
        table = {"<think>": [10], "</think>": [11], "A": [5], "B": [6], "Yes": [7], "No": [8]}
        return table.get(text, [1])

    def __call__(self, texts, return_tensors="pt", padding=False):
        torch = pytest.importorskip("torch")
        batch = texts if isinstance(texts, list) else [texts]
        encoded = [[1, 10] if "<think>" in text else [1] for text in batch]
        width = max(len(row) for row in encoded)
        input_ids = torch.tensor([[self.pad_token_id] * (width - len(row)) + row for row in encoded])
        attention_mask = torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in encoded])
        return SimpleNamespace(input_ids=input_ids, attention_mask=attention_mask)

    def decode(self, token_ids, skip_special_tokens=True):
        pieces = {11: "</think>", 20: "hidden", 30: "answer", 99: ""}
        return "".join(pieces.get(int(token_id), "") for token_id in token_ids)


class FakeThinkingModel:
    def __init__(self):
        torch = pytest.importorskip("torch")
        self.device = torch.device("cpu")
        self.generate_calls = []
        self.forward_calls = 0

    def eval(self):
        return self

    def generate(self, **kwargs):
        torch = pytest.importorskip("torch")
        self.generate_calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        logits_processor = kwargs.get("logits_processor") or []
        if logits_processor:
            output_ids = torch.cat([input_ids, torch.tensor([[20, 11]])], dim=1)
            scores = torch.zeros((input_ids.shape[0], 128))
            scores[:, 5] = 5.0
            scores[:, 6] = 1.0
            for processor in logits_processor:
                scores = processor(output_ids, scores)
            return output_ids
        if len(self.generate_calls) == 1:
            return torch.cat([input_ids, torch.tensor([[20, 11, 99]])], dim=1)
        return torch.cat([input_ids, torch.tensor([[30]])], dim=1)

    def __call__(self, **kwargs):
        self.forward_calls += 1
        raise AssertionError("thinking score_choices should use captured logits before fallback forward")


def _fake_thinking_backend(**kwargs):
    backend = HfBackend("Qwen/Qwen3-0.6B", dtype="float32", device="cpu", **kwargs)
    backend._tokenizer = FakeThinkingTokenizer()
    backend._model = FakeThinkingModel()
    return backend


def test_hf_generate_thinking_two_stage_populates_metadata():
    backend = _fake_thinking_backend(enable_thinking=True, max_thinking_tokens=1, max_answer_tokens=3)

    result = backend.generate([Message("user", "answer")], max_tokens=3)

    assert result.text == "answer"
    assert result.thinking_tokens == 1
    assert result.max_thinking_tokens == 1
    assert result.thinking_truncated is True
    assert len(backend._model.generate_calls) == 2
    assert backend._model.generate_calls[1]["input_ids"][0, -1].item() == 11


def test_hf_score_choices_thinking_captures_post_close_logits_without_forward_fallback():
    backend = _fake_thinking_backend(enable_thinking=True, max_thinking_tokens=8)
    choice_set = ChoiceSet(backend._tokenizer, ["A", "B"])

    scores = backend.score_choices([Message("user", "A or B?")], choice_set)

    assert scores.complete is True
    assert scores.fallback_count == 0
    assert scores.scores["A"] > scores.scores["B"]
    assert backend._model.forward_calls == 0
