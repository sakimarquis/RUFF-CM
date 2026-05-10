from __future__ import annotations

import pytest

from ruff_cm.llm.backends.base import BackendCapabilityError, CaptureResult, GenerateResult, Message
from ruff_cm.llm.extract_hiddens.capture import CaptureMode, CaptureSpec
from ruff_cm.llm.choice import ChoiceSet
from ruff_cm.llm.families import ModelFamily, PostMarkerTerminal
from ruff_cm.llm.inference import BudgetSpec, SamplingConfig, ScoringSpec, generate


class RuntimeTokenizer:
    name_or_path = "Qwen/Qwen3-4B"
    chat_template = "<|im_start|>{role}\n{content}<|im_end|>\n"

    def apply_chat_template(self, messages, *, add_generation_prompt=False, tokenize=False, **kwargs):
        rendered = "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n" for message in messages
        )
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        return self.encode(rendered) if tokenize else rendered

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        encoded = {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}
        if return_offsets_mapping:
            encoded["offset_mapping"] = [(idx, idx + 1) for idx in range(len(text))]
        return encoded

    def encode(self, text, *, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(int(token_id)) for token_id in ids)


class CustomBudgetProcessor:
    pass


class RuntimeBackend:
    name = "runtime-test"
    capabilities = frozenset({"generate", "hidden_teacher_forcing_sparse"})

    def __init__(self):
        self.tokenizer = RuntimeTokenizer()
        self.family = ModelFamily(
            id="runtime-test-family",
            name_markers=("runtime-test",),
            terminal_answer_strategy=PostMarkerTerminal(),
            budget_processor=CustomBudgetProcessor,
        )
        self.generated_text = "<think>hidden</think> A."
        self.generate_kwargs = None
        self.capture_specs = []

    def generate(
        self,
        messages,
        *,
        temperature=0.0,
        max_tokens=256,
        stop=None,
        seed=None,
        thinking_budget=None,
        budget_processor=None,
    ):
        self.generate_kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": stop,
            "seed": seed,
            "thinking_budget": thinking_budget,
            "budget_processor": budget_processor,
        }
        return GenerateResult(text=self.generated_text, finish_reason="stop", raw={"ok": True})

    def capture(self, messages, spec):
        torch = pytest.importorskip("torch")
        self.capture_specs.append(spec)
        n_positions = len(spec.positions)
        logits = torch.full((1, n_positions, 128), -8.0)
        logits[:, :, ord("A")] = 2.0
        logits[:, :, ord("B")] = -2.0
        hiddens = {
            layer: torch.arange(n_positions * 2, dtype=torch.float32).reshape(1, n_positions, 2) + 100 * layer
            for layer in spec.layers
        }
        return CaptureResult(
            hiddens=hiddens,
            logits=logits if spec.with_logits else None,
            token_ids=torch.tensor([[1, 2, 3]]),
            spec=spec,
            valid_mask=torch.ones((1, n_positions), dtype=torch.bool),
        )


def test_runtime_generate_builds_trajectory_and_passes_budget_processor_from_family():
    backend = RuntimeBackend()
    result = generate(
        backend,
        [Message("user", "question")],
        budget=BudgetSpec(max_thinking_tokens=7),
        sampling=SamplingConfig(max_tokens=11, temperature=0.25, stop=("END",), seed=3),
    )

    assert result.text == backend.generated_text
    assert result.finish == "stop"
    assert result.trajectory.thinking_span is not None
    assert backend.generate_kwargs == {
        "temperature": 0.25,
        "max_tokens": 11,
        "stop": ["END"],
        "seed": 3,
        "thinking_budget": 7,
        "budget_processor": CustomBudgetProcessor,
    }


def test_runtime_resolves_lazy_backend_tokenizer_after_generation():
    class LazyBackend(RuntimeBackend):
        def __init__(self):
            super().__init__()
            del self.tokenizer
            self._tokenizer = None

        def generate(self, messages, **kwargs):
            self._tokenizer = RuntimeTokenizer()
            return super().generate(messages, **kwargs)

    result = generate(LazyBackend(), [Message("user", "question")])

    assert result.trajectory.tokenizer_id == "Qwen/Qwen3-4B"


def test_runtime_capture_and_score_share_one_capture_pass_with_position_union():
    backend = RuntimeBackend()
    choices = ChoiceSet(backend.tokenizer, ["A", "B"])
    capture = CaptureSpec(CaptureMode.TEACHER_FORCING_SPARSE, layers=[0], positions="last")
    score = ScoringSpec.post_think(choices)

    result = generate(backend, [Message("user", "question")], capture=capture, score=score)

    assert len(backend.capture_specs) == 1
    runtime_spec = backend.capture_specs[0]
    assert runtime_spec.with_logits is True
    assert runtime_spec.target_text == backend.generated_text
    assert runtime_spec.positions == [
        len(result.trajectory.tokens) - 1,
        result.trajectory.terminal_answer.start - 1,
    ]
    assert result.hiddens.hiddens[0].shape == (1, 1, 2)
    assert result.scores.scores["A"] > result.scores.scores["B"]


def test_runtime_score_without_user_capture_returns_scores_only():
    backend = RuntimeBackend()
    score = ScoringSpec.choices(ChoiceSet(backend.tokenizer, ["A", "B"]), positions="last_token")

    result = generate(backend, [Message("user", "question")], score=score)

    assert len(backend.capture_specs) == 1
    assert backend.capture_specs[0].positions == [len(result.trajectory.tokens) - 1]
    assert result.hiddens is None
    assert result.scores.complete is True


def test_runtime_fails_loudly_when_backend_cannot_satisfy_capture_or_score():
    class GenerateOnly:
        name = "generate-only"
        capabilities = frozenset({"generate"})

        def generate(self, messages, *, temperature=0.0, max_tokens=256, stop=None, seed=None):
            return GenerateResult(text="A", finish_reason="stop")

    score = ScoringSpec.choices(ChoiceSet(RuntimeTokenizer(), ["A", "B"]))
    with pytest.raises(BackendCapabilityError, match="does not support hidden/logit capture"):
        generate(GenerateOnly(), [Message("user", "question")], score=score)
