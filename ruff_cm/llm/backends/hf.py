from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..hooks import CaptureMode, HiddenCapture
from .base import BackendCapabilityError, CaptureResult, ChoiceScores, GenerateResult, Message


class HfBackend:
    capabilities = frozenset({"generate", "score_exact", "hidden_prefill", "hidden_teacher_forcing_sparse"})

    def __init__(
        self,
        model_id: str,
        *,
        dtype: str = "bfloat16",
        device: str = "cuda",
        attn_implementation: str | None = "sdpa",
        chat_template: str | None = None,
        trust_remote_code: bool = False,
        name: str | None = None,
    ):
        self.model_id = model_id
        self.dtype = dtype
        self.device = device
        self.attn_implementation = attn_implementation
        self.chat_template = chat_template
        self.trust_remote_code = trust_remote_code
        self.name = name or model_id
        self._model = None
        self._tokenizer = None

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> GenerateResult:
        torch = self._torch()
        self._ensure_loaded()
        if seed is not None:
            torch.manual_seed(seed)

        input_ids, attention_mask = self._encode_batch(messages)
        generated = self._model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=self._tokenizer.pad_token_id,
        )
        new_ids = generated[0, input_ids.shape[1] :]
        text = self._tokenizer.decode(new_ids, skip_special_tokens=True)
        stopped = False
        if stop is not None:
            text, stopped = _trim_stop(text, stop)
        finish_reason = "stop" if stopped or new_ids.shape[0] < max_tokens else "length"
        return GenerateResult(text=text, finish_reason=finish_reason, raw={"token_ids": generated})

    def score_choices(self, messages: list[Message], choice_set: Any) -> ChoiceScores:
        torch = self._torch()
        self._ensure_loaded()
        input_ids, attention_mask = self._encode_batch(messages)
        with torch.no_grad():
            outputs = self._model(input_ids, attention_mask=attention_mask, use_cache=False)
        last_index = attention_mask[0].sum() - 1
        return choice_set.from_logits(outputs.logits[0, last_index, :])

    def capture(self, messages: list[Message] | list[list[Message]], spec: Any) -> CaptureResult:
        if spec.mode == CaptureMode.GENERATE_STEPS:
            raise BackendCapabilityError("HfBackend does not support generate-step hidden capture")

        torch = self._torch()
        self._ensure_loaded()
        target_text = spec.target_text if spec.mode == CaptureMode.TEACHER_FORCING_SPARSE else None
        input_ids, attention_mask = self._encode_batch(messages, target_text=target_text)
        capture_spec = _spec_with_non_pad_last_positions(spec, attention_mask) if spec.positions == "last" else spec
        with HiddenCapture(self._model, capture_spec) as capture:
            with torch.no_grad():
                outputs = self._model(input_ids, attention_mask=attention_mask, use_cache=False)
        return capture.collect(token_ids=input_ids, logits=outputs.logits)

    def _ensure_loaded(self) -> None:
        if self._tokenizer is None or self._model is None:
            torch = self._torch()
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=self.trust_remote_code)
            model_kwargs = {"torch_dtype": getattr(torch, self.dtype), "trust_remote_code": self.trust_remote_code}
            if self.attn_implementation is not None:
                model_kwargs["attn_implementation"] = self.attn_implementation
            self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs).to(self.device)
            self._model.eval()

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        if self.chat_template is not None:
            self._tokenizer.chat_template = self.chat_template

    def _render_chat(self, messages: list[Message]) -> str:
        if self._tokenizer.chat_template is not None:
            return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        lines = [f"{message.role}: {message.content}" for message in messages]
        lines.append("assistant:")
        return "\n".join(lines)

    def _encode_batch(self, messages: list[Message] | list[list[Message]], target_text: str | list[str] | None = None):
        self._ensure_loaded()
        batch_messages = [messages] if messages and isinstance(messages[0], Message) else messages
        prompts = [self._render_chat(sample) for sample in batch_messages]
        if target_text is not None:
            targets = [target_text] if isinstance(target_text, str) else target_text
            prompts = [prompt + target for prompt, target in zip(prompts, targets)]
        encoded = self._tokenizer(prompts, return_tensors="pt", padding=True)
        return encoded.input_ids.to(self.device), encoded.attention_mask.to(self.device)

    def _torch(self):
        import torch

        return torch


def _spec_with_non_pad_last_positions(spec: Any, attention_mask: Any) -> Any:
    positions = [[int(row.nonzero()[-1].item())] for row in attention_mask]
    return replace(spec, positions=positions)


def _trim_stop(text: str, stop: list[str]) -> tuple[str, bool]:
    stop_positions = [text.find(stop_text) for stop_text in stop if stop_text in text]
    return (text[: min(stop_positions)], True) if stop_positions else (text, False)
