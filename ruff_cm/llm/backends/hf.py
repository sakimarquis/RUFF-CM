from __future__ import annotations

from dataclasses import replace
import inspect
from typing import Any

from ruff_cm.configs.thinking import ThinkingConfig
from ruff_cm.llm.inference.thinking import (
    HfThinkingCodec,
    ThinkingBudgetProcessor,
    _AllCaptured,
    _CapturePostThinkLogits,
    recover_uncaptured_logits,
    resolve_thinking_protocol,
)
from ruff_cm.llm.prompt.messages import to_chat_dicts

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
        enable_thinking: bool = False,
        max_thinking_tokens: int | None = None,
        max_answer_tokens: int | None = None,
        batch_size: int = 8,
        thinking: ThinkingConfig | None = None,
    ):
        self.model_id = model_id
        self.dtype = dtype
        self.device = device
        self.attn_implementation = attn_implementation
        self.chat_template = chat_template
        self.trust_remote_code = trust_remote_code
        self.name = name or model_id
        self.enable_thinking = thinking.enable_thinking if thinking is not None else enable_thinking
        self.max_thinking_tokens = thinking.thinking_budget if thinking is not None else max_thinking_tokens
        self.max_answer_tokens = thinking.reasoning_budget if thinking is not None and thinking.reasoning_budget else max_answer_tokens
        self.batch_size = int(batch_size)
        self._model = None
        self._tokenizer = None
        self._thinking_codec: HfThinkingCodec | None = None
        self._thinking_protocol = None
        self._logits_to_keep_kwarg: str | None = None

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
        stop: list[str] | None = None,
        seed: int | None = None,
        thinking_budget: int | None = None,
    ) -> GenerateResult:
        torch = self._torch()
        self._ensure_loaded()
        if seed is not None:
            torch.manual_seed(seed)
        if self.enable_thinking:
            return self._generate_with_thinking(messages, temperature=temperature, max_tokens=max_tokens, thinking_budget=thinking_budget)

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
        if self.enable_thinking:
            inputs = self._encode_messages(messages, enable_thinking=True)
            logits, n_fallback = self._think_then_score(inputs, self._effective_thinking_budget(None))
            scores = choice_set.from_logits(logits[0, :])
            scores.fallback_count = n_fallback
            return scores

        input_ids, attention_mask = self._encode_batch(messages)
        with torch.no_grad():
            outputs = self._model(input_ids, attention_mask=attention_mask, use_cache=False)
        last_index = attention_mask[0].sum() - 1
        return choice_set.from_logits(outputs.logits[0, last_index, :])

    async def score_binary(
        self,
        messages_list: list[list[Message]],
        *,
        thinking_budget: int | None = None,
    ) -> tuple[Any, int]:
        return self.score_binary_sync(messages_list, thinking_budget=thinking_budget)

    def score_binary_sync(
        self,
        messages_list: list[list[Message]],
        *,
        thinking_budget: int | None = None,
    ) -> tuple[Any, int]:
        torch = self._torch()
        self._ensure_loaded()
        if not messages_list:
            return torch.tensor([]), 0
        yes_ids, no_ids = self._resolve_binary_ids()
        inputs = self._encode_messages_batch(messages_list, enable_thinking=self.enable_thinking)
        if self.enable_thinking:
            logits, n_fallback = self._think_then_score(inputs, self._effective_thinking_budget(thinking_budget))
        else:
            logits = self._last_token_logits(inputs)
            n_fallback = 0

        probs = torch.softmax(logits, dim=-1)
        p_yes = probs[:, yes_ids].sum(dim=-1)
        p_no = probs[:, no_ids].sum(dim=-1)
        denom = p_yes + p_no
        fallback = denom == 0
        n_fallback += int(fallback.sum().item())
        return torch.where(fallback, torch.full_like(p_yes, 0.5), p_yes / denom), n_fallback

    async def score_binary_with_shared_thinking(
        self,
        thinking_messages: list[Message],
        messages_list: list[list[Message]],
        *,
        thinking_budget: int | None = None,
    ) -> tuple[Any, int]:
        return self.score_binary_with_shared_thinking_sync(
            thinking_messages,
            messages_list,
            thinking_budget=thinking_budget,
        )

    def score_binary_with_shared_thinking_sync(
        self,
        thinking_messages: list[Message],
        messages_list: list[list[Message]],
        *,
        thinking_budget: int | None = None,
    ) -> tuple[Any, int]:
        torch = self._torch()
        if not messages_list:
            return torch.tensor([]), 0
        thinking_message = self._generate_thinking_message(thinking_messages, self._effective_thinking_budget(thinking_budget))
        continued = _continued_messages(thinking_messages, thinking_message, messages_list)
        chunks = []
        n_fallback = 0
        old_enable_thinking = self.enable_thinking
        self.enable_thinking = False
        try:
            for index in range(0, len(continued), self.batch_size):
                scores, fallback = self.score_binary_sync(continued[index : index + self.batch_size])
                chunks.append(scores)
                n_fallback += fallback
        finally:
            self.enable_thinking = old_enable_thinking
        return torch.cat(chunks), n_fallback

    def generate_batch(
        self,
        messages_list: list[list[Message]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        thinking_budget: int | None = None,
    ) -> list[GenerateResult]:
        if self.enable_thinking:
            return [self.generate(messages, max_tokens=max_tokens, temperature=temperature, thinking_budget=thinking_budget) for messages in messages_list]

        self._ensure_loaded()
        inputs = self._encode_messages_batch(messages_list, enable_thinking=False)
        generated = self._model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=self._tokenizer.pad_token_id,
        )
        prompt_width = inputs["input_ids"].shape[1]
        return [
            GenerateResult(
                text=self._tokenizer.decode(generated[row_idx, prompt_width:], skip_special_tokens=True),
                finish_reason="stop" if generated[row_idx, prompt_width:].shape[0] < max_tokens else "length",
                raw={"token_ids": generated[row_idx]},
            )
            for row_idx in range(len(messages_list))
        ]

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

    def _generate_with_thinking(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        thinking_budget: int | None,
    ) -> GenerateResult:
        protocol = self._ensure_thinking()
        budget = self._effective_thinking_budget(thinking_budget)
        answer_budget = int(self.max_answer_tokens or max_tokens)
        inputs = self._encode_messages(messages, enable_thinking=True)
        processor = (
            ThinkingBudgetProcessor(
                protocol,
                prompt_len=int(inputs["input_ids"].shape[1]),
                thinking_budget=budget,
                answer_budget=0,
            )
            if protocol.supports_forced_close
            else None
        )
        stage1 = self._model.generate(
            **inputs,
            max_new_tokens=budget + len(protocol.close_marker_ids) + 1,
            do_sample=False,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=None,
            logits_processor=[processor] if processor is not None else None,
        )
        states = (
            processor.finalize(stage1)
            if processor is not None
            else {0: {"thinking_truncated": _find_sequence(stage1[0, inputs["input_ids"].shape[1] :].tolist(), tuple(protocol.close_marker_ids)) is None}}
        )
        context_ids, context_mask, thinking_tokens = self._answer_context_from_stage1(inputs, stage1, protocol)
        stage2 = self._model.generate(
            input_ids=context_ids,
            attention_mask=context_mask,
            max_new_tokens=answer_budget,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=self._tokenizer.pad_token_id,
        )
        answer_ids = stage2[0, context_ids.shape[1] :]
        text = self._tokenizer.decode(answer_ids, skip_special_tokens=True)
        finish_reason = "stop" if answer_ids.shape[0] < answer_budget else "length"
        state = states[0]
        return GenerateResult(
            text=text,
            finish_reason=finish_reason,
            raw={"stage1_token_ids": stage1, "stage2_token_ids": stage2, "thinking_state": state},
            thinking_tokens=thinking_tokens,
            max_thinking_tokens=budget,
            thinking_truncated=bool(state["thinking_truncated"]),
        )

    def _answer_context_from_stage1(self, inputs: dict[str, Any], output_ids: Any, protocol: Any):
        prompt_width = int(inputs["input_ids"].shape[1])
        generated = output_ids[0, prompt_width:].tolist()
        close_start = _find_sequence(generated, tuple(protocol.close_marker_ids))
        if close_start is None:
            return self._force_answer_context(output_ids, _full_attention_mask(inputs, output_ids), protocol.close_marker_ids) + (len(generated),)

        close_end = prompt_width + close_start + len(protocol.close_marker_ids)
        context_ids = output_ids[:, :close_end]
        context_mask = _full_attention_mask(inputs, output_ids)[:, :close_end]
        return context_ids, context_mask, close_start

    def _think_then_score(self, inputs: dict[str, Any], thinking_budget: int):
        torch = self._torch()
        protocol = self._ensure_thinking()
        batch_size = int(inputs["input_ids"].shape[0])
        prompt_width = int(inputs["input_ids"].shape[1])
        capture = _CapturePostThinkLogits(protocol.close_marker_ids, batch_size, prompt_width=prompt_width)
        processors = []
        if protocol.supports_forced_close:
            processors.append(ThinkingBudgetProcessor(protocol, prompt_len=prompt_width, thinking_budget=thinking_budget))
        processors.append(capture)
        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=thinking_budget + len(protocol.close_marker_ids) + 1,
            do_sample=False,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=None,
            logits_processor=processors,
            stopping_criteria=[_AllCaptured(capture)],
        )
        logits_by_row = [capture.captured.get(row_idx) for row_idx in range(batch_size)]
        missing = [row_idx for row_idx, logits in enumerate(logits_by_row) if logits is None]
        if missing:
            recovered = recover_uncaptured_logits(self._model, inputs, output_ids, missing, protocol)
            for row_idx, logits in recovered.items():
                logits_by_row[row_idx] = logits
        return torch.stack(logits_by_row), len(missing)

    def _generate_thinking_message(self, messages: list[Message], thinking_budget: int) -> Message:
        protocol = self._ensure_thinking()
        inputs = self._encode_messages(messages, enable_thinking=True)
        processor = (
            ThinkingBudgetProcessor(
                protocol,
                prompt_len=int(inputs["input_ids"].shape[1]),
                thinking_budget=thinking_budget,
                answer_budget=0,
            )
            if protocol.supports_forced_close
            else None
        )
        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=thinking_budget + len(protocol.close_marker_ids) + 1,
            do_sample=False,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=None,
            logits_processor=[processor] if processor is not None else None,
        )
        prompt_width = int(inputs["input_ids"].shape[1])
        think_ids, _answer_ids = self._thinking_codec.split_think_answer_safe(output_ids[0, prompt_width:].tolist())
        thinking_text = self._tokenizer.decode(think_ids, skip_special_tokens=True).strip()
        return Message("assistant", self._format_thinking_block(thinking_text))

    def _format_thinking_block(self, thinking_text: str) -> str:
        protocol = self._ensure_thinking()
        separator = "" if protocol.open_marker_text.endswith("\n") else "\n"
        return f"{protocol.open_marker_text}{separator}{thinking_text}{protocol.close_marker_text}\n"

    def _force_answer_context(self, input_ids, attention_mask=None, close_ids=None):
        torch = self._torch()
        marker_ids = list(close_ids if close_ids is not None else self._ensure_thinking().close_marker_ids)
        close_tensor = torch.tensor([marker_ids], dtype=input_ids.dtype, device=input_ids.device).expand(input_ids.shape[0], -1)
        forced_ids = torch.cat([input_ids, close_tensor], dim=1)
        if attention_mask is None:
            return forced_ids, None
        close_mask = attention_mask.new_ones((attention_mask.shape[0], len(marker_ids)))
        return forced_ids, torch.cat([attention_mask, close_mask], dim=1)

    def _letter_result(self, logits, choices: list[str] | None = None, **metadata) -> GenerateResult:
        from ruff_cm.llm.extract_answer.choice import build_letter_token_ids, compute_letter_log_probs

        letters = choices or [chr(ord("A") + index) for index in range(26)]
        token_map = build_letter_token_ids(self._tokenizer, letters, variants=[])
        log_probs = compute_letter_log_probs(logits, token_map)
        answer = max(log_probs, key=log_probs.get)
        return GenerateResult(text=answer, finish_reason="stop", raw={"log_probs": log_probs}, **metadata)

    def _last_token_logits(self, inputs: dict[str, Any]):
        kwargs = dict(inputs)
        if self._logits_to_keep_kwarg is not None:
            kwargs[self._logits_to_keep_kwarg] = 1
        return self._model(**kwargs).logits[:, -1, :]

    def _resolve_logits_to_keep_kwarg(self) -> str | None:
        forward = getattr(self._model, "forward", None)
        if forward is None:
            return None
        params = inspect.signature(forward).parameters
        if "logits_to_keep" in params:
            return "logits_to_keep"
        if "num_logits_to_keep" in params:
            return "num_logits_to_keep"
        return None

    def _resolve_binary_ids(self) -> tuple[list[int], list[int]]:
        yes_ids = _first_token_ids(self._tokenizer, ["Yes", "yes", "YES", " Yes", " yes"])
        no_ids = _first_token_ids(self._tokenizer, ["No", "no", "NO", " No", " no"])
        return yes_ids, no_ids

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
            self._logits_to_keep_kwarg = self._resolve_logits_to_keep_kwarg()

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        if self.chat_template is not None:
            self._tokenizer.chat_template = self.chat_template

    def _ensure_thinking(self):
        self._ensure_loaded()
        if self._thinking_protocol is None or self._thinking_codec is None:
            cfg = ThinkingConfig(True, self._effective_thinking_budget(None), None, 0, None, 0, "_thinking")
            self._thinking_protocol = resolve_thinking_protocol(self._tokenizer, cfg)
            self._thinking_codec = HfThinkingCodec(self._tokenizer, self._thinking_protocol)
        return self._thinking_protocol

    def _effective_thinking_budget(self, thinking_budget: int | None) -> int:
        if thinking_budget is not None:
            return int(thinking_budget)
        if self.max_thinking_tokens is not None:
            return int(self.max_thinking_tokens)
        if self._thinking_protocol is not None and self._thinking_protocol.max_thinking_tokens is not None:
            return int(self._thinking_protocol.max_thinking_tokens)
        return 256

    def _render_chat(self, messages: list[Message], *, enable_thinking: bool | None = None) -> str:
        chat_messages = to_chat_dicts(messages)
        if self._tokenizer.chat_template is not None:
            kwargs = {"tokenize": False, "add_generation_prompt": True}
            if enable_thinking:
                kwargs["enable_thinking"] = True
            return self._tokenizer.apply_chat_template(chat_messages, **kwargs)
        lines = [f"{message['role']}: {message['content']}" for message in chat_messages]
        lines.append("assistant:")
        return "\n".join(lines)

    def _encode_text(self, text: str | list[str], *, padding: bool = False) -> dict[str, Any]:
        encoded = self._tokenizer(text, return_tensors="pt", padding=padding)
        input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        attention_mask = encoded["attention_mask"] if isinstance(encoded, dict) else encoded.attention_mask
        device = _model_device(self._model, self.device)
        return {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)}

    def _encode_messages(self, messages: list[Message], *, enable_thinking: bool | None = None) -> dict[str, Any]:
        return self._encode_text(self._render_chat(messages, enable_thinking=enable_thinking))

    def _encode_messages_batch(self, messages: list[list[Message]], *, enable_thinking: bool | None = None) -> dict[str, Any]:
        old_padding_side = getattr(self._tokenizer, "padding_side", None)
        self._tokenizer.padding_side = "left"
        try:
            return self._encode_text([self._render_chat(sample, enable_thinking=enable_thinking) for sample in messages], padding=True)
        finally:
            if old_padding_side is not None:
                self._tokenizer.padding_side = old_padding_side

    def _encode_batch(self, messages: list[Message] | list[list[Message]], target_text: str | list[str] | None = None):
        self._ensure_loaded()
        batch_messages = [messages] if messages and _is_message(messages[0]) else messages
        prompts = [self._render_chat(sample) for sample in batch_messages]
        if target_text is not None:
            targets = [target_text] if isinstance(target_text, str) else target_text
            prompts = [prompt + target for prompt, target in zip(prompts, targets)]
        encoded = self._encode_text(prompts, padding=True)
        return encoded["input_ids"], encoded["attention_mask"]

    def _torch(self):
        import torch

        return torch


def _spec_with_non_pad_last_positions(spec: Any, attention_mask: Any) -> Any:
    positions = [[int(row.nonzero()[-1].item())] for row in attention_mask]
    return replace(spec, positions=positions)


def _trim_stop(text: str, stop: list[str]) -> tuple[str, bool]:
    stop_positions = [text.find(stop_text) for stop_text in stop if stop_text in text]
    return (text[: min(stop_positions)], True) if stop_positions else (text, False)


def _find_sequence(items: list[int], needle: tuple[int, ...]) -> int | None:
    width = len(needle)
    if width == 0:
        return None
    for idx in range(len(items) - width + 1):
        if tuple(items[idx : idx + width]) == needle:
            return idx
    return None


def _full_attention_mask(inputs: dict[str, Any], output_ids: Any):
    attention_mask = inputs["attention_mask"]
    generated_width = output_ids.shape[1] - attention_mask.shape[1]
    if generated_width <= 0:
        return attention_mask[:, : output_ids.shape[1]]
    return _cat_tensors(attention_mask, attention_mask.new_ones((attention_mask.shape[0], generated_width)))


def _cat_tensors(left, right):
    import torch

    return torch.cat([left, right], dim=1)


def _first_token_ids(tokenizer, variants: list[str]) -> list[int]:
    token_ids = []
    for variant in variants:
        ids = list(tokenizer.encode(variant, add_special_tokens=False))
        if ids and ids[0] not in token_ids:
            token_ids.append(int(ids[0]))
    return token_ids


def _continued_messages(
    thinking_messages: list[Message],
    thinking_message: Message,
    messages_list: list[list[Message]],
) -> list[list[Message]]:
    return [list(thinking_messages) + [thinking_message, messages[-1]] for messages in messages_list]


def _is_message(value: Any) -> bool:
    return isinstance(value, Message) or (isinstance(value, dict) and "role" in value and "content" in value)


def _model_device(model: Any, fallback: str):
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return fallback
