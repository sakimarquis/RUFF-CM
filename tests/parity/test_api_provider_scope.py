from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace
from unittest.mock import MagicMock

from ruff_cm.llm.backends.api import ApiBackend, api_run_policy
from ruff_cm.llm.backends.base import Message
from ruff_cm.llm.choice import ChoiceSet


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        return {"A": [0], "B": [1], "Yes": [2], "No": [3]}[text.strip()]


def _response(text: str = "ok", *, top_logprobs=None):
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message, finish_reason="stop", logprobs=None)
    if top_logprobs is not None:
        entries = [SimpleNamespace(token=token, logprob=logprob) for token, logprob in top_logprobs]
        choice.logprobs = SimpleNamespace(content=[SimpleNamespace(top_logprobs=entries)])
    return SimpleNamespace(choices=[choice], model_dump=lambda: {"choices": [{"message": {"content": text}}]})


def test_google_gemini_lower_request_uses_vertex_shape():
    backend = ApiBackend("google/gemini-3.1-flash-lite-preview", provider="google_cloud", api_key="key", client=MagicMock())
    body = backend.adapter.lower_request([Message("user", "hi")], max_tokens=3)
    assert body["model"] == "gemini-3.1-flash-lite-preview"
    assert body["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    assert body["config"]["maxOutputTokens"] == 65_535


def test_anthropic_vertex_generate_uses_messages_endpoint():
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(content=[SimpleNamespace(type="text", text=" done ")])
    backend = ApiBackend("anthropic/claude-sonnet-4", provider="anthropic_vertex", client=client)
    result = asyncio.run(backend.generate_async([Message("user", "hi")], max_tokens=8))
    assert result.text == "done"
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-sonnet-4"
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_anthropic_cache_control_is_system_message_only():
    backend = ApiBackend("anthropic/claude-sonnet-4", provider="openrouter", api_key="key", client=MagicMock())
    body = backend._chat_body([Message("system", "rules"), Message("user", "hi")], max_tokens=1)
    assert body["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert body["messages"][1] == {"role": "user", "content": "hi"}


def test_local_guided_choice_extra_body_is_provider_extension():
    backend = ApiBackend("qwen3", provider="local", client=MagicMock())
    body = backend._chat_body([Message("user", "pick")], max_tokens=1, guided_choice=["A", "B"])
    assert body["extra_body"]["guided_choice"] == ["A", "B"]


def test_text_marker_thinking_uses_two_stage_answer_only_call():
    client = MagicMock()
    client.chat.completions.create.side_effect = [_response("reasoning"), _response("answer")]
    backend = ApiBackend("qwen3", provider="local", client=client, enable_thinking=True)
    result = asyncio.run(backend.generate_async([Message("user", "solve")], max_tokens=4))
    assert result.text == "answer"
    first, second = client.chat.completions.create.call_args_list
    assert first.kwargs["stop"] == ["</think>"]
    assert second.kwargs["extra_body"]["continue_final_message"] is True
    assert second.kwargs["messages"][-1]["content"] == "<think>reasoning</think>"


def test_score_choices_strips_bpe_prefixes_before_choice_mapping():
    client = MagicMock()
    client.chat.completions.create.return_value = _response(top_logprobs=[("\u0120A", -0.1), ("B", -2.0)])
    backend = ApiBackend("gpt-4o", provider="openai", api_key="key", client=client)
    scores = backend.score_choices([Message("user", "pick")], ChoiceSet(FakeTokenizer(), ["A", "B"]))
    assert scores.complete is True
    assert scores.scores["A"] > scores.scores["B"]


def test_score_binary_aggregates_yes_no_top_logprobs():
    client = MagicMock()
    client.chat.completions.create.return_value = _response(top_logprobs=[("\u0120Yes", math.log(0.8)), ("No", math.log(0.2))])
    backend = ApiBackend("gpt-4o", provider="openai", api_key="key", client=client)
    scores, fallback = asyncio.run(backend.score_binary([[Message("user", "yes?")]]))
    assert fallback == 0
    assert abs(float(scores[0]) - 0.8) < 1e-6


def test_openai_batch_collect_returns_custom_id_order():
    client = MagicMock()
    client.batches.retrieve.return_value = SimpleNamespace(
        id="batch-1",
        status="completed",
        error_file_id=None,
        output_file_id="out",
    )
    client.files.content.return_value = SimpleNamespace(
        text="\n".join(
            [
                '{"custom_id":"b","response":{"status_code":200,"body":{"choices":[{"message":{"content":"second"}}]}}}',
                '{"custom_id":"a","response":{"status_code":200,"body":{"choices":[{"message":{"content":"first"}}]}}}',
            ]
        )
    )
    backend = ApiBackend("gpt-4o", provider="openai", api_key="key", client=client)
    results = asyncio.run(backend.collect_batch("batch-1", custom_ids=["a", "b"]))
    assert [result.text for result in results] == ["first", "second"]


def test_api_run_policy_batches_large_openai_and_flexes_small_openai():
    assert api_run_policy("openai", 100, batch_min_trials=20) == "batch"
    assert api_run_policy("openai", 2, batch_min_trials=20) == "flex"
    assert api_run_policy("openrouter", 100, batch_min_trials=20) == "immediate"
