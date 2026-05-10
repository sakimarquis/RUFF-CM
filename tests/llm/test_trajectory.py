import re

import numpy as np

from ruff_cm.llm import (
    TokenSpan,
    Trajectory,
    chat_template_parser,
    combined,
    select_hiddens,
    select_logits,
    select_steps,
    step_parser,
    thinking_parser,
)
from ruff_cm.llm.trajectory import Segment


class ChatFixtureTokenizer:
    def __init__(self, start="<|{role}|>", end="\n", name_or_path="Qwen/Qwen3-4B"):
        self.start = start
        self.end = end
        self.name_or_path = name_or_path
        self.chat_template = f"{start}{{content}}{end}<think></think>"

    def apply_chat_template(
        self, messages, *, add_generation_prompt=False, tokenize=False, return_dict=False, **kwargs
    ):
        rendered = "".join(
            f"{self.start.format(role=message['role'])}{message['content']}{self.end}" for message in messages
        )
        if add_generation_prompt:
            rendered += self.start.format(role="assistant")
        return list(rendered.encode("utf-8")) if tokenize else rendered

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        encoded = {"input_ids": list(text.encode("utf-8"))}
        if return_offsets_mapping:
            encoded["offset_mapping"] = [(idx, idx + 1) for idx in range(len(text))]
        return encoded

    def decode(self, ids):
        return bytes(ids).decode("utf-8")


def render_chat(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, tokenize=False)


def test_chat_template_parser_recovers_role_segments_for_common_chat_fixtures():
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    fixtures = [
        ChatFixtureTokenizer("<|im_start|>{role}\n", "<|im_end|>\n", "Qwen/Qwen3-4B"),
        ChatFixtureTokenizer("<start_of_turn>{role}\n", "<end_of_turn>\n", "google/gemma-3-1b-it"),
        ChatFixtureTokenizer("<|start_header_id|>{role}<|end_header_id|>\n\n", "<|eot_id|>", "meta-llama/Llama-3.1"),
    ]

    for tokenizer in fixtures:
        traj = Trajectory.parse(render_chat(tokenizer, messages), tokenizer, chat_template_parser(tokenizer))

        assert [segment.name for segment in traj.by_role("assistant")] == ["assistant_1"]
        assert traj.by_role("user")[0].text.endswith("question" + tokenizer.end)
        assert traj.first_assistant_token.positions(traj.context) == [traj.by_name("assistant_1").token_span[0]]


def test_thinking_parser_separates_thinking_content_and_answer_with_family_defaults():
    tokenizer = ChatFixtureTokenizer()
    text = "<|assistant|><think>hidden reasoning</think> final answer\n"

    traj = Trajectory.parse(text, tokenizer, thinking_parser())

    assert traj.by_name("thinking_1").text == "hidden reasoning"
    assert traj.by_name("answer_1").text == " final answer\n"
    assert traj.answer.positions(traj.context) == list(range(*traj.by_name("answer_1").token_span))


def test_thinking_parser_uses_thought_channel_tags_for_gemma_style_templates():
    tokenizer = ChatFixtureTokenizer(name_or_path="google/gemma-4-9b-it")
    tokenizer.chat_template = "<|channel>thought\n{{thought}}<channel|>"
    text = "<start_of_turn>model\n<|channel>thought\nhidden<channel|>visible"

    traj = Trajectory.parse(text, tokenizer, thinking_parser())

    assert traj.by_name("thinking_1").text == "hidden"
    assert traj.by_name("answer_1").text == "visible"


def test_step_parser_aligns_step_segments_to_char_and_token_spans():
    tokenizer = ChatFixtureTokenizer()
    text = "Step 1: x. Step 2: y. Step 3: z."

    traj = Trajectory.parse(text, tokenizer, step_parser())

    assert [segment.meta["index"] for segment in traj.by_meta(kind="step")] == [1, 2, 3]
    assert traj.by_name("step_2").text == "Step 2: y. "
    assert traj.by_name("step_2").token_span == traj.by_name("step_2").char_span


def test_combined_parser_preserves_role_and_overlapping_step_segments():
    tokenizer = ChatFixtureTokenizer("<|im_start|>{role}\n", "<|im_end|>\n")
    messages = [{"role": "assistant", "content": "Step 1: x. Step 2: y."}]

    traj = Trajectory.parse(
        render_chat(tokenizer, messages),
        tokenizer,
        combined(chat_template_parser(tokenizer), step_parser()),
    )

    assert traj.by_role("assistant")
    assert [segment.name for segment in traj.by_meta(kind="step")] == ["step_1", "step_2"]
    assert traj.by_name("step_1").char_span[0] > traj.by_name("assistant_1").char_span[0]


def test_mask_for_composes_role_and_regex_name_filters_with_token_mask_algebra():
    tokenizer = ChatFixtureTokenizer()
    text = "<|assistant|><think>hidden</think> final"
    parser = combined(
        lambda text, tokenizer: [Segment("assistant_1", "assistant", text, (0, len(text)))],
        thinking_parser(),
    )

    traj = Trajectory.parse(text, tokenizer, parser)
    visible_assistant = traj.mask_for(role="assistant") & ~traj.mask_for(name=re.compile(r"thinking_.*"))

    visible_positions = visible_assistant.positions(traj.context)
    assert text.index("final") in visible_positions
    assert text.index("hidden") not in visible_positions
    np.testing.assert_array_equal(
        traj.mask_for("assistant").positions(traj.context),
        traj.mask_for(role="assistant").positions(traj.context),
    )


def test_from_generated_builds_canonical_spans_and_selectors_from_family_rules():
    tokenizer = ChatFixtureTokenizer("<|im_start|>{role}\n", "<|im_end|>\n", "Qwen/Qwen3-4B")
    messages = [{"role": "user", "content": "question"}]
    generated = "<think>hidden reasoning</think> final. Step 1: choose. Step 2: done."

    traj = Trajectory.from_generated(messages, generated, tokenizer)

    assert tokenizer.decode(traj.tokens) == traj.text
    assert traj.tokenizer_id == "Qwen/Qwen3-4B"
    assert traj.attention_mask == tuple(1 for _ in traj.tokens)
    assert traj.thinking_span == TokenSpan(
        traj.text.index("hidden reasoning"),
        traj.text.index("hidden reasoning") + len("hidden reasoning"),
        kind="thinking",
    )
    assert traj.terminal_answer is not None
    assert traj.text[traj.terminal_answer.start : traj.terminal_answer.end] == " final. Step 1: choose. Step 2: done."
    assert [traj.text[span.start : span.end].strip() for span in select_steps(traj, kind="step_header")] == [
        "Step 1: choose.",
        "Step 2: done.",
    ]

    answer_first = select_hiddens(traj, span=traj.terminal_answer, at="first_token").positions(traj.context)
    assert answer_first == [traj.terminal_answer.start]
    assert select_hiddens(traj, role="assistant", at="last_token").positions(traj.context) == [len(traj.tokens) - 1]
    assert select_logits(traj, position="before_answer") == traj.terminal_answer.start - 1


def test_from_streamed_is_equivalent_to_joined_generated_text():
    tokenizer = ChatFixtureTokenizer(name_or_path="Qwen/Qwen3-4B")
    messages = [{"role": "user", "content": "question"}]

    streamed = Trajectory.from_streamed(messages, ["<think>x</think>", " y"], tokenizer)
    joined = Trajectory.from_generated(messages, "<think>x</think> y", tokenizer)

    assert streamed.tokens == joined.tokens
    assert streamed.thinking_span == joined.thinking_span
    assert streamed.terminal_answer == joined.terminal_answer
