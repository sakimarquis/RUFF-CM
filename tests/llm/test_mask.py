import numpy as np

from ruff_cm.llm import (
    TokenContext,
    apply_loss_mask,
    at,
    at_positions,
    between_tags,
    build_token_context,
    in_char_range,
    in_span,
    last_n,
    matches,
    not_thinking,
    role,
)
from ruff_cm.llm.prompt.tokenize import tokenize_with_loss_mask


class ChatFixtureTokenizer:
    def __init__(self, start="<|{role}|>", end="\n"):
        self.start = start
        self.end = end

    def apply_chat_template(self, messages, *, add_generation_prompt=False, tokenize=False, return_dict=False):
        rendered = "".join(
            f"{self.start.format(role=message['role'])}{message['content']}{self.end}" for message in messages
        )
        if add_generation_prompt:
            rendered += self.start.format(role="assistant")
        return list(rendered.encode("utf-8")) if tokenize else rendered

    def decode(self, ids):
        return bytes(ids).decode("utf-8")


def reference_loss_mask(tokenizer, messages, *, assistant_role="assistant", ignore_index=-100):
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False)
    labels = [ignore_index] * len(input_ids)
    span_start = 0
    for message_idx, message in enumerate(messages):
        prefix_ids = tokenizer.apply_chat_template(messages[: message_idx + 1], tokenize=True, return_dict=False)
        span_end = len(prefix_ids)
        if message["role"] == assistant_role:
            labels[span_start:span_end] = input_ids[span_start:span_end]
        span_start = span_end
    return labels


def test_tokenize_with_loss_mask_without_mask_keeps_chat_template_fixture_labels():
    messages = [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]
    fixtures = [
        ChatFixtureTokenizer("<|im_start|>{role}\n", "<|im_end|>\n"),
        ChatFixtureTokenizer("<start_of_turn>{role}\n", "<end_of_turn>\n"),
        ChatFixtureTokenizer("[INST:{role}]", "[/INST]"),
    ]

    for tokenizer in fixtures:
        encoded = tokenize_with_loss_mask(tokenizer, messages)

        assert encoded["labels"] == reference_loss_mask(tokenizer, messages)


def test_role_mask_matches_default_assistant_loss_mask():
    tokenizer = ChatFixtureTokenizer()
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]

    default_encoded = tokenize_with_loss_mask(tokenizer, messages)
    mask_encoded = tokenize_with_loss_mask(tokenizer, messages, mask=role("assistant"))

    assert mask_encoded == default_encoded


def test_thinking_region_can_be_selected_inside_assistant_tokens():
    tokenizer = ChatFixtureTokenizer("<|im_start|>{role}\n", "<|im_end|>\n")
    messages = [{"role": "assistant", "content": "<think>hidden</think> final"}]

    encoded = tokenize_with_loss_mask(tokenizer, messages, mask=role("assistant") & ~not_thinking())
    rendered = tokenizer.decode(encoded["input_ids"])
    hidden_start = rendered.index("hidden")
    final_start = rendered.index("final")

    assert encoded["labels"][hidden_start : hidden_start + len("hidden")] == encoded["input_ids"][
        hidden_start : hidden_start + len("hidden")
    ]
    assert encoded["labels"][final_start] == -100


def test_in_char_range_uses_overlap_semantics_for_loss_spans():
    ctx = TokenContext(
        tokens=[10, 11, 12, 13],
        text="abcdef",
        char_offsets=[(0, 1), (1, 3), (3, 5), (5, 6)],
        spans={},
        role_at=[None, None, None, None],
    )

    assert in_char_range(2, 5).positions(ctx) == [1, 2]


def test_position_and_span_constructors_resolve_expected_positions():
    ctx = TokenContext(
        tokens=[1, 2, 3, 1, 2, 4],
        text="abcdef",
        char_offsets=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
        spans={"answer": (2, 5)},
        role_at=["user", "assistant", "assistant", None, "assistant", "user"],
    )

    assert last_n(2).positions(ctx) == [4, 5]
    assert at(1).positions(ctx) == [1]
    assert at_positions([0, 5]).positions(ctx) == [0, 5]
    assert in_span("answer").positions(ctx) == [2, 3, 4]
    assert matches([1, 2]).positions(ctx) == [0, 3]
    assert between_tags([1], [4]).positions(ctx) == [1, 2, 3, 4]


def test_mask_algebra_is_lazy_hashable_and_distributive_on_sampled_contexts():
    masks = [role("assistant"), last_n(3), at_positions([1, 4])]
    composed = (masks[0] & masks[1]) | masks[2]

    assert isinstance(hash(composed), int)

    rng = np.random.default_rng(0)
    for _ in range(3):
        roles = rng.choice(["user", "assistant", None], size=8).tolist()
        ctx = TokenContext(
            tokens=rng.integers(0, 5, size=8).tolist(),
            text="abcdefgh",
            char_offsets=[(idx, idx + 1) for idx in range(8)],
            spans={},
            role_at=roles,
        )
        left = ((masks[0] & masks[1]) | masks[2])(ctx)
        right = ((masks[0] | masks[2]) & (masks[1] | masks[2]))(ctx)

        np.testing.assert_array_equal(left, right)


def test_apply_loss_mask_keeps_selected_ids_only():
    ctx = TokenContext(
        tokens=[7, 8, 9],
        text="abc",
        char_offsets=[(0, 1), (1, 2), (2, 3)],
        spans={},
        role_at=["user", "assistant", "assistant"],
    )

    assert apply_loss_mask([7, 8, 9], role("assistant"), ctx) == [-100, 8, 9]


def test_build_token_context_exposes_message_spans_roles_and_thinking_spans():
    tokenizer = ChatFixtureTokenizer()
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "<think>x</think> y"}]

    ctx = build_token_context(tokenizer, messages)

    assert ctx.tokens
    assert len(ctx.tokens) == len(ctx.char_offsets) == len(ctx.role_at)
    assert "assistant_1" in ctx.spans
    assert "thinking_1" in ctx.spans
    assert role("assistant").positions(ctx)
