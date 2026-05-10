import pytest

from ruff_cm.llm.backends import Message
from ruff_cm.llm.prompt.compose import (
    compose_preamble,
    filter_system_messages,
    fmt_comma,
    fmt_json,
    fmt_numbered,
    render_template,
)


def test_prompt_formatters_are_stable():
    assert fmt_comma(["a", "b", "c"]) == "a, b, c"
    assert fmt_numbered(["alpha", "beta"], start=3) == "3. alpha\n4. beta"
    assert fmt_json({"b": 2, "a": 1}) == '{\n  "a": 1,\n  "b": 2\n}'


def test_render_template_replaces_slots_and_rejects_unfilled_slots():
    assert render_template("<<x>> + <<y>>", {"x": "a", "y": "b"}) == "a + b"

    with pytest.raises(KeyError, match="missing"):
        render_template("<<x>> + <<missing>>", {"x": "a"})


def test_compose_preamble_returns_system_and_user_messages():
    messages = compose_preamble("shared context", ["question", "answer format"])

    assert messages == [
        Message(role="system", content="shared context"),
        Message(role="user", content="question\n\nanswer format"),
    ]


def test_filter_system_messages_collapses_adjacent_system_messages():
    messages = [
        Message(role="system", content="first"),
        Message(role="system", content="second"),
        Message(role="user", content="question"),
        Message(role="system", content="tail"),
    ]

    assert filter_system_messages(messages) == [
        Message(role="system", content="first\n\nsecond"),
        Message(role="user", content="question"),
        Message(role="system", content="tail"),
    ]
