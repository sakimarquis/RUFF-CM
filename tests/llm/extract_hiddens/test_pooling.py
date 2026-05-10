import pytest

torch = pytest.importorskip("torch")

from ruff_cm.llm.extract_hiddens.pooling import pool_for, pool_layered, pool_span, pool_spans
from ruff_cm.llm.trajectory import TokenSpan


def test_pool_span_mean_over_explicit_range():
    # hidden: (seq=5, hidden=2). Mean over positions [1, 4) averages rows 1, 2, 3.
    hidden = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [9.0, 9.0],
        ]
    )
    pooled = pool_span(hidden, TokenSpan(1, 4), "mean")
    assert pooled.shape == (2,)
    assert torch.allclose(pooled, torch.tensor([3.0, 4.0]))


def test_pool_span_last_returns_endpoint_minus_one():
    hidden = torch.arange(10, dtype=torch.float32).view(5, 2)
    pooled = pool_span(hidden, TokenSpan(1, 4), "last")
    assert torch.equal(pooled, hidden[3])


def test_pool_span_first_returns_start():
    hidden = torch.arange(10, dtype=torch.float32).view(5, 2)
    pooled = pool_span(hidden, TokenSpan(1, 4), "first")
    assert torch.equal(pooled, hidden[1])


def test_pool_span_accepts_tuple_span():
    hidden = torch.arange(10, dtype=torch.float32).view(5, 2)
    pooled = pool_span(hidden, (1, 4), "mean")
    assert torch.allclose(pooled, hidden[1:4].mean(dim=0))


def test_pool_span_supports_batched_hidden():
    hidden = torch.arange(30, dtype=torch.float32).view(2, 5, 3)
    pooled = pool_span(hidden, TokenSpan(1, 4), "mean")
    assert pooled.shape == (2, 3)
    assert torch.allclose(pooled[0], hidden[0, 1:4].mean(dim=0))
    assert torch.allclose(pooled[1], hidden[1, 1:4].mean(dim=0))


def test_pool_span_mean_preserves_bfloat16_dtype():
    hidden = torch.randn(5, 4, dtype=torch.bfloat16)
    pooled = pool_span(hidden, TokenSpan(1, 4), "mean")
    assert pooled.dtype == torch.bfloat16
    expected = hidden[1:4].float().mean(dim=0).to(torch.bfloat16)
    assert torch.equal(pooled, expected)


def test_pool_span_rejects_unknown_mode():
    hidden = torch.zeros(3, 2)
    with pytest.raises(ValueError):
        pool_span(hidden, TokenSpan(0, 2), "max")


def test_pool_span_rejects_empty_span():
    hidden = torch.zeros(3, 2)
    with pytest.raises(ValueError):
        pool_span(hidden, TokenSpan(2, 2), "mean")


def test_pool_spans_returns_stacked_tensor():
    hidden = torch.arange(20, dtype=torch.float32).view(10, 2)
    spans = [TokenSpan(0, 3), TokenSpan(3, 6), TokenSpan(7, 10)]
    pooled = pool_spans(hidden, spans, "mean")
    assert pooled.shape == (3, 2)
    assert torch.allclose(pooled[0], hidden[0:3].mean(dim=0))
    assert torch.allclose(pooled[1], hidden[3:6].mean(dim=0))
    assert torch.allclose(pooled[2], hidden[7:10].mean(dim=0))


def test_pool_spans_supports_batched_hidden():
    hidden = torch.arange(60, dtype=torch.float32).view(2, 10, 3)
    spans = [TokenSpan(0, 3), TokenSpan(3, 6)]
    pooled = pool_spans(hidden, spans, "mean")
    assert pooled.shape == (2, 2, 3)
    assert torch.allclose(pooled[0, 0], hidden[0, 0:3].mean(dim=0))


def test_pool_spans_empty_input_raises():
    hidden = torch.zeros(5, 2)
    with pytest.raises(ValueError):
        pool_spans(hidden, [], "mean")


def test_pool_layered_applies_pool_per_layer():
    layers = {
        0: torch.arange(10, dtype=torch.float32).view(5, 2),
        4: torch.arange(10, 20, dtype=torch.float32).view(5, 2),
    }
    pooled = pool_layered(layers, TokenSpan(1, 4), "mean")
    assert set(pooled) == {0, 4}
    assert torch.allclose(pooled[0], layers[0][1:4].mean(dim=0))
    assert torch.allclose(pooled[4], layers[4][1:4].mean(dim=0))


def test_pool_layered_preserves_layer_keys():
    layers = {3: torch.zeros(4, 2), 7: torch.zeros(4, 2)}
    pooled = pool_layered(layers, TokenSpan(0, 2), "first")
    assert sorted(pooled) == [3, 7]


def test_pool_for_resolves_assistant_role():
    from ruff_cm.llm.mask import TokenContext
    from ruff_cm.llm.trajectory import Segment, Trajectory

    text = "user-q assistant-a"
    tokens = (1, 2, 3, 4, 5, 6, 7)
    segments = (
        Segment(name="user_1", role="user", text="user-q", char_span=(0, 6), token_span=(0, 3)),
        Segment(name="assistant_1", role="assistant", text="assistant-a", char_span=(7, 18), token_span=(3, 7)),
    )
    context = TokenContext(tokens=list(tokens), text=text, char_offsets=[(0, 0)] * 7, spans={}, role_at=[None] * 7)
    traj = Trajectory(text=text, tokens=tokens, segments=segments, context=context)

    hidden = torch.arange(7 * 4, dtype=torch.float32).view(7, 4)
    pooled = pool_for(traj, hidden, "assistant", "mean")
    assert pooled.shape == (4,)
    assert torch.allclose(pooled, hidden[3:7].mean(dim=0))


def test_pool_for_rejects_role_with_multiple_spans():
    from ruff_cm.llm.mask import TokenContext
    from ruff_cm.llm.trajectory import Segment, Trajectory

    segments = (
        Segment(name="user_1", role="user", text="u1", char_span=(0, 2), token_span=(0, 1)),
        Segment(name="assistant_1", role="assistant", text="a1", char_span=(2, 4), token_span=(1, 2)),
        Segment(name="user_2", role="user", text="u2", char_span=(4, 6), token_span=(2, 3)),
        Segment(name="assistant_2", role="assistant", text="a2", char_span=(6, 8), token_span=(3, 4)),
    )
    context = TokenContext(tokens=[1, 2, 3, 4], text="u1a1u2a2", char_offsets=[(0, 0)] * 4, spans={}, role_at=[None] * 4)
    traj = Trajectory(text="u1a1u2a2", tokens=(1, 2, 3, 4), segments=segments, context=context)

    hidden = torch.zeros(4, 2)
    with pytest.raises(ValueError, match="multiple"):
        pool_for(traj, hidden, "assistant", "mean")


def test_pool_for_unknown_selector_raises():
    from ruff_cm.llm.mask import TokenContext
    from ruff_cm.llm.trajectory import Segment, Trajectory

    segments = (
        Segment(name="assistant_1", role="assistant", text="a", char_span=(0, 1), token_span=(0, 1)),
    )
    context = TokenContext(tokens=[1], text="a", char_offsets=[(0, 1)], spans={}, role_at=["assistant"])
    traj = Trajectory(text="a", tokens=(1,), segments=segments, context=context)

    hidden = torch.zeros(1, 2)
    with pytest.raises(KeyError):
        pool_for(traj, hidden, "nonexistent_segment", "mean")
