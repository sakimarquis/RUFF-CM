import pytest

torch = pytest.importorskip("torch")

from ruff_cm.llm.extract_hiddens.pooling import pool_span
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
