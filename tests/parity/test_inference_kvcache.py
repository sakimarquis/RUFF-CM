from __future__ import annotations

import pytest

from ruff_cm.llm.inference.kvcache import (
    HybridCacheAdapter,
    clone_kv,
    concat_kv,
    forward_with_kv_delta,
    is_hybrid_supported,
    kv_seq_len,
    reposition_kv,
    tail_kv,
    to_legacy_kv,
    truncate_kv,
)


def tuple_kv_cache(torch, start: int, length: int, layers: int = 2):
    return tuple(
        (
            torch.arange(start, start + length, dtype=torch.float32).view(1, 1, length, 1) + layer * 100,
            torch.arange(start, start + length, dtype=torch.float32).view(1, 1, length, 1) + layer * 1000,
        )
        for layer in range(layers)
    )


def test_tuple_kv_cache_truncate_concat_clone_and_round_trip():
    torch = pytest.importorskip("torch")
    cache = tuple_kv_cache(torch, 0, 4)

    assert to_legacy_kv(cache) is cache
    truncated = truncate_kv(cache, 2)
    assert kv_seq_len(truncated) == 2
    assert torch.equal(truncated[0][0].flatten(), torch.tensor([0.0, 1.0]))

    tailed = tail_kv(cache, 2)
    assert kv_seq_len(tailed) == 2
    assert torch.equal(tailed[0][0].flatten(), torch.tensor([2.0, 3.0]))

    appended = concat_kv(truncated, tuple_kv_cache(torch, 10, 2))
    assert kv_seq_len(appended) == 4
    assert torch.equal(appended[0][0].flatten(), torch.tensor([0.0, 1.0, 10.0, 11.0]))

    cloned = clone_kv(appended)
    cloned[0][0][..., 0, :] = -1
    assert appended[0][0][..., 0, :].item() == 0.0


def test_reposition_kv_rebases_synthetic_rope_cache():
    torch = pytest.importorskip("torch")

    class Model:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.model = type("Inner", (), {"rotary_emb": self.rotary_emb})()

        def parameters(self):
            yield self.weight

        def rotary_emb(self, _dummy, position_ids):
            angles = position_ids.to(dtype=torch.float32).unsqueeze(-1)
            emb = torch.cat((angles, angles), dim=-1)
            return emb.cos(), emb.sin()

    cache = ((torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2)),)
    repositioned = reposition_kv(Model(), cache, old_start_pos=3, m=2)

    assert kv_seq_len(repositioned) == 2
    assert not torch.equal(repositioned[0][0], cache[0][0])
    assert torch.equal(repositioned[0][1], cache[0][1])


def test_hybrid_adapter_detects_non_attention_native_cache():
    torch = pytest.importorskip("torch")

    class NativeCache:
        def __init__(self):
            self.key_cache = [torch.zeros(1, 1, 2, 1), torch.zeros(1, 2)]
            self.value_cache = [torch.zeros(1, 1, 2, 1), torch.zeros(1, 2)]

        def get_seq_length(self):
            return 2

    cache = NativeCache()
    adapter = HybridCacheAdapter(cache)
    assert is_hybrid_supported(cache) is True
    assert adapter.seq_len == 2
    assert adapter.supports_concat is False


def test_forward_with_kv_delta_feeds_only_new_tokens_for_standard_cache():
    torch = pytest.importorskip("torch")

    class Cache:
        def get_seq_length(self):
            return 2

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {})()
            self.seen_shape = None
            self.seen_cache_position = None

        def forward(self, input_ids, attention_mask, past_key_values, cache_position, use_cache=True):
            self.seen_shape = tuple(input_ids.shape)
            self.seen_cache_position = cache_position.detach().cpu().tolist()
            return type("Output", (), {"past_key_values": "new-cache"})

    model = Model()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    output, new_kv = forward_with_kv_delta(model, base_kv=Cache(), new_input_ids=input_ids)

    assert new_kv == "new-cache"
    assert output.past_key_values == "new-cache"
    assert model.seen_shape == (1, 2)
    assert model.seen_cache_position == [2, 3]
