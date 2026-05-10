from __future__ import annotations

import pytest

from ruff_cm.llm.inference.kvcache import reposition_kv


class RotaryModel:
    def __init__(self, torch, rotary_dim: int, rope_base: float = 10_000.0):
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.model = type("Inner", (), {"rotary_emb": self._rotary_emb})()
        self.rotary_dim = rotary_dim
        self.rope_base = rope_base
        self.torch = torch

    def parameters(self):
        yield self.weight

    def _rotary_emb(self, _dummy, position_ids):
        positions = position_ids.to(dtype=self.torch.float32)
        dim_index = self.torch.arange(0, self.rotary_dim, 2, device=positions.device, dtype=self.torch.float32)
        inv_freq = 1.0 / (self.rope_base ** (dim_index / self.rotary_dim))
        freqs = positions[..., None] * inv_freq
        emb = self.torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def split_rotate_half(torch, x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_split_rope(torch, x, cos, sin):
    return x * cos.unsqueeze(1) + split_rotate_half(torch, x) * sin.unsqueeze(1)


def hb_reference_reposition(torch, key, cos_old, sin_old, cos_new, sin_new):
    rotary_dim = cos_old.shape[-1]
    key_rot, key_pass = key[..., :rotary_dim].float(), key[..., rotary_dim:]
    key_unrot = key_rot * cos_old.unsqueeze(1).float() - split_rotate_half(torch, key_rot) * sin_old.unsqueeze(1).float()
    key_new = key_unrot * cos_new.unsqueeze(1).float() + split_rotate_half(torch, key_unrot) * sin_new.unsqueeze(1).float()
    return torch.cat([key_new.to(key.dtype), key_pass], dim=-1)


def test_reposition_kv_matches_split_half_reference_byte_for_byte():
    torch = pytest.importorskip("torch")
    model = RotaryModel(torch, rotary_dim=4)
    old_start_pos = 5
    seq_len = 3

    base_rotary = torch.arange(1, 13, dtype=torch.float32).view(1, 1, seq_len, 4) / 10
    passthrough = torch.arange(20, 26, dtype=torch.float32).view(1, 1, seq_len, 2) / 10
    value = torch.arange(30, 48, dtype=torch.float32).view(1, 1, seq_len, 6) / 10

    old_pos = torch.arange(old_start_pos, old_start_pos + seq_len).unsqueeze(0)
    new_pos = torch.arange(seq_len).unsqueeze(0)
    cos_old, sin_old = model.model.rotary_emb(torch.zeros(1), old_pos)
    cos_new, sin_new = model.model.rotary_emb(torch.zeros(1), new_pos)

    old_key = torch.cat([apply_split_rope(torch, base_rotary, cos_old, sin_old), passthrough], dim=-1)
    expected_key = hb_reference_reposition(torch, old_key, cos_old, sin_old, cos_new, sin_new)

    repositioned = reposition_kv(model, ((old_key, value),), old_start_pos=old_start_pos, m=seq_len)

    torch.testing.assert_close(repositioned[0][0], expected_key, rtol=0, atol=0)
    assert torch.equal(repositioned[0][1], value)
