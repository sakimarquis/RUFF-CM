from __future__ import annotations

import pytest

from ruff_cm.llm.hooks import CaptureMode, CaptureSpec, HiddenCapture


def old_last_token_hook_fixture(model, input_ids, layer_idx):
    torch = pytest.importorskip("torch")
    layers = model.transformer.h
    captured = {}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured[layer_idx] = hidden[:, -1:, :].detach()

    handle = layers[layer_idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            outputs = model(input_ids, use_cache=False)
    finally:
        handle.remove()
    return captured[layer_idx], outputs.logits[:, -1:, :]


@pytest.mark.hf
@pytest.mark.parity
def test_hidden_capture_matches_old_last_token_hook_fixture(tiny_hf):
    torch = pytest.importorskip("torch")
    input_ids = tiny_hf["tokenizer"]("hello world", return_tensors="pt").input_ids
    old_hidden, old_logits = old_last_token_hook_fixture(tiny_hf["model"], input_ids, 0)
    spec = CaptureSpec(mode=CaptureMode.PREFILL, layers=[0], positions="last", with_logits=True)
    with HiddenCapture(tiny_hf["model"], spec) as capture:
        with torch.no_grad():
            outputs = tiny_hf["model"](input_ids, use_cache=False)
    result = capture.collect(token_ids=input_ids, logits=outputs.logits)
    assert torch.equal(result.hiddens[0], old_hidden)
    assert torch.equal(result.logits, old_logits)
