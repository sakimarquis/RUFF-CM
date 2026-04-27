from __future__ import annotations

import pytest

from ruff_cm.llm.backends.base import CaptureResult
from ruff_cm.llm.hooks import CaptureMode, CaptureSpec, HiddenCapture, UnsupportedArchitectureError


def test_capture_spec_defaults():
    spec = CaptureSpec(mode=CaptureMode.PREFILL)
    assert spec.layers == "all"
    assert spec.positions == "last"
    assert spec.with_logits is False


def test_generate_steps_is_representable_but_not_phase1_backend_capability():
    spec = CaptureSpec(mode=CaptureMode.GENERATE_STEPS)
    assert spec.mode.value == "generate_steps"


@pytest.mark.hf
def test_hidden_capture_prefill_last_token(tiny_hf):
    torch = pytest.importorskip("torch")
    spec = CaptureSpec(mode=CaptureMode.PREFILL, layers=[0], positions="last", with_logits=True)
    input_ids = tiny_hf["tokenizer"]("hello world", return_tensors="pt").input_ids
    with HiddenCapture(tiny_hf["model"], spec) as capture:
        with torch.no_grad():
            outputs = tiny_hf["model"](input_ids, use_cache=False)
    result = capture.collect(token_ids=input_ids, logits=outputs.logits)
    assert isinstance(result, CaptureResult)
    assert set(result.hiddens) == {0}
    assert result.hiddens[0].shape[:2] == (1, 1)
    assert result.logits.shape[:2] == (1, 1)


@pytest.mark.hf
def test_hidden_capture_prefill_explicit_positions(tiny_hf):
    torch = pytest.importorskip("torch")
    spec = CaptureSpec(mode=CaptureMode.PREFILL, layers=[0], positions=[0, 1])
    input_ids = tiny_hf["tokenizer"]("hello world", return_tensors="pt").input_ids
    with HiddenCapture(tiny_hf["model"], spec) as capture:
        with torch.no_grad():
            tiny_hf["model"](input_ids, use_cache=False)
    result = capture.collect(token_ids=input_ids)
    assert result.hiddens[0].shape[1] == 2


@pytest.mark.hf
def test_hidden_capture_per_sample_sparse_positions(tiny_hf):
    torch = pytest.importorskip("torch")
    tokenizer = tiny_hf["tokenizer"]
    tokenizer.padding_side = "right"
    batch = tokenizer(["hello world", "hello"], return_tensors="pt", padding=True).input_ids
    spec = CaptureSpec(mode=CaptureMode.TEACHER_FORCING_SPARSE, layers=[0], positions=[[0, 1], [0]])
    with HiddenCapture(tiny_hf["model"], spec) as capture:
        with torch.no_grad():
            tiny_hf["model"](batch, use_cache=False)
    result = capture.collect(token_ids=batch)
    assert result.hiddens[0].shape[:2] == (2, 2)
    assert result.valid_mask.tolist() == [[True, True], [True, False]]


def test_per_sample_sparse_positions_must_match_batch_size():
    torch = pytest.importorskip("torch")
    spec = CaptureSpec(mode=CaptureMode.TEACHER_FORCING_SPARSE, layers=[0], positions=[[0]])
    capture = HiddenCapture(_one_layer_model(torch), spec)
    capture.hiddens[0] = torch.randn(2, 2, 3)
    with pytest.raises(ValueError):
        capture.collect()


def test_logits_follow_capture_dtype_and_device():
    torch = pytest.importorskip("torch")
    spec = CaptureSpec(mode=CaptureMode.PREFILL, layers=[0], positions="last", with_logits=True, dtype=torch.float64, device="cpu")
    capture = HiddenCapture(_one_layer_model(torch), spec)
    capture.hiddens[0] = torch.randn(1, 2, 3)
    result = capture.collect(logits=torch.randn(1, 2, 5))
    assert result.hiddens[0].dtype == torch.float64
    assert result.logits.dtype == torch.float64
    assert result.logits.device.type == "cpu"


def test_sparse_valid_mask_follows_capture_device_without_dtype_cast():
    torch = pytest.importorskip("torch")
    spec = CaptureSpec(mode=CaptureMode.TEACHER_FORCING_SPARSE, layers=[0], positions=[[0, 1], [0]], dtype=torch.float64, device="meta")
    capture = HiddenCapture(_one_layer_model(torch), spec)
    capture.hiddens[0] = torch.randn(2, 2, 3)
    result = capture.collect()
    assert result.hiddens[0].device.type == "meta"
    assert result.valid_mask.device.type == "meta"
    assert result.valid_mask.dtype == torch.bool


def test_enter_clears_previous_hidden_state():
    torch = pytest.importorskip("torch")
    capture = HiddenCapture(_one_layer_model(torch), CaptureSpec(mode=CaptureMode.PREFILL, layers=[0]))
    capture.hiddens[0] = torch.randn(1, 1, 3)
    with capture:
        pass
    assert capture.collect().hiddens == {}


def test_unsupported_architecture_raises():
    class NoLayers:
        pass

    with pytest.raises(UnsupportedArchitectureError):
        HiddenCapture(NoLayers(), CaptureSpec(mode=CaptureMode.PREFILL))


def _one_layer_model(torch):
    class OneLayerModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Identity()])

    return OneLayerModel()
