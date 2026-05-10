from __future__ import annotations

from types import SimpleNamespace

import pytest

from ruff_cm.llm.backends.loaders import LoaderConfig, ProcessorTokenizerAdapter, load_hf_model_and_renderer


class FakeTokenizer:
    name_or_path = "fake"
    chat_template = None
    pad_token = None
    eos_token = "<eos>"
    pad_token_id = None
    eos_token_id = 1
    unk_token_id = 0
    padding_side = "right"

    def __len__(self):
        return 10

    def encode(self, text):
        return [len(text)]

    def decode(self, token_ids, **kwargs):
        return "decoded"

    def batch_decode(self, batch_ids, **kwargs):
        return ["decoded" for _ in batch_ids]

    def convert_tokens_to_ids(self, token):
        return 1 if token == self.eos_token else 0


class FakeProcessor:
    name_or_path = "fake-processor"
    chat_template = "template"

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return {"input_ids": [1]}

    def apply_chat_template(self, *args, **kwargs):
        return "rendered"

    def parse_response(self, *args, **kwargs):
        return {"answer": "A"}

    def decode(self, *args, **kwargs):
        return "processor-decoded"

    def batch_decode(self, batch_ids, **kwargs):
        return ["processor-decoded" for _ in batch_ids]


def test_processor_tokenizer_adapter_exposes_tokenizer_surface():
    processor = FakeProcessor()
    adapter = ProcessorTokenizerAdapter(processor)

    assert len(adapter) == 10
    assert adapter("prompt", return_tensors="pt") == {"input_ids": [1]}
    assert processor.calls == [("prompt", {"return_tensors": "pt"})]
    assert adapter.apply_chat_template([]) == "rendered"
    assert adapter.batch_decode([[1]]) == ["processor-decoded"]
    adapter.padding_side = "left"
    adapter.pad_token = "<eos>"

    assert processor.tokenizer.padding_side == "left"
    assert processor.tokenizer.pad_token == "<eos>"


def test_load_hf_model_and_renderer_uses_auto_dtype_and_padding(monkeypatch):
    torch = pytest.importorskip("torch")
    calls = []

    class FakeModel:
        config = SimpleNamespace(use_cache=False)

        def eval(self):
            self.eval_called = True

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_id):
            calls.append(("config", model_id))
            return SimpleNamespace(torch_dtype=torch.float32)

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append(("tokenizer", model_id, kwargs))
            return FakeTokenizer()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append(("model", model_id, kwargs))
            return FakeModel()

    fake_tfm = SimpleNamespace(
        AutoConfig=FakeAutoConfig,
        AutoModelForCausalLM=FakeAutoModel,
        AutoModelForImageTextToText=None,
        AutoModelForMultimodalLM=None,
        AutoProcessor=None,
        AutoTokenizer=FakeAutoTokenizer,
        Mistral3ForConditionalGeneration=None,
    )
    monkeypatch.setattr("ruff_cm.llm.backends.loaders._load_transformers_classes", lambda: fake_tfm)

    model, tokenizer = load_hf_model_and_renderer(
        LoaderConfig(model_id="Qwen/Qwen3-4B", dtype=None, device_map="balanced", padding_side="left")
    )

    assert model.eval_called is True
    assert model.config.use_cache is True
    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token == "<eos>"
    assert ("config", "Qwen/Qwen3-4B") in calls
    assert (
        "model",
        "Qwen/Qwen3-4B",
        {"device_map": "balanced", "torch_dtype": torch.bfloat16, "trust_remote_code": False},
    ) in calls


def test_load_hf_model_and_renderer_uses_processor_for_multimodal(monkeypatch):
    class FakeModel:
        config = SimpleNamespace(use_cache=False)

        def eval(self):
            pass

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_id):
            return SimpleNamespace(torch_dtype="auto")

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            return FakeProcessor()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            return FakeModel()

    fake_tfm = SimpleNamespace(
        AutoConfig=FakeAutoConfig,
        AutoModelForCausalLM=None,
        AutoModelForImageTextToText=FakeAutoModel,
        AutoModelForMultimodalLM=None,
        AutoProcessor=FakeAutoProcessor,
        AutoTokenizer=None,
        Mistral3ForConditionalGeneration=None,
    )
    monkeypatch.setattr("ruff_cm.llm.backends.loaders._load_transformers_classes", lambda: fake_tfm)

    _, renderer = load_hf_model_and_renderer(LoaderConfig(model_id="google/gemma-3-27b-it"))

    assert isinstance(renderer, ProcessorTokenizerAdapter)
    assert renderer.apply_chat_template([]) == "rendered"


@pytest.mark.hf
def test_load_hf_model_and_renderer_qwen_smoke():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the gated loader smoke test")

    model, tokenizer = load_hf_model_and_renderer(
        LoaderConfig(model_id="Qwen/Qwen3-0.6B", dtype="bfloat16", device_map="auto")
    )
    encoded = tokenizer("hello", return_tensors="pt")
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded, use_cache=False)

    assert outputs.logits.shape[0] == 1
    assert next(model.parameters()).dtype == torch.bfloat16
