from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import groupby
from types import SimpleNamespace
from typing import Any, Literal

import torch

from .families import is_gemma3_family, is_gemma3_vlm, is_gemma4_family, is_mistral3_family, uses_processor_renderer


@dataclass(frozen=True)
class LoaderConfig:
    model_id: str
    dtype: Any | str | None = None
    device_map: str | dict | None = "balanced"
    padding_side: Literal["left", "right"] = "left"
    trust_remote_code: bool = False
    peft_path: str | None = None
    use_unsloth: bool = False
    attn_implementation: str | None = None


class ProcessorTokenizerAdapter:
    """Expose a multimodal processor through the tokenizer surface used downstream."""

    def __init__(self, processor: Any):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.name_or_path = getattr(processor, "name_or_path", None) or getattr(self.tokenizer, "name_or_path", "")

    def __len__(self):
        return len(self.tokenizer)

    def __call__(self, text, *args, **kwargs):
        return self.processor(text=text, *args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        return self.processor.apply_chat_template(*args, **kwargs)

    def parse_response(self, *args, **kwargs):
        return self.processor.parse_response(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self.tokenizer.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        decoder = getattr(self.processor, "decode", None) or self.tokenizer.decode
        return decoder(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        decoder = getattr(self.processor, "batch_decode", None) or self.tokenizer.batch_decode
        return decoder(*args, **kwargs)

    def convert_tokens_to_ids(self, *args, **kwargs):
        return self.tokenizer.convert_tokens_to_ids(*args, **kwargs)

    @property
    def chat_template(self):
        return getattr(self.processor, "chat_template", None) or getattr(self.tokenizer, "chat_template", None)

    @chat_template.setter
    def chat_template(self, value):
        self.processor.chat_template = value
        self.tokenizer.chat_template = value

    @property
    def pad_token(self):
        return self.tokenizer.pad_token

    @pad_token.setter
    def pad_token(self, value):
        self.tokenizer.pad_token = value

    @property
    def eos_token(self):
        return self.tokenizer.eos_token

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @pad_token_id.setter
    def pad_token_id(self, value):
        self.tokenizer.pad_token_id = value

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def unk_token_id(self):
        return self.tokenizer.unk_token_id

    @property
    def padding_side(self):
        return self.tokenizer.padding_side

    @padding_side.setter
    def padding_side(self, value):
        self.tokenizer.padding_side = value


def load_hf_model_and_renderer(cfg: LoaderConfig) -> tuple[Any, Any]:
    """Load an HF model plus the tokenizer-like renderer used to format inputs."""
    model, renderer = _load_unsloth(cfg) if cfg.use_unsloth else _load_transformers_model(cfg)
    tokenizer_like = _tokenizer_like(renderer, cfg.model_id)

    if cfg.peft_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, cfg.peft_path).merge_and_unload()

    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    tokenizer_like.padding_side = cfg.padding_side
    if tokenizer_like.pad_token is None:
        tokenizer_like.pad_token = tokenizer_like.eos_token
    return model, tokenizer_like


def print_device_map(model: Any) -> None:
    """Print per-device layer placement and CUDA memory after model loading."""
    if not hasattr(model, "hf_device_map"):
        return

    device_map = model.hf_device_map
    counts = Counter(str(device) for device in device_map.values())
    print(f"-- Device map ({len(device_map)} modules across {len(counts)} device(s)) --")
    for device, count in sorted(counts.items()):
        print(f"  device {device}: {count} modules")

    layer_devices = {}
    for name, device in device_map.items():
        parts = name.split(".")
        for index, part in enumerate(parts):
            if part == "layers" and index + 1 < len(parts) and parts[index + 1].isdigit():
                layer_devices[int(parts[index + 1])] = str(device)
                break
    if layer_devices:
        print("  Layer placement:")
        for device, group in groupby(sorted(layer_devices.items()), key=lambda item: item[1]):
            indexes = [item[0] for item in group]
            print(f"  layers {indexes[0]}-{indexes[-1]} -> device {device}")

    for index in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(index) / 1024**3
        total = torch.cuda.get_device_properties(index).total_memory / 1024**3
        print(f"  GPU {index}: {allocated:.1f} / {total:.1f} GiB allocated")


def _load_transformers_model(cfg: LoaderConfig) -> tuple[Any, Any]:
    tfm = _load_transformers_classes()
    renderer_cls = tfm.AutoProcessor if uses_processor_renderer(cfg.model_id) else tfm.AutoTokenizer
    renderer = renderer_cls.from_pretrained(cfg.model_id, trust_remote_code=cfg.trust_remote_code)
    model_cls = _model_class(tfm, cfg.model_id)
    model_kwargs = {
        "device_map": cfg.device_map,
        "torch_dtype": _resolve_torch_dtype(cfg.model_id, cfg.dtype, tfm),
        "trust_remote_code": cfg.trust_remote_code,
    }
    if cfg.attn_implementation is not None:
        model_kwargs["attn_implementation"] = cfg.attn_implementation
    return model_cls.from_pretrained(cfg.model_id, **model_kwargs), renderer


def _load_unsloth(cfg: LoaderConfig) -> tuple[Any, Any]:
    import unsloth

    model_api = getattr(unsloth, _select_unsloth_loader_name(cfg.model_id))
    load_kwargs = {"model_name": cfg.model_id, "dtype": None, "load_in_fp8": True, "load_in_4bit": False}
    if cfg.device_map is not None:
        load_kwargs["device_map"] = cfg.device_map
    return model_api.from_pretrained(**load_kwargs)


def _tokenizer_like(renderer: Any, model_id: str) -> Any:
    if uses_processor_renderer(model_id):
        return ProcessorTokenizerAdapter(renderer)
    return getattr(renderer, "tokenizer", renderer)


def _load_transformers_classes() -> SimpleNamespace:
    import transformers

    return SimpleNamespace(
        AutoConfig=transformers.AutoConfig,
        AutoModelForCausalLM=transformers.AutoModelForCausalLM,
        AutoModelForImageTextToText=getattr(transformers, "AutoModelForImageTextToText", None),
        AutoModelForMultimodalLM=getattr(transformers, "AutoModelForMultimodalLM", None),
        AutoProcessor=transformers.AutoProcessor,
        AutoTokenizer=transformers.AutoTokenizer,
        Mistral3ForConditionalGeneration=getattr(transformers, "Mistral3ForConditionalGeneration", None),
    )


def _model_class(tfm: SimpleNamespace, model_id: str):
    if is_gemma4_family(model_id):
        return tfm.AutoModelForMultimodalLM
    if is_mistral3_family(model_id):
        return tfm.Mistral3ForConditionalGeneration
    if is_gemma3_vlm(model_id):
        return tfm.AutoModelForImageTextToText
    return tfm.AutoModelForCausalLM


def _resolve_torch_dtype(model_id: str, dtype: Any | str | None, tfm: SimpleNamespace):
    if dtype not in {None, "auto"}:
        return getattr(torch, dtype) if isinstance(dtype, str) else dtype

    config = tfm.AutoConfig.from_pretrained(model_id)
    resolved = getattr(config, "torch_dtype", None) or getattr(config, "dtype", None)
    if resolved is None or resolved == torch.float32:
        return torch.bfloat16
    return resolved


def _select_unsloth_loader_name(model_id: str) -> str:
    normalized = model_id.lower()
    fastmodel_markers = ("gemma-4-26b-a4b", "qwen3.6-35b-a3b")
    if (
        is_gemma3_family(model_id)
        or is_mistral3_family(model_id)
        or any(marker in normalized for marker in fastmodel_markers)
    ):
        return "FastModel"
    return "FastLanguageModel"
