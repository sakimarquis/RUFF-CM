from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fake_openai_client():
    return MagicMock()


def make_openai_response(text: str = "ok", finish_reason: str = "stop", top_logprobs: list[tuple[str, float]] | None = None):
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason, logprobs=None)
    if top_logprobs is not None:
        entries = [SimpleNamespace(token=token, logprob=logprob) for token, logprob in top_logprobs]
        choice.logprobs = SimpleNamespace(content=[SimpleNamespace(top_logprobs=entries)])
    return SimpleNamespace(choices=[choice], model_dump=lambda: {"choices": [{"message": {"content": text}}]})


@pytest.fixture
def openai_response_factory():
    return make_openai_response


@pytest.fixture(scope="session")
def tiny_hf():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2", torch_dtype=torch.float32)
    model.eval()
    return {"model": model, "tokenizer": tokenizer, "model_id": "sshleifer/tiny-gpt2"}
