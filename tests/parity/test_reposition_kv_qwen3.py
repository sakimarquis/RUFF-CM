from __future__ import annotations

import pytest

from ruff_cm.llm.inference.kvcache import reposition_kv


@pytest.mark.hf
def test_reposition_kv_qwen3_matches_fresh_forward_after_rebase():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    model = transformers.AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.float32,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model.eval()

    input_ids = tokenizer("A B C D", return_tensors="pt").input_ids[:, :4]
    next_id = tokenizer(" E", return_tensors="pt", add_special_tokens=False).input_ids[:, :1]
    old_start_pos = 10
    old_position_ids = torch.arange(old_start_pos, old_start_pos + input_ids.shape[1]).unsqueeze(0)

    with torch.no_grad():
        old_outputs = model(input_ids=input_ids, position_ids=old_position_ids, use_cache=True)
        rebased_kv = reposition_kv(model, old_outputs.past_key_values, old_start_pos=old_start_pos, m=input_ids.shape[1])
        rebased_outputs = model(
            input_ids=next_id,
            attention_mask=torch.ones(1, input_ids.shape[1] + 1, dtype=torch.long),
            past_key_values=rebased_kv,
            cache_position=torch.tensor([input_ids.shape[1]]),
            use_cache=True,
        )
        fresh_outputs = model(input_ids=torch.cat([input_ids, next_id], dim=1), use_cache=False)

    cosine = torch.nn.functional.cosine_similarity(rebased_outputs.logits[:, -1], fresh_outputs.logits[:, -1])
    assert cosine.item() > 1 - 1e-4
