import numpy as np

from ruff_cm.llm.spans import assistant_header, find_subsequences, locate_message, tokenize_with_loss_mask


class FakeTokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt=False, tokenize=False):
        rendered = "".join(f"<|{message['role']}|>{message['content']}\n" for message in messages)
        if add_generation_prompt:
            rendered += "<|assistant|>"
        return list(rendered.encode("utf-8")) if tokenize else rendered

    def decode(self, ids):
        return bytes(ids).decode("utf-8")


def test_assistant_header_returns_text_and_token_ids():
    tokenizer = FakeTokenizer()

    text = assistant_header(tokenizer)
    ids = assistant_header(tokenizer, tokenize=True)

    assert text
    assert "assistant" in text
    assert ids
    assert all(isinstance(token_id, int) for token_id in ids)


def test_locate_message_span_decodes_target_content():
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "target answer"},
        {"role": "user", "content": "followup"},
    ]

    full_ids, start, end = locate_message(tokenizer, messages, target_idx=1)

    assert "target answer" in tokenizer.decode(full_ids[start:end])
    assert start < end


def test_find_subsequences_returns_all_occurrences_and_missing_pattern():
    hits = find_subsequences([1, 2, 1, 2, 1], {"pair": [1, 2], "missing": [3]})

    assert hits == {"pair": [(0, 2), (2, 4)], "missing": []}


def test_find_subsequences_accepts_ndarray():
    hits = find_subsequences(np.array([4, 5, 4]), {"value": [4]})

    assert hits == {"value": [(0, 1), (2, 3)]}


def test_tokenize_with_loss_mask_ignores_user_tokens_and_keeps_assistant_tokens():
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]

    encoded = tokenize_with_loss_mask(tokenizer, messages)
    user_start = tokenizer.decode(encoded["input_ids"]).index("question")
    assistant_start = tokenizer.decode(encoded["input_ids"]).index("answer")

    assert encoded["labels"][user_start] == -100
    assert encoded["labels"][assistant_start] == encoded["input_ids"][assistant_start]
    assert any(label != -100 for label in encoded["labels"])
    assert encoded["attention_mask"] == [1] * len(encoded["input_ids"])


def test_tokenize_with_loss_mask_handles_back_to_back_assistant_turns():
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "first"},
        {"role": "assistant", "content": "second"},
    ]

    encoded = tokenize_with_loss_mask(tokenizer, messages)
    rendered = tokenizer.decode(encoded["input_ids"])
    first_start = rendered.index("first")
    second_start = rendered.index("second")

    assert encoded["labels"][first_start] == encoded["input_ids"][first_start]
    assert encoded["labels"][second_start] == encoded["input_ids"][second_start]


def test_tokenize_with_loss_mask_labels_identical_back_to_back_assistant_turns():
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "same"},
        {"role": "assistant", "content": "same"},
    ]

    encoded = tokenize_with_loss_mask(tokenizer, messages)
    rendered = tokenizer.decode(encoded["input_ids"])
    first_start = rendered.index("same")
    second_start = rendered.index("same", first_start + len("same"))

    for start in (first_start, second_start):
        end = start + len("same")
        assert encoded["labels"][start:end] == encoded["input_ids"][start:end]
