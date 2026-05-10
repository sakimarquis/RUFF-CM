import numpy as np

from ruff_cm.llm.backends import Message
from ruff_cm.llm.prompt.tokenize import find_subsequences, tokenize_with_loss_mask
from ruff_cm.llm.prompt.tokenize import find_subsequences as span_find_subsequences
from ruff_cm.llm.prompt.tokenize import tokenize_with_loss_mask as span_tokenize_with_loss_mask


class MaskTokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt=False, tokenize=False, return_dict=False):
        rendered = "".join(f"<|{message['role']}|>{message['content']}\n" for message in messages)
        if add_generation_prompt:
            rendered += "<|assistant|>"
        return list(rendered.encode("utf-8")) if tokenize else rendered

    def decode(self, ids):
        return bytes(ids).decode("utf-8")


def test_find_subsequences_keeps_named_spans_and_supports_single_needles():
    assert find_subsequences(np.array([1, 2, 1, 2, 1]), {"pair": [1, 2]}) == {"pair": [(0, 2), (2, 4)]}
    assert span_find_subsequences([1, 2, 1, 2], {"pair": [1, 2]}) == {"pair": [(0, 2), (2, 4)]}
    assert find_subsequences([1, 2, 1, 2], [1, 2]) == [0, 2]


def test_tokenize_with_loss_mask_matches_span_import_and_accepts_message_objects():
    tokenizer = MaskTokenizer()
    messages = [Message(role="user", content="question"), Message(role="assistant", content="answer")]

    encoded = tokenize_with_loss_mask(tokenizer, messages)
    span_encoded = span_tokenize_with_loss_mask(tokenizer, messages)
    answer_start = tokenizer.decode(encoded["input_ids"]).index("answer")

    assert encoded == span_encoded
    assert encoded["labels"][answer_start] == encoded["input_ids"][answer_start]
