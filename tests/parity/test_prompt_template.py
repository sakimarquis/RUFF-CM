from ruff_cm.llm.backends import Message
from ruff_cm.llm.prompt.template import (
    assistant_header,
    compute_encoding_offset,
    detect_assistant_suffix,
    detect_bos_prefix,
    locate_message,
)
from ruff_cm.llm.spans import assistant_header as legacy_assistant_header


class TemplateTokenizer:
    bos = "[BOS]"

    def apply_chat_template(
        self, messages, *, add_generation_prompt=False, tokenize=False, return_dict=False, enable_thinking=False
    ):
        rendered = self.bos
        for message in messages:
            rendered += f"<|{message['role']}|>{message['content']}<end>"
        if add_generation_prompt:
            rendered += "<|assistant|>"
        return list(rendered.encode("utf-8")) if tokenize else rendered

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(text.encode("utf-8"))}

    def decode(self, ids):
        return bytes(ids).decode("utf-8")


def test_prompt_template_exports_existing_span_helpers():
    tokenizer = TemplateTokenizer()
    messages = [Message(role="user", content="question"), Message(role="assistant", content="answer")]

    full_ids, start, end = locate_message(tokenizer, messages, target_idx=1)

    assert assistant_header(tokenizer, tokenize=True) == legacy_assistant_header(tokenizer, tokenize=True)
    assert "answer" in tokenizer.decode(full_ids[start:end])


def test_detect_bos_prefix_and_assistant_suffix_are_tokenized_template_diffs():
    tokenizer = TemplateTokenizer()

    assert tokenizer.decode(detect_bos_prefix(tokenizer)) == "[BOS]"
    assert tokenizer.decode(detect_assistant_suffix(tokenizer)) == "<end>"


def test_compute_encoding_offset_counts_suffix_tokens_after_next_message_content():
    tokenizer = TemplateTokenizer()

    offset = compute_encoding_offset(tokenizer, [Message(role="system", content="policy")])

    assert offset == len("<end><|assistant|>".encode("utf-8"))
