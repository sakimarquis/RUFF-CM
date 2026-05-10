from ruff_cm.llm.backends import Message
from ruff_cm.llm.prompt.template import compute_encoding_offset


class ThinkingOffsetTokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt=False, tokenize=False, enable_thinking=False):
        rendered = ""
        for message in messages:
            rendered += f"<|{message['role']}|>{message['content']}<end>"
        if add_generation_prompt:
            rendered += "<|assistant|>"
            if enable_thinking:
                rendered += "<think>"
        return list(rendered.encode("utf-8")) if tokenize else rendered

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(text.encode("utf-8"))}


def test_compute_encoding_offset_accounts_for_thinking_generation_marker():
    tokenizer = ThinkingOffsetTokenizer()
    prior_messages = [Message(role="system", content="policy")]

    plain_offset = compute_encoding_offset(tokenizer, prior_messages)
    thinking_offset = compute_encoding_offset(tokenizer, prior_messages, enable_thinking=True)

    assert plain_offset == len("<end><|assistant|>".encode("utf-8"))
    assert thinking_offset - plain_offset == len("<think>".encode("utf-8"))
