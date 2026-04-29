"""Chat-template-aware token span helpers."""


def _as_list(tokens):
    return tokens.tolist() if hasattr(tokens, "tolist") else list(tokens)


def _input_ids(tokenized):
    if hasattr(tokenized, "keys") and "input_ids" in tokenized:
        tokenized = tokenized["input_ids"]
    return _as_list(tokenized)


def _chat_ids(tokenizer, messages: list[dict[str, str]], *, add_generation_prompt: bool = False) -> list[int]:
    tokenized = tokenizer.apply_chat_template(
        messages, add_generation_prompt=add_generation_prompt, tokenize=True, return_dict=False
    )
    return _input_ids(tokenized)


def _diff_span(full, without):
    start = 0
    while start < len(full) and start < len(without) and full[start] == without[start]:
        start += 1

    suffix = 0
    while suffix < len(full) - start and suffix < len(without) - start and full[-suffix - 1] == without[-suffix - 1]:
        suffix += 1

    return start, len(full) - suffix


def _message_spans(tokenizer, messages: list[dict[str, str]]) -> list[tuple[int, int]]:
    spans = []
    span_start = 0
    for idx in range(len(messages)):
        prefix_ids = _chat_ids(tokenizer, messages[: idx + 1])
        span_end = len(prefix_ids)
        spans.append((span_start, span_end))
        span_start = span_end
    return spans


def assistant_header(tokenizer, *, tokenize: bool = False):
    """Return the assistant generation header introduced by the tokenizer template."""
    messages = [{"role": "user", "content": ""}]
    if tokenize:
        prompted = _chat_ids(tokenizer, messages, add_generation_prompt=True)
        plain = _chat_ids(tokenizer, messages, add_generation_prompt=False)
    else:
        prompted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        plain = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    start, end = _diff_span(prompted, plain)
    return prompted[start:end]


def locate_message(
    tokenizer, messages: list[dict[str, str]], *, target_idx: int, add_generation_prompt: bool = True
) -> tuple[list[int], int, int]:
    """Locate the token span introduced by one message in a rendered chat."""
    full_ids = _chat_ids(tokenizer, messages, add_generation_prompt=add_generation_prompt)
    start, end = _message_spans(tokenizer, messages)[target_idx]
    return full_ids, start, end


def find_subsequences(tokens, named: dict[str, list[int]]) -> dict[str, list[tuple[int, int]]]:
    token_list = _as_list(tokens)
    hits = {}
    for name, pattern in named.items():
        pattern = _as_list(pattern)
        if not pattern:
            hits[name] = []
            continue
        width = len(pattern)
        hits[name] = [
            (idx, idx + width) for idx in range(len(token_list) - width + 1) if token_list[idx:idx + width] == pattern
        ]
    return hits


def tokenize_with_loss_mask(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    max_length: int = 4096,
    assistant_role: str = "assistant",
    ignore_index: int = -100,
) -> dict[str, list[int]]:
    input_ids = _chat_ids(tokenizer, messages)
    labels = [ignore_index] * len(input_ids)

    # Prefix growth gives each message its own span even when adjacent content is identical.
    for message, (span_start, span_end) in zip(messages, _message_spans(tokenizer, messages)):
        if message["role"] == assistant_role:
            labels[span_start:span_end] = input_ids[span_start:span_end]

    input_ids = input_ids[:max_length]
    labels = labels[:max_length]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}
