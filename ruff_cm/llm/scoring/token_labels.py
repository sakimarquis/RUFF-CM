from __future__ import annotations


def strip_bpe_prefix(token: str) -> str:
    text = str(token).strip()
    while text.startswith(("\u0120", "\u2581")):
        text = text[1:].lstrip()
    return text.strip()
