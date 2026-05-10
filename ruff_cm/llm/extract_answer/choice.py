from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import torch

from ..backends.base import ChoiceScores


@dataclass(frozen=True)
class VariantRule:
    """Expansion rule for single-token answer variants."""

    name: str

    @classmethod
    def case_insensitive(cls) -> "VariantRule":
        return cls("case_insensitive")

    @classmethod
    def with_leading_space(cls) -> "VariantRule":
        return cls("with_leading_space")

    @classmethod
    def with_leading_newline(cls) -> "VariantRule":
        return cls("with_leading_newline")

    def expand(self, text: str) -> tuple[str, ...]:
        match self.name:
            case "case_insensitive":
                return (text, text.upper(), text.lower())
            case "with_leading_space":
                return (text, f" {text}")
            case "with_leading_newline":
                return (text, f"\n{text}")
            case _:
                raise ValueError(f"unknown variant rule {self.name!r}")


def build_letter_token_ids(tokenizer, letters: list[str], *, variants: list[VariantRule]) -> dict[str, list[int]]:
    token_map: dict[str, list[int]] = {}
    for letter in letters:
        token_map[letter] = _collect_single_token_ids(tokenizer, _render_with_rules(letter, variants))
    return token_map


def compute_letter_log_probs(logits, token_id_map: Mapping[str, Sequence[int]]) -> dict[str, float]:
    raw_scores = []
    letters = list(token_id_map)
    for letter in letters:
        token_ids = list(token_id_map[letter])
        if token_ids:
            raw_scores.append(torch.logsumexp(logits[token_ids], dim=0))
        else:
            raw_scores.append(torch.tensor(-float("inf"), device=logits.device))
    score_tensor = torch.stack(raw_scores)
    log_probs = score_tensor - torch.logsumexp(score_tensor, dim=0)
    return {letter: float(log_probs[index].item()) for index, letter in enumerate(letters)}


class ChoiceSet:
    def __init__(
        self,
        tokenizer,
        candidates: Sequence[str],
        variants: Iterable[str | VariantRule] | None = None,
        decorators: Iterable[str] = ("{c}",),
        aggregation: str | None = None,
        *,
        mode: str = "exact",
    ):
        if mode not in {"exact", "partial"}:
            raise ValueError(f"unknown choice mode {mode!r}")

        self._tokenizer = tokenizer
        self._candidates = list(candidates)
        self._variants = tuple(("raw",) if variants is None else variants)
        self._decorators = tuple(decorators)
        has_rule_variants = any(isinstance(variant, VariantRule) for variant in self._variants)
        self._aggregation = aggregation or ("logsumexp" if has_rule_variants else "max")
        if self._aggregation not in {"max", "logsumexp"}:
            raise ValueError(f"unknown choice aggregation {self._aggregation!r}")

        self._token_map: dict[str, list[int]] = {}
        self._rendered: dict[str, list[str]] = {}
        for candidate in self._candidates:
            raw_token_ids = self._encode(candidate)
            if len(raw_token_ids) != 1:
                raise ValueError("multi-token candidate %r - multi-token scoring is out of scope for v0.3" % candidate)
            rendered_strings = self._render_candidate(candidate)
            self._token_map[candidate] = sorted(_collect_single_token_ids(self._tokenizer, rendered_strings))
            self._rendered[candidate] = rendered_strings

    @property
    def candidates(self) -> list[str]:
        return list(self._candidates)

    @property
    def token_map(self) -> dict[str, list[int]]:
        return {candidate: list(token_ids) for candidate, token_ids in self._token_map.items()}

    def score(self, logits, normalize: bool = True) -> ChoiceScores:
        return self.from_logits(logits, normalize=normalize)

    def from_logits(self, logits, normalize: bool = True) -> ChoiceScores:
        candidate_scores = [self._aggregate_logits(logits[..., token_ids]) for token_ids in self._token_map.values()]
        score_tensor = torch.stack(candidate_scores, dim=-1)
        if normalize:
            score_tensor = torch.log_softmax(score_tensor, dim=-1)

        scores = {candidate: self._to_python_score(score_tensor[..., index]) for index, candidate in enumerate(self._candidates)}
        return ChoiceScores(method="exact", scores=scores, complete=True, missing=[], fallback_count=0)

    def score_from_top_logprobs(self, top_logprobs, normalize: bool = True) -> ChoiceScores:
        return self.from_top_logprobs(_top_logprobs_to_mapping(top_logprobs), normalize=normalize)

    def from_top_logprobs(self, top_logprobs: Mapping[str, float], normalize: bool = True) -> ChoiceScores:
        present: dict[str, float] = {}
        missing: list[str] = []
        for candidate in self._candidates:
            scores = [top_logprobs[rendered] for rendered in self._rendered[candidate] if rendered in top_logprobs]
            if scores:
                present[candidate] = self._aggregate_python_scores(scores)
            else:
                missing.append(candidate)

        if normalize and present:
            normalizer = math.log(sum(math.exp(score) for score in present.values()))
            present = {candidate: score - normalizer for candidate, score in present.items()}

        return ChoiceScores(method="partial", scores=present, complete=not missing, missing=missing, fallback_count=0)

    def _aggregate_logits(self, values):
        if self._aggregation == "max":
            return values.max(dim=-1).values
        if self._aggregation == "logsumexp":
            return torch.logsumexp(values, dim=-1)
        raise ValueError(f"unknown choice aggregation {self._aggregation!r}")

    def _aggregate_python_scores(self, scores: list[float]) -> float:
        if self._aggregation == "max":
            return max(scores)
        if self._aggregation == "logsumexp":
            return math.log(sum(math.exp(score) for score in scores))
        raise ValueError(f"unknown choice aggregation {self._aggregation!r}")

    def _render_candidate(self, candidate: str) -> list[str]:
        if all(isinstance(variant, VariantRule) for variant in self._variants):
            variant_texts = _render_with_rules(candidate, list(self._variants))
        else:
            variant_texts = [self._apply_legacy_variant(candidate, variant) for variant in self._variants]

        rendered: list[str] = []
        for variant_text in variant_texts:
            for decorator in self._decorators:
                decorated = decorator.format(c=variant_text)
                if decorated not in rendered:
                    rendered.append(decorated)
        return rendered

    def _apply_legacy_variant(self, candidate: str, variant: str | VariantRule) -> str:
        if isinstance(variant, VariantRule):
            raise ValueError("cannot mix VariantRule instances with legacy string variants")
        match variant:
            case "raw":
                return candidate
            case "with_space":
                return f" {candidate}"
            case "upper":
                return candidate.upper()
            case "lower":
                return candidate.lower()
            case _:
                raise ValueError(f"unknown choice variant {variant!r}")

    def _encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def _to_python_score(self, score):
        if score.ndim == 0:
            return float(score.item())
        return score.tolist()


def _render_with_rules(text: str, rules: Sequence[VariantRule]) -> list[str]:
    rendered = [text]
    for rule in rules:
        expanded: list[str] = []
        for item in rendered:
            expanded.extend(rule.expand(item))
        rendered = _dedupe_preserve_order([*rendered, *expanded])
    return rendered


def _collect_single_token_ids(tokenizer, variants: Iterable[str]) -> list[int]:
    token_ids: list[int] = []
    for variant in variants:
        try:
            encoded = list(tokenizer.encode(variant, add_special_tokens=False))
        except KeyError:
            continue
        if len(encoded) == 1 and encoded[0] not in token_ids:
            token_ids.append(encoded[0])
    return token_ids


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    for item in items:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _top_logprobs_to_mapping(top_logprobs) -> Mapping[str, float]:
    if isinstance(top_logprobs, Mapping):
        return top_logprobs
    merged: dict[str, float] = {}
    for entry in top_logprobs:
        if isinstance(entry, Mapping) and "token" in entry and "logprob" in entry:
            merged[str(entry["token"])] = float(entry["logprob"])
        else:
            merged.update({str(token): float(logprob) for token, logprob in entry.items()})
    return merged


__all__ = ["ChoiceSet", "VariantRule", "build_letter_token_ids", "compute_letter_log_probs"]
