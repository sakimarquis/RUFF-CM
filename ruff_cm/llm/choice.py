from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

from .backends.base import ChoiceScores


class ChoiceSet:
    def __init__(
        self,
        tokenizer,
        candidates: Sequence[str],
        variants: Iterable[str] = ("raw",),
        decorators: Iterable[str] = ("{c}",),
    ):
        self._tokenizer = tokenizer
        self._candidates = list(candidates)
        self._variants = tuple(variants)
        self._decorators = tuple(decorators)
        self._token_map: dict[str, list[int]] = {}
        self._rendered: dict[str, list[str]] = {}

        for candidate in self._candidates:
            raw_token_ids = self._encode(candidate)
            if len(raw_token_ids) != 1:
                raise ValueError("multi-token candidate %r - multi-token scoring is out of scope for v0.3" % candidate)

            token_ids: list[int] = []
            rendered_strings: list[str] = []
            for rendered in self._render_candidate(candidate):
                try:
                    encoded = self._encode(rendered)
                except KeyError:
                    continue
                if len(encoded) == 1:
                    token_id = encoded[0]
                    if token_id not in token_ids:
                        token_ids.append(token_id)
                    if rendered not in rendered_strings:
                        rendered_strings.append(rendered)

            self._token_map[candidate] = sorted(token_ids)
            self._rendered[candidate] = rendered_strings

    @property
    def candidates(self) -> list[str]:
        return list(self._candidates)

    @property
    def token_map(self) -> dict[str, list[int]]:
        return {candidate: list(token_ids) for candidate, token_ids in self._token_map.items()}

    def from_logits(self, logits, normalize: bool = True) -> ChoiceScores:
        import torch

        candidate_scores = [logits[..., token_ids].max(dim=-1).values for token_ids in self._token_map.values()]
        score_tensor = torch.stack(candidate_scores, dim=-1)
        if normalize:
            score_tensor = torch.log_softmax(score_tensor, dim=-1)

        scores = {candidate: self._to_python_score(score_tensor[..., index]) for index, candidate in enumerate(self._candidates)}
        return ChoiceScores(method="exact", scores=scores, complete=True, missing=[], fallback_count=0)

    def from_top_logprobs(self, top_logprobs: Mapping[str, float], normalize: bool = True) -> ChoiceScores:
        present: dict[str, float] = {}
        missing: list[str] = []
        for candidate in self._candidates:
            scores = [top_logprobs[rendered] for rendered in self._rendered[candidate] if rendered in top_logprobs]
            if scores:
                present[candidate] = max(scores)
            else:
                missing.append(candidate)

        if normalize and present:
            normalizer = math.log(sum(math.exp(score) for score in present.values()))
            present = {candidate: score - normalizer for candidate, score in present.items()}

        return ChoiceScores(method="partial", scores=present, complete=not missing, missing=missing, fallback_count=0)

    def _render_candidate(self, candidate: str) -> list[str]:
        rendered: list[str] = []
        for variant in self._variants:
            variant_text = self._apply_variant(candidate, variant)
            for decorator in self._decorators:
                decorated = decorator.format(c=variant_text)
                if decorated not in rendered:
                    rendered.append(decorated)
        return rendered

    def _apply_variant(self, candidate: str, variant: str) -> str:
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
