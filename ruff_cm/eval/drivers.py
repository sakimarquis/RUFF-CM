from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from tqdm import tqdm

from .finalize import finalize_accuracy
from .generate import generate_text_with_budget, mc_answer
from .trial import add_generation_metadata, add_trial_metadata


def _init_accuracy_state(categories: Sequence[str]) -> tuple[dict[str, dict[str, int]], dict[str, int], list[dict[str, Any]]]:
    cat_stats = {category: {"correct": 0, "total": 0} for category in categories}
    cat_counters = {category: 0 for category in categories}
    return cat_stats, cat_counters, []


def _record_accuracy_trial(
    cat_stats: dict[str, dict[str, int]],
    cat_counters: dict[str, int],
    trials: list[dict[str, Any]],
    benchmark_id: str,
    category: str,
    trial: dict[str, Any],
) -> None:
    hit = bool(trial["correct"])
    cat_stats[category]["correct"] += hit
    cat_stats[category]["total"] += 1
    trials.append(add_trial_metadata(trial, benchmark_id, category, cat_counters))


def _token_budget(max_new_tokens: int | Callable[[Any], int], sample: Any) -> int:
    return max_new_tokens(sample) if callable(max_new_tokens) else max_new_tokens


def run_accuracy_benchmark(
    generator,
    tokenizer,
    samples,
    categories,
    *,
    desc,
    build_messages,
    build_trial,
    benchmark_id,
    max_new_tokens=512,
):
    """Run a sampled benchmark with the shared free-generation accuracy loop."""
    cat_stats, cat_counters, trials = _init_accuracy_state(categories)
    for sample in tqdm(samples, desc=desc, leave=False):
        category = sample[0]
        token_budget = _token_budget(max_new_tokens, sample)
        response, n_tokens, truncated, n_input_tokens = generate_text_with_budget(
            generator,
            tokenizer,
            build_messages(sample),
            max_new_tokens=token_budget,
        )
        trial = build_trial(sample, response)
        trial = add_generation_metadata(trial, response, n_tokens, truncated, n_input_tokens, token_budget)
        _record_accuracy_trial(cat_stats, cat_counters, trials, benchmark_id, category, trial)
    return finalize_accuracy(cat_stats, trials)


def run_mc_accuracy_benchmark(
    scorer,
    tokenizer,
    samples,
    categories,
    *,
    desc,
    build_messages,
    build_trial,
    benchmark_id,
    choices,
):
    """Run a sampled benchmark with shared single-token multiple-choice bookkeeping."""
    cat_stats, cat_counters, trials = _init_accuracy_state(categories)
    for sample in tqdm(samples, desc=desc, leave=False):
        category = sample[0]
        pred = mc_answer(scorer, tokenizer, build_messages(sample), choices=choices)
        trial = build_trial(sample, pred)
        trial = add_generation_metadata(trial, None, None, None, None, None)
        _record_accuracy_trial(cat_stats, cat_counters, trials, benchmark_id, category, trial)
    return finalize_accuracy(cat_stats, trials)


def run_partial_credit_benchmark(
    generator,
    tokenizer,
    samples,
    categories,
    *,
    desc,
    build_messages,
    score_fn,
    build_trial,
    benchmark_id,
    stat_factory,
    finalize,
    category_fn=lambda sample: sample[0],
    max_new_tokens=512,
):
    """Run a sampled benchmark with shared generation and partial-credit stats."""
    cat_stats = {category: stat_factory() for category in categories}
    cat_counters = {category: 0 for category in categories}
    trials = []
    for sample in tqdm(samples, desc=desc, leave=False):
        category = category_fn(sample)
        token_budget = _token_budget(max_new_tokens, sample)
        response, n_tokens, truncated, n_input_tokens = generate_text_with_budget(
            generator,
            tokenizer,
            build_messages(sample),
            max_new_tokens=token_budget,
        )
        score, stat_updates, trial_extras = score_fn(sample, response)
        for key, value in stat_updates.items():
            cat_stats[category][key] += value
        cat_stats[category]["total"] += 1

        trial = build_trial(sample, response, score, trial_extras)
        trial = add_generation_metadata(trial, response, n_tokens, truncated, n_input_tokens, token_budget)
        trials.append(add_trial_metadata(trial, benchmark_id, category, cat_counters))
    return finalize(cat_stats, trials)


__all__ = ["run_accuracy_benchmark", "run_mc_accuracy_benchmark", "run_partial_credit_benchmark"]
