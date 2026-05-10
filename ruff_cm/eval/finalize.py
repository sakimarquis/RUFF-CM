from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def finalize_accuracy(cat_stats: Mapping[str, Mapping[str, float]], trials: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(int(stats["total"]) for stats in cat_stats.values())
    correct = sum(float(stats["correct"]) for stats in cat_stats.values())
    categories = {}
    for category, stats in cat_stats.items():
        if stats["total"] > 0:
            categories[category] = {"score": stats["correct"] / stats["total"], **stats}
    return {"score": correct / total if total else 0.0, "correct": correct, "total": total, "categories": categories, "trials": trials}


def finalize_f1(cat_stats: Mapping[str, Mapping[str, float]], trials: list[dict[str, Any]], *, score_key: str = "em") -> dict[str, Any]:
    total = sum(int(stats["total"]) for stats in cat_stats.values())
    em_sum = sum(float(stats["em"]) for stats in cat_stats.values())
    f1_sum = sum(float(stats["f1"]) for stats in cat_stats.values())
    categories = {}
    for category, stats in cat_stats.items():
        if stats["total"] > 0:
            categories[category] = {
                "score": stats[score_key] / stats["total"],
                "em": stats["em"] / stats["total"],
                "f1": stats["f1"] / stats["total"],
                "total": stats["total"],
            }
    score = em_sum / total if score_key == "em" else f1_sum / total
    return {"score": score, "em": em_sum / total, "f1": f1_sum / total, "correct": int(em_sum), "total": total, "categories": categories, "trials": trials}


def finalize_partial_credit(
    cat_stats: Mapping[str, Mapping[str, float]],
    trials: list[dict[str, Any]],
    *,
    score_sum_key: str,
    score_name: str | None = None,
    rate_keys: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    rate_keys = rate_keys or {}
    total = sum(int(stats["total"]) for stats in cat_stats.values())
    score_sum = sum(float(stats[score_sum_key]) for stats in cat_stats.values())
    categories = {}
    for category, stats in cat_stats.items():
        if stats["total"] > 0:
            category_result = {"score": stats[score_sum_key] / stats["total"]}
            for output_key, stat_key in rate_keys.items():
                category_result[output_key] = stats[stat_key] / stats["total"]
            category_result["total"] = stats["total"]
            categories[category] = category_result

    score = score_sum / total if total else 0.0
    result = {"score": score, "correct": score_sum, "total": total, "categories": categories, "trials": trials}
    if score_name is not None:
        result[score_name] = score
    for output_key, stat_key in rate_keys.items():
        stat_sum = sum(float(stats[stat_key]) for stats in cat_stats.values())
        result[output_key] = stat_sum / total if total else 0.0
    return result


__all__ = ["finalize_accuracy", "finalize_f1", "finalize_partial_credit"]
