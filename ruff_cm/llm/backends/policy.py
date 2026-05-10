from __future__ import annotations

from typing import Any, Literal

OPENAI_BATCH_MIN_TRIALS = 20
OPENAI_BATCH_POLL_INTERVAL = 30
OPENAI_BATCH_MAX_PROMPT_CHARS = 3_000_000


def default_api_run_policy() -> dict[str, Any]:
    return {
        "batch_min_trials": OPENAI_BATCH_MIN_TRIALS,
        "batch_poll_interval": OPENAI_BATCH_POLL_INTERVAL,
        "batch_max_prompt_chars": OPENAI_BATCH_MAX_PROMPT_CHARS,
        "flex_small_runs": True,
    }


def api_run_policy(
    provider: str,
    request_count: int,
    *,
    batch_min_trials: int = OPENAI_BATCH_MIN_TRIALS,
    batch_max_chars: int = OPENAI_BATCH_MAX_PROMPT_CHARS,
    prompt_chars: int = 0,
    flex_small_runs: bool = True,
    force_flex: bool = False,
) -> Literal["immediate", "batch", "flex"]:
    supports_batch = provider in {"openai", "google_cloud"}
    supports_flex = provider == "openai"
    if supports_batch and request_count >= batch_min_trials and prompt_chars <= batch_max_chars and not force_flex:
        return "batch"
    if supports_flex and flex_small_runs:
        return "flex"
    return "immediate"


def should_use_batch(backend: Any, n_requests: int, policy: dict[str, Any] | None = None) -> bool:
    policy = policy or default_api_run_policy()
    return bool(getattr(backend, "supports_batch", False)) and n_requests >= int(policy["batch_min_trials"])


def immediate_service_tier(backend: Any, n_requests: int, policy: dict[str, Any] | None = None) -> str | None:
    policy = policy or default_api_run_policy()
    if (
        getattr(backend, "supports_flex", False)
        and policy["flex_small_runs"]
        and (
            policy.get("force_flex")
            or not getattr(backend, "supports_batch", False)
            or n_requests < int(policy["batch_min_trials"])
        )
    ):
        return "flex"
    return None


def flex_only_policy(policy: dict[str, Any] | None = None) -> dict[str, Any]:
    updated = dict(policy or default_api_run_policy())
    updated["force_flex"] = True
    return updated


def batch_state(batch: Any) -> str:
    return getattr(batch, "status", None) or str(getattr(batch, "state")).removeprefix("JobState.")


def batch_done(batch: Any) -> bool:
    return batch_state(batch) in {
        "completed",
        "failed",
        "expired",
        "cancelled",
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_PAUSED",
    }


def batch_progress(batch: Any) -> str | None:
    counts = getattr(batch, "request_counts", None)
    if counts is not None:
        return f"{counts.completed}/{counts.total} complete, {counts.failed} failed"
    stats = getattr(batch, "completion_stats", None)
    if stats is not None:
        return f"{stats.successful_count} succeeded, {stats.failed_count} failed"
    return None


def batch_name(batch: Any) -> str:
    return getattr(batch, "id", None) or batch.name
