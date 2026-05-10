from __future__ import annotations

import concurrent.futures
import errno
import time
from dataclasses import dataclass
from typing import Any, Callable

import torch

from ruff_cm.llm.backends import GenerateResult, Generator, Message
from ruff_cm.llm.inference.generate import ParseFailure, parse_or_failure


def with_api_retry(
    generator: Generator,
    *,
    retry_schedule: tuple[float, ...] = (1, 2, 4, 8, 16, 32),
    retry_exceptions: tuple[type[Exception], ...] = (Exception,),
    timeout_per_attempt: float | None = None,
) -> Generator:
    """Wrap a generator with schedule-driven retry for transient API errors."""

    return _ApiRetryGenerator(generator, retry_schedule, retry_exceptions, timeout_per_attempt)


def with_oom_halving(
    generator: Generator,
    *,
    initial_batch_size: int = 8,
    min_batch_size: int = 1,
) -> Generator:
    """Wrap a generator with batch-size halving for CUDA OOM failures."""

    return _OomHalvingGenerator(generator, initial_batch_size, min_batch_size)


def with_parse_retry(
    generator: Generator,
    parser: Callable[[str], Any | None],
    *,
    max_retries: int = 3,
    on_failure: Callable[[ParseFailure], None] | None = None,
) -> Generator:
    """Wrap a generator so parse failures trigger fresh generation attempts."""

    return _ParseRetryGenerator(generator, parser, max_retries, on_failure)


def is_transient_api_error(exc: Exception) -> bool:
    status_code = _status_code(exc)
    if status_code is not None:
        return status_code in {408, 409, 425, 429, 500, 502, 503, 504}

    if isinstance(exc, (TimeoutError, ConnectionError, concurrent.futures.TimeoutError)):
        return True
    reset_codes = {errno.ECONNRESET, errno.ETIMEDOUT, errno.ECONNABORTED}
    if isinstance(exc, OSError) and getattr(exc, "errno", None) in reset_codes:
        return True

    name = type(exc).__name__.lower()
    message = str(exc).lower()
    transient_markers = ("timeout", "timed out", "connection reset", "connection aborted", "temporarily unavailable")
    return any(marker in name or marker in message for marker in transient_markers)


@dataclass
class _ApiRetryGenerator:
    generator: Generator
    retry_schedule: tuple[float, ...]
    retry_exceptions: tuple[type[Exception], ...]
    timeout_per_attempt: float | None

    @property
    def name(self) -> str:
        return self.generator.name

    @property
    def capabilities(self) -> frozenset[str]:
        return self.generator.capabilities

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> GenerateResult:
        # Each schedule entry is the delay before the next attempt.
        for retry_idx in range(len(self.retry_schedule) + 1):
            try:
                return _call_generate(
                    self.generator,
                    messages,
                    timeout_per_attempt=self.timeout_per_attempt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                    seed=seed,
                )
            except self.retry_exceptions as exc:
                if not is_transient_api_error(exc) or retry_idx == len(self.retry_schedule):
                    raise
                time.sleep(self.retry_schedule[retry_idx])
        raise RuntimeError("unreachable API retry state")


@dataclass
class _OomHalvingGenerator:
    generator: Generator
    initial_batch_size: int
    min_batch_size: int

    @property
    def name(self) -> str:
        return self.generator.name

    @property
    def capabilities(self) -> frozenset[str]:
        return self.generator.capabilities

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> GenerateResult:
        return self.generator.generate(messages, temperature=temperature, max_tokens=max_tokens, stop=stop, seed=seed)

    def generate_batch(
        self,
        messages_list: list[list[Message]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> list[GenerateResult]:
        batch_size = self.initial_batch_size
        while True:
            try:
                # Retry the same logical request list with a smaller caller-side batch.
                return _call_generate_batch(
                    self.generator,
                    messages_list,
                    batch_size=batch_size,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                    seed=seed,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch_size <= self.min_batch_size:
                    raise
                batch_size = max(self.min_batch_size, batch_size // 2)


@dataclass
class _ParseRetryGenerator:
    generator: Generator
    parser: Callable[[str], Any | None]
    max_retries: int
    on_failure: Callable[[ParseFailure], None] | None

    @property
    def name(self) -> str:
        return self.generator.name

    @property
    def capabilities(self) -> frozenset[str]:
        return self.generator.capabilities

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> GenerateResult:
        last_failure: ParseFailure | None = None
        for retry_idx in range(self.max_retries + 1):
            result = self.generator.generate(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
                seed=seed,
            )
            _, failure = parse_or_failure(result.text, self.parser, request_idx=0, attempt=retry_idx + 1)
            if failure is None:
                # Return the original result so finish_reason/raw metadata survive parsing.
                return result
            last_failure = failure
            if self.on_failure is not None:
                self.on_failure(failure)
        raise last_failure


def _call_generate(
    generator: Generator,
    messages: list[Message],
    *,
    timeout_per_attempt: float | None,
    temperature: float,
    max_tokens: int,
    stop: list[str] | None,
    seed: int | None,
) -> GenerateResult:
    if timeout_per_attempt is None:
        return generator.generate(messages, temperature=temperature, max_tokens=max_tokens, stop=stop, seed=seed)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(
        generator.generate,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop,
        seed=seed,
    )
    try:
        return future.result(timeout=timeout_per_attempt)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _call_generate_batch(
    generator: Generator,
    messages_list: list[list[Message]],
    *,
    batch_size: int,
    temperature: float,
    max_tokens: int,
    stop: list[str] | None,
    seed: int | None,
) -> list[GenerateResult]:
    generate_batch = getattr(generator, "generate_batch", None)
    if callable(generate_batch):
        return generate_batch(
            messages_list,
            batch_size=batch_size,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            seed=seed,
        )

    results: list[GenerateResult] = []
    for start in range(0, len(messages_list), batch_size):
        for messages in messages_list[start : start + batch_size]:
            results.append(
                generator.generate(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                    seed=seed,
                )
            )
    return results


def _status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


__all__ = [
    "is_transient_api_error",
    "with_api_retry",
    "with_oom_halving",
    "with_parse_retry",
]
