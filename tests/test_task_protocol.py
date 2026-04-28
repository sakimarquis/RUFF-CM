from __future__ import annotations

from ruff_cm.task_protocol import TaskProtocol


def correct(answer: str, sample: dict) -> bool:
    return answer == sample["answer"]


def validity(answer: str, sample: dict) -> dict:
    return {"valid": bool(answer)}


def coverage(sample: dict, steps: list[str]) -> list[float]:
    return [1.0 for _ in steps]


def coverage_trace(sample: dict, steps: list[str]) -> dict:
    return {"steps": steps}


def test_task_protocol_properties():
    protocol = TaskProtocol(
        dataset_name="toy",
        answer_correctness_fn=correct,
        validity_fn=validity,
        validity_kind="formal",
        coverage_fn=coverage,
        coverage_trace_fn=coverage_trace,
    )
    assert protocol.has_validity is True
    assert protocol.has_formal_validity is True
    assert protocol.has_coverage is True
    assert protocol.has_coverage_trace is True
    assert protocol.answer_correctness_fn("A", {"answer": "A"}) is True


def test_task_protocol_allows_missing_optional_functions():
    protocol = TaskProtocol("toy", correct, None, None, None, None)
    assert protocol.has_validity is False
    assert protocol.has_formal_validity is False
    assert protocol.has_coverage is False
    assert protocol.has_coverage_trace is False
