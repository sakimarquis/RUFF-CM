import pytest

from ruff_cm.eval.verifier import StepResult, Verifier, VerifierRegistry, VerifierResult, step_row, summarize


def test_step_row_marks_no_error_when_description_is_none():
    row = step_row(1, None, verified=True)
    assert isinstance(row, StepResult)
    assert row.step_num == 1
    assert row.has_local_error is False
    assert row.error_description is None
    assert row.verified is True


def test_step_row_marks_error_when_description_is_present():
    row = step_row(2, "premise not established", verified=True)
    assert row.has_local_error is True
    assert row.error_description == "premise not established"


def test_step_row_supports_unverified_meta_step():
    row = step_row(3, None, verified=False)
    assert row.verified is False
    assert row.has_local_error is False


def test_summarize_counts_verified_rows_as_actual_steps():
    rows = (
        step_row(1, None, verified=True),
        step_row(2, "bad rule", verified=True),
        step_row(3, None, verified=False),
    )
    result = summarize(rows, optimal_steps=2)
    assert result.actual_steps == 2
    assert result.optimal_steps == 2
    assert result.excess_steps == 0


def test_summarize_excess_steps_is_none_when_optimal_unknown():
    rows = (step_row(1, None, verified=True),)
    result = summarize(rows, optimal_steps=None)
    assert result.optimal_steps is None
    assert result.excess_steps is None
    assert result.actual_steps == 1


def test_summarize_passes_extras_through():
    result = summarize((), optimal_steps=None, dataset="prontoqa", n_hops=3)
    assert result.extras["dataset"] == "prontoqa"
    assert result.extras["n_hops"] == 3


def test_verifier_result_steps_is_immutable_tuple():
    result = summarize((step_row(1, None, verified=True),), optimal_steps=None)
    assert isinstance(result.steps, tuple)
    with pytest.raises(TypeError):
        result.steps[0] = step_row(99, None, verified=False)


def _passthrough_verifier(cot_text, problem):
    rows = tuple(
        step_row(idx + 1, None, verified=True)
        for idx, line in enumerate(cot_text.splitlines())
        if line.strip()
    )
    return summarize(rows, optimal_steps=problem.get("n_hops"))


def test_registry_register_and_get_round_trip():
    registry = VerifierRegistry()
    registry.register("prontoqa", _passthrough_verifier)
    fetched = registry.get("prontoqa")
    assert fetched is _passthrough_verifier


def test_registry_contains_uses_registered_name():
    registry = VerifierRegistry()
    registry.register("prontoqa", _passthrough_verifier)
    assert "prontoqa" in registry
    assert "nonexistent" not in registry


def test_registry_get_missing_raises_keyerror_with_known_names():
    registry = VerifierRegistry()
    registry.register("a", _passthrough_verifier)
    registry.register("b", _passthrough_verifier)
    with pytest.raises(KeyError) as excinfo:
        registry.get("c")
    msg = str(excinfo.value)
    assert "c" in msg
    assert "a" in msg and "b" in msg


def test_registry_register_duplicate_raises():
    registry = VerifierRegistry()
    registry.register("x", _passthrough_verifier)
    with pytest.raises(ValueError):
        registry.register("x", _passthrough_verifier)


def test_registry_names_is_sorted_tuple():
    registry = VerifierRegistry()
    registry.register("zebra", _passthrough_verifier)
    registry.register("alpha", _passthrough_verifier)
    assert registry.names() == ("alpha", "zebra")


def test_verifier_protocol_call_round_trip():
    registry = VerifierRegistry()
    registry.register("simple", _passthrough_verifier)
    verifier: Verifier = registry.get("simple")
    result = verifier("Step 1: foo\nStep 2: bar\n", {"n_hops": 2})
    assert isinstance(result, VerifierResult)
    assert result.actual_steps == 2
    assert result.excess_steps == 0
