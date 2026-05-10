from __future__ import annotations

import pytest

from ruff_cm.eval import TRIAL_REQUIRED_FIELDS, Trial, make_sample_id, validate_trial


def test_trial_constructor_validates_plan_acceptance_shape():
    trial = Trial(
        stage="test",
        epoch=0,
        sample_id="s1",
        response="A",
        pred="A",
        gold="A",
        correct=True,
        score=1.0,
        source="mmlu",
        extra={},
    )

    validate_trial(trial)
    row = trial.to_dict()
    assert tuple(row) == TRIAL_REQUIRED_FIELDS
    assert row["source"] == {"type": "mmlu"}
    assert make_sample_id("mmlu", "stem", 3) == "mmlu:stem:3"


def test_validate_trial_keeps_pr_required_field_contract():
    row = Trial(
        stage=0,
        epoch=1.0,
        benchmark="bench",
        sample_id="bench:cat:0",
        category="cat",
        response="A",
        pred="A",
        gold="B",
        correct=False,
        score=None,
        n_tokens=1,
        truncated=False,
        prompt_truncated_to=12,
        max_new_tokens=5,
        source={"type": "fixture", "row_idx": 7},
        extra={},
    ).to_dict()

    validate_trial(row)
    del row["gold"]
    with pytest.raises(ValueError, match="missing required fields"):
        validate_trial(row)
