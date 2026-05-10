from ruff_cm.eval.trial import Trial, add_generation_metadata, validate_trial


def test_validate_trial_accepts_minimal_generic_trial_without_sft_fields():
    trial = {
        "benchmark": "mybench",
        "sample_id": "mybench:cat:0",
        "category": "cat",
        "response": "yes",
        "pred": "yes",
        "gold": "yes",
        "correct": True,
        "score": None,
        "source": {"type": "synthetic"},
        "extra": {},
    }
    validate_trial(trial)


def test_to_dict_omits_unset_optional_trial_fields():
    trial = Trial(
        sample_id="mybench:cat:0",
        response="yes",
        pred="yes",
        gold="yes",
        correct=True,
        score=None,
        source={"type": "synthetic"},
        benchmark="mybench",
        category="cat",
    )
    row = trial.to_dict()

    assert "stage" not in row
    assert "epoch" not in row
    assert "prompt_truncated_to" not in row
    assert "max_new_tokens" not in row


def test_add_generation_metadata_omits_none_values():
    row = add_generation_metadata(
        {
            "benchmark": "mybench",
            "sample_id": "mybench:cat:0",
            "category": "cat",
            "response": None,
            "pred": "yes",
            "gold": "yes",
            "correct": True,
            "score": None,
            "source": {"type": "synthetic"},
            "extra": {},
        },
        response="yes",
        n_tokens=None,
        truncated=None,
        n_input_tokens=None,
        max_new_tokens=None,
    )

    assert row["response"] == "yes"
    assert "n_tokens" not in row
    assert "truncated" not in row
    assert "prompt_truncated_to" not in row
    assert "max_new_tokens" not in row
