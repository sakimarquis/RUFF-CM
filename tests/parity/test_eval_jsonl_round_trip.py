from __future__ import annotations

from ruff_cm.eval import Trial, append_benchmark_trials, init_benchmark_trial_jsonls, read_trials


def _trial() -> Trial:
    return Trial(
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


def test_direct_benchmark_jsonl_round_trip(tmp_path):
    init_benchmark_trial_jsonls(tmp_path, ["acc"])
    append_benchmark_trials(tmp_path, "acc", [_trial()])

    rows = read_trials(tmp_path / "acc.jsonl")
    assert rows == [_trial().to_dict()]


def test_pr_run_directory_jsonl_shape(tmp_path):
    init_benchmark_trial_jsonls(tmp_path, "run1", ["acc"])
    result = {"acc": {"trials": [_trial().to_dict()]}}
    append_benchmark_trials(tmp_path, "run1", result, stage=2, epoch=3.5)

    rows = read_trials(tmp_path / "run1_trials" / "acc.jsonl")
    assert rows[0]["stage"] == 2
    assert rows[0]["epoch"] == 3.5
