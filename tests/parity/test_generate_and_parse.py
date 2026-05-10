from __future__ import annotations

import json

from ruff_cm.llm.backends import GenerateResult, Message
from ruff_cm.llm.inference.generate import generate_and_parse, save_parse_failure_report


class PromptEchoGenerator:
    name = "echo"
    capabilities = frozenset({"generate"})

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def generate(self, messages, *, temperature=0.0, max_tokens=256, stop=None, seed=None):
        text = self.outputs[self.calls]
        self.calls += 1
        return GenerateResult(text=text, finish_reason="stop")


def _parse_int(text: str):
    return int(text) if text.isdigit() else None


def test_generate_and_parse_returns_one_success_or_failure_per_request():
    generator = PromptEchoGenerator(["1", "bad", "still bad", "nope"])
    report = generate_and_parse(
        generator,
        [[Message("user", "one")], [Message("user", "two")]],
        _parse_int,
        max_retries=2,
        progress=False,
    )

    assert report.successes == [1]
    assert len(report.failures) == 1
    assert report.failures[0].request_idx == 1
    assert report.failures[0].raw_text == "nope"
    assert report.n_attempts == 4
    assert len(report.successes) + len(report.failures) == 2


def test_generate_and_parse_writes_parseable_jsonl_failure_log(tmp_path):
    generator = PromptEchoGenerator(["bad", "nope"])
    log_path = tmp_path / "parse_failures.jsonl"
    report = generate_and_parse(
        generator,
        [[Message("user", "one")]],
        _parse_int,
        max_retries=1,
        failure_log_path=log_path,
        progress=False,
    )

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["type"] == "parse_failure"
    assert rows[0]["request_idx"] == report.failures[0].request_idx
    assert rows[1] == {"type": "summary", "successes": 0, "failures": 1, "n_attempts": 2}


def test_save_parse_failure_report_round_trips_report_shape(tmp_path):
    generator = PromptEchoGenerator(["bad"])
    report = generate_and_parse(
        generator,
        [[Message("user", "one")]],
        lambda text: None,
        max_retries=0,
        progress=False,
    )
    path = tmp_path / "report.jsonl"

    save_parse_failure_report(report, path)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["type"] for row in rows] == ["parse_failure", "summary"]
