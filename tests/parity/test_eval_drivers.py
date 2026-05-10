from __future__ import annotations

from ruff_cm.eval import run_accuracy_benchmark, stratified_sample_hf
from ruff_cm.llm.backends import GenerateResult, Message


class TinyTokenizer:
    model_max_length = 128

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return "\n".join(message["content"] for message in messages)

    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))


class EchoGenerator:
    name = "echo"
    capabilities = frozenset({"generate"})

    def generate(self, messages, *, temperature=0.0, max_tokens=256, stop=None, seed=None):
        return GenerateResult(text=messages[-1].content, finish_reason="stop")


def _build_messages(sample):
    return [Message("user", sample[2]["answer"])]


def _build_trial(sample, response):
    return {
        "category": sample[0],
        "pred": response,
        "gold": sample[2]["answer"],
        "correct": response == sample[2]["answer"],
        "score": None,
        "source": {"type": "fixture", "row_idx": sample[1]},
        "extra": {},
    }


def test_run_accuracy_benchmark_builds_pr_style_trials():
    samples = [("a", 0, {"answer": "yes"}), ("b", 1, {"answer": "no"})]

    result = run_accuracy_benchmark(
        EchoGenerator(),
        TinyTokenizer(),
        samples,
        ["a", "b"],
        desc="fixture",
        build_messages=_build_messages,
        build_trial=_build_trial,
        benchmark_id="toy",
        max_new_tokens=8,
    )

    assert result["score"] == 1.0
    assert [trial["sample_id"] for trial in result["trials"]] == ["toy:a:0", "toy:b:0"]
    assert result["trials"][0]["n_tokens"] == 3


def test_stratified_sample_hf_matches_pr_shuffle_then_take_order():
    data = [{"cat": "x", "id": i} for i in range(4)] + [{"cat": "y", "id": i} for i in range(10, 14)]
    import random

    rng = random.Random(13)
    actual = stratified_sample_hf(data, ["x", "y"], lambda row: row["cat"], 2, rng)

    ref_rng = random.Random(13)
    x_pool = list(enumerate(data[:4]))
    y_pool = list(enumerate(data[4:], start=4))
    ref_rng.shuffle(x_pool)
    ref_rng.shuffle(y_pool)
    expected = [("x", idx, row) for idx, row in x_pool[:2]] + [("y", idx, row) for idx, row in y_pool[:2]]
    assert actual == expected
