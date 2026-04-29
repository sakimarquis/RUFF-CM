from __future__ import annotations

import pytest

from ruff_cm.llm.batch import JobManifest, RequestRecord, collect_ordered_results


def test_job_manifest_round_trips_json(tmp_path):
    manifest = JobManifest(
        run_id="run-1",
        provider="openai",
        request_path="requests.jsonl",
        metadata_path="requests_meta.jsonl",
        result_path="results.jsonl",
        records=[RequestRecord("row-0", 0, {"cell": "a"}), RequestRecord("row-1", 1, {"cell": "b"})],
    )
    path = tmp_path / "manifest.json"
    manifest.write(path)
    loaded = JobManifest.read(path)
    assert loaded == manifest


def test_collect_ordered_results_reorders_by_custom_id():
    records = [RequestRecord("row-0", 0, {}), RequestRecord("row-1", 1, {})]
    rows = {"row-1": "b", "row-0": "a"}
    assert collect_ordered_results(records, rows) == ["a", "b"]


def test_collect_ordered_results_fails_on_missing_custom_id():
    records = [RequestRecord("row-0", 0, {}), RequestRecord("row-1", 1, {})]
    with pytest.raises(KeyError):
        collect_ordered_results(records, {"row-0": "a"})
