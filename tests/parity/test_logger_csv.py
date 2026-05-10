import csv
import json
from pathlib import Path

import pytest

from ruff_cm.logger import ABCLogger, CsvLogger, Logger, WandBLogger, WandbLogger, make_logger


@pytest.mark.parity
def test_logger_root_exports_logger_names():
    assert CsvLogger is not None
    assert WandbLogger is not None
    assert make_logger(["noop"], project="proj", run_name="run", config={}, base_dir=".").get_ckpt() is None
    assert issubclass(Logger, ABCLogger)
    assert WandBLogger is not None


@pytest.mark.parity
def test_csv_logger_extends_header_and_backfills_rows(tmp_path: Path):
    logger = CsvLogger(tmp_path)

    logger.log({"loss": 0.1, "acc": 0.9}, step=1)
    logger.log({"loss": 0.05, "ppl": 1.2}, step=2)

    with (tmp_path / "metrics.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert rows == [
        {"step": "1", "loss": "0.1", "acc": "0.9", "ppl": ""},
        {"step": "2", "loss": "0.05", "acc": "", "ppl": "1.2"},
    ]


@pytest.mark.parity
def test_csv_logger_latest_manifest_round_trips_portable_ckpt_path(tmp_path: Path):
    ckpt = tmp_path / "checkpoints" / "step-10.pt"
    ckpt.parent.mkdir()
    ckpt.write_text("weights", encoding="utf-8")
    logger = CsvLogger(tmp_path)

    logger.record_ckpt(ckpt, extras={"epoch": 3})

    manifest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert manifest == {"ckpt_rel": "checkpoints/step-10.pt", "epoch": 3}
    assert logger.get_ckpt() == ckpt


@pytest.mark.parity
def test_csv_logger_set_summary_accepts_dict_only(tmp_path: Path):
    logger = CsvLogger(tmp_path)

    logger.set_summary({"loss": 0.1, "acc": 0.9})
    logger.set_summary({"epoch": 3})

    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8")) == {
        "loss": 0.1,
        "acc": 0.9,
        "epoch": 3,
    }
    with pytest.raises(TypeError):
        logger.set_summary(loss=0.1)
