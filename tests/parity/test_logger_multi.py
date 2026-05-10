from pathlib import Path

import pytest

from ruff_cm.logger import CsvLogger, MultiLogger, NoopLogger


class RecordingLogger(NoopLogger):
    def __init__(self, ckpt: Path | None = None):
        self.events = []
        self._ckpt = ckpt

    def log(self, metrics: dict, *, step: int | None = None) -> None:
        self.events.append(("log", metrics, step))

    def set_summary(self, metrics: dict) -> None:
        self.events.append(("summary", metrics))

    def record_ckpt(self, ckpt_path: Path, *, extras: dict | None = None) -> None:
        self.events.append(("ckpt", ckpt_path, extras))

    def get_ckpt(self) -> Path | None:
        return self._ckpt


@pytest.mark.parity
def test_multi_logger_fans_out_and_returns_first_ckpt(tmp_path: Path):
    first = RecordingLogger()
    second = RecordingLogger(tmp_path / "model.pt")
    logger = MultiLogger([first, second])

    logger.log({"loss": 0.2}, step=4)
    logger.set_summary({"best_loss": 0.2})
    logger.record_ckpt(tmp_path / "model.pt", extras={"score": 1.0})

    assert first.events == [
        ("log", {"loss": 0.2}, 4),
        ("summary", {"best_loss": 0.2}),
        ("ckpt", tmp_path / "model.pt", {"score": 1.0}),
    ]
    assert second.events == first.events
    assert logger.get_ckpt() == tmp_path / "model.pt"


@pytest.mark.parity
def test_noop_logger_accepts_all_protocol_calls(tmp_path: Path):
    logger = NoopLogger()

    logger.log({"loss": 0.2}, step=1)
    logger.set_summary({"loss": 0.2})
    with pytest.raises(TypeError):
        logger.set_summary(loss=0.2)
    logger.record_ckpt(tmp_path / "missing.pt")
    logger.finish()

    assert logger.get_ckpt() is None
    assert logger.hf_report_to() == []
    assert logger.hf_callbacks() == []


@pytest.mark.parity
def test_multi_logger_merges_hf_integrations(tmp_path: Path):
    csv_logger = CsvLogger(tmp_path)
    logger = MultiLogger([NoopLogger(), csv_logger])

    assert logger.hf_report_to() == []
    assert len(logger.hf_callbacks()) == 1
