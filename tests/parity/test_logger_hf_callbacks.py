from pathlib import Path
from types import SimpleNamespace

import pytest

from ruff_cm.logger import CsvLogger


@pytest.mark.hf
@pytest.mark.parity
def test_csv_hf_callback_writes_trainer_log_row(tmp_path: Path):
    transformers = pytest.importorskip("transformers")
    logger = CsvLogger(tmp_path)

    [callback] = logger.hf_callbacks()
    assert isinstance(callback, transformers.TrainerCallback)

    callback.on_log(None, SimpleNamespace(global_step=7), None, logs={"loss": 0.3})

    assert (tmp_path / "metrics.csv").read_text(encoding="utf-8").splitlines() == ["step,loss", "7,0.3"]
