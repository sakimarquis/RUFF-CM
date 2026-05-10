from pathlib import Path

import pytest

from ruff_cm.logger import CsvLogger, MultiLogger, NoopLogger, make_logger, resume_logger
import ruff_cm.logger.factory as logger_factory


class FakeWandbLogger(NoopLogger):
    started = []
    resumed = []

    @classmethod
    def start(cls, *, project: str, run_name: str, config: dict, base_dir: Path | str):
        cls.started.append((project, run_name, config, Path(base_dir)))
        return cls()

    @classmethod
    def resume(cls, *, project: str, run_name: str, base_dir: Path | str):
        cls.resumed.append((project, run_name, Path(base_dir)))
        return cls()


def test_make_logger_accepts_backend_lists_and_collapses_singletons(tmp_path: Path):
    logger = make_logger(["csv"], project="proj", run_name="run", config={"seed": 0}, base_dir=tmp_path)

    assert isinstance(logger, CsvLogger)
    assert logger.out_dir == tmp_path / "logs" / "proj" / "run"


def test_make_logger_rejects_string_kind(tmp_path: Path):
    with pytest.raises(TypeError):
        make_logger("csv", project="proj", run_name="run", config={}, base_dir=tmp_path)


def test_make_logger_list_builds_multi_logger_with_shared_kwargs(monkeypatch, tmp_path: Path):
    FakeWandbLogger.started = []
    monkeypatch.setitem(logger_factory._REGISTRY, "wandb", FakeWandbLogger)

    logger = make_logger(["wandb", "csv"], project="proj", run_name="run", config={"seed": 1}, base_dir=tmp_path)

    assert isinstance(logger, MultiLogger)
    assert [type(child) for child in logger.loggers] == [FakeWandbLogger, CsvLogger]
    assert FakeWandbLogger.started == [("proj", "run", {"seed": 1}, tmp_path)]


def test_resume_logger_uses_backend_lists_and_collapses_singletons(tmp_path: Path):
    make_logger(["csv"], project="proj", run_name="run", config={"seed": 0}, base_dir=tmp_path)

    logger = resume_logger(["csv"], project="proj", run_name="run", base_dir=tmp_path)

    assert isinstance(logger, CsvLogger)
    assert logger.config == {"seed": 0}


def test_resume_logger_list_builds_multi_logger(monkeypatch, tmp_path: Path):
    FakeWandbLogger.resumed = []
    monkeypatch.setitem(logger_factory._REGISTRY, "wandb", FakeWandbLogger)

    logger = resume_logger(["wandb", "csv"], project="proj", run_name="run", base_dir=tmp_path)

    assert isinstance(logger, MultiLogger)
    assert [type(child) for child in logger.loggers] == [FakeWandbLogger, CsvLogger]
    assert FakeWandbLogger.resumed == [("proj", "run", tmp_path)]
