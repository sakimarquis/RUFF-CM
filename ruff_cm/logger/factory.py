from __future__ import annotations

from ruff_cm.logger.csv import CsvLogger
from ruff_cm.logger.multi import MultiLogger, NoopLogger
from ruff_cm.logger.wandb import WandbLogger


_REGISTRY = {"csv": CsvLogger, "noop": NoopLogger, "wandb": WandbLogger}


def _collapse_loggers(loggers):
    if not loggers:
        return NoopLogger()
    return loggers[0] if len(loggers) == 1 else MultiLogger(loggers)


def _require_backend_list(backends):
    if not isinstance(backends, list):
        raise TypeError("logger backends must be a list[str]")
    return backends


def make_logger(backends: list[str], *, project: str, run_name: str, config: dict, base_dir):
    loggers = [
        _REGISTRY[backend].start(project=project, run_name=run_name, config=config, base_dir=base_dir)
        for backend in _require_backend_list(backends)
    ]
    return _collapse_loggers(loggers)


def resume_logger(backends: list[str], *, project: str, run_name: str, base_dir):
    loggers = [
        _REGISTRY[backend].resume(project=project, run_name=run_name, base_dir=base_dir)
        for backend in _require_backend_list(backends)
    ]
    return _collapse_loggers(loggers)
