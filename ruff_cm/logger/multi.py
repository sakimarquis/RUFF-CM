from __future__ import annotations

from pathlib import Path

from ruff_cm.logger.base import Logger


class NoopLogger:
    @classmethod
    def start(cls, **kwargs) -> "NoopLogger":
        return cls()

    @classmethod
    def resume(cls, **kwargs) -> "NoopLogger":
        return cls()

    def log(self, metrics: dict, *, step: int | None = None) -> None:
        pass

    def set_summary(self, metrics: dict) -> None:
        pass

    def record_ckpt(self, ckpt_path: Path, *, extras: dict | None = None) -> None:
        pass

    def get_ckpt(self) -> Path | None:
        return None

    def hf_report_to(self) -> list[str]:
        return []

    def hf_callbacks(self) -> list:
        return []

    def finish(self) -> None:
        pass


class MultiLogger:
    def __init__(self, loggers: list[Logger]):
        self.loggers = loggers

    @property
    def config(self) -> dict:
        for logger in self.loggers:
            config = getattr(logger, "config", {})
            if config:
                return config
        return {}

    def log(self, metrics: dict, *, step: int | None = None) -> None:
        for logger in self.loggers:
            logger.log(metrics, step=step)

    def set_summary(self, metrics: dict) -> None:
        for logger in self.loggers:
            logger.set_summary(metrics)

    def record_ckpt(self, ckpt_path: Path, *, extras: dict | None = None) -> None:
        for logger in self.loggers:
            logger.record_ckpt(ckpt_path, extras=extras)

    def get_ckpt(self) -> Path | None:
        for logger in self.loggers:
            ckpt = logger.get_ckpt()
            if ckpt is not None:
                return ckpt
        return None

    def hf_report_to(self) -> list[str]:
        return [name for logger in self.loggers for name in logger.hf_report_to()]

    def hf_callbacks(self) -> list:
        return [callback for logger in self.loggers for callback in logger.hf_callbacks()]

    def finish(self) -> None:
        for logger in self.loggers:
            logger.finish()
