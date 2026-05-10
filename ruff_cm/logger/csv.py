from __future__ import annotations

import csv
import json
from pathlib import Path
from ruff_cm.experimenter.io import from_portable_relpath, portable_relpath
from ruff_cm.logger.hf_callbacks import CsvLoggingCallback


class CsvLogger:
    """Streaming CSV logger with widened headers and a portable latest-checkpoint manifest."""

    def __init__(self, out_dir: Path | str, *, base_dir: Path | str | None = None):
        self.out_dir = Path(out_dir)
        self.base_dir = Path(base_dir) if base_dir is not None else self.out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.out_dir / "metrics.csv"
        self.summary_path = self.out_dir / "summary.json"
        self.latest_path = self.out_dir / "latest.json"
        self._fields = self._read_header()

    @classmethod
    def start(cls, *, project: str, run_name: str, config: dict, base_dir: Path | str) -> "CsvLogger":
        logger = cls(Path(base_dir) / "logs" / project / run_name, base_dir=base_dir)
        (logger.out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
        return logger

    @classmethod
    def resume(cls, *, project: str, run_name: str, base_dir: Path | str) -> "CsvLogger":
        return cls(Path(base_dir) / "logs" / project / run_name, base_dir=base_dir)

    @property
    def config(self) -> dict:
        path = self.out_dir / "config.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def log(self, metrics: dict, *, step: int | None = None) -> None:
        row = {"step": step, **metrics} if step is not None else dict(metrics)
        new_keys = [key for key in row if key not in self._fields]
        if new_keys:
            self._extend_header(new_keys)
        with self.metrics_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self._fields).writerow({key: row.get(key, "") for key in self._fields})

    def set_summary(self, metrics: dict) -> None:
        summary = json.loads(self.summary_path.read_text(encoding="utf-8")) if self.summary_path.exists() else {}
        summary.update(metrics)
        self.summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    def record_ckpt(self, ckpt_path: Path, *, extras: dict | None = None) -> None:
        payload = {"ckpt_rel": portable_relpath(Path(ckpt_path), self.base_dir)}
        if extras:
            payload.update(extras)
        self.latest_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def get_ckpt(self) -> Path | None:
        if not self.latest_path.exists():
            return None
        payload = json.loads(self.latest_path.read_text(encoding="utf-8"))
        return from_portable_relpath(payload["ckpt_rel"], self.base_dir)

    def hf_report_to(self) -> list[str]:
        return []

    def hf_callbacks(self) -> list:
        return [CsvLoggingCallback(self)]

    def finish(self) -> None:
        pass

    def _read_header(self) -> list[str]:
        if not self.metrics_path.exists():
            return []
        with self.metrics_path.open(newline="", encoding="utf-8") as f:
            return next(csv.reader(f), [])

    def _extend_header(self, new_keys: list[str]) -> None:
        # New metric keys are rare; rewriting keeps each row rectangular and CSV-reader friendly.
        old_rows = []
        if self.metrics_path.exists() and self._fields:
            with self.metrics_path.open(newline="", encoding="utf-8") as f:
                old_rows = list(csv.DictReader(f))
        self._fields = self._fields + new_keys
        with self.metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fields)
            writer.writeheader()
            for row in old_rows:
                writer.writerow({key: row.get(key, "") for key in self._fields})
