from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .create_training_configs import autoname, vary_config


@dataclass(frozen=True)
class Cell:
    factors: dict[str, Any]
    name: str
    path: Path

    @classmethod
    def from_factors(cls, factors: dict[str, Any], *, root: Path, name_keys: list[str] | None = None) -> Cell:
        name = CellId.encode(factors, name_keys=name_keys)
        return cls(factors=dict(factors), name=name, path=root / name)

    def exists(self, marker: str = "_done") -> bool:
        return (self.path / marker).exists()

    def mark_done(self, marker: str = "_done") -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / marker).write_text("done\n")


class CellId:
    @staticmethod
    def encode(factors: dict[str, Any], name_keys: list[str] | None = None) -> str:
        return autoname([dict(factors)], [dict(factors)], name_keys=name_keys)[0]

    @staticmethod
    def decode(name: str) -> dict[str, str]:
        if name == "":
            return {}
        return dict(part.split("-", 1) for part in name.split("_"))


def expand_grid(
    axes: dict[str, list[Any]], *, mode: str = "combinatorial", name_keys: list[str] | None = None, root: Path
) -> list[Cell]:
    configs, names = vary_config({}, axes, mode, name_keys=name_keys)
    return [Cell(factors=config, name=name, path=root / name) for config, name in zip(configs, names)]
