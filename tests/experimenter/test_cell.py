from __future__ import annotations

from pathlib import Path

from ruff_cm.experimenter.cell import Cell, CellId, expand_grid


def test_cell_id_encode_matches_autoname_style():
    assert CellId.encode({"BETA_R": 0.001, "REG": False}) == "BETAR-0.001_REG-False"


def test_cell_id_decode_round_trip_for_valid_string_values():
    encoded = CellId.encode({"task": "nback", "lure": False})
    assert CellId.decode(encoded) == {"task": "nback", "lure": "False"}


def test_cell_from_factors_sets_path(tmp_path: Path):
    cell = Cell.from_factors({"BETA_R": 0.001, "REG": False}, root=tmp_path)
    assert cell.name == "BETAR-0.001_REG-False"
    assert cell.path == tmp_path / "BETAR-0.001_REG-False"


def test_exists_and_mark_done(tmp_path: Path):
    cell = Cell.from_factors({"x": 1}, root=tmp_path)
    assert cell.exists() is False
    cell.mark_done()
    assert cell.exists() is True


def test_expand_grid_combinatorial(tmp_path: Path):
    cells = expand_grid({"x": [1, 2], "y": ["a", "b"]}, root=tmp_path)
    assert [cell.name for cell in cells] == ["x-1_y-a", "x-1_y-b", "x-2_y-a", "x-2_y-b"]
