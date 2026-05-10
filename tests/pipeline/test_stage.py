from ruff_cm.pipeline.stage import banner


def test_banner_writes_three_lines_through_log_callable():
    lines = []
    banner("PHASE 1", log=lines.append)
    assert len(lines) == 3
    assert all("=" * 60 == lines[i] for i in (0, 2))
    assert lines[1] == "PHASE 1"


def test_banner_default_log_is_print(capsys):
    banner("HELLO")
    captured = capsys.readouterr()
    assert "HELLO" in captured.out
    assert "=" * 60 in captured.out
