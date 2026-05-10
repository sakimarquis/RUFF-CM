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


def test_stage_is_a_named_callable_with_default_enabled_true():
    from ruff_cm.pipeline.stage import Stage

    calls = []
    stage = Stage(name="generate", run=lambda ctx: calls.append(ctx["dataset"]))
    stage.run({"dataset": "ds1"})
    assert calls == ["ds1"]
    assert stage.enabled({}) is True
    assert stage.name == "generate"


def test_stage_with_custom_enabled_predicate():
    from ruff_cm.pipeline.stage import Stage

    stage = Stage(
        name="verifier",
        run=lambda ctx: None,
        enabled=lambda ctx: ctx.get("verifier_on", False),
    )
    assert stage.enabled({}) is False
    assert stage.enabled({"verifier_on": True}) is True


def test_stage_is_frozen():
    import pytest
    from ruff_cm.pipeline.stage import Stage

    stage = Stage(name="x", run=lambda ctx: None)
    with pytest.raises(Exception):
        stage.name = "y"
