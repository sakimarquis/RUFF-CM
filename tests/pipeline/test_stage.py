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


def test_pipeline_runs_stages_in_declared_order():
    from ruff_cm.pipeline.stage import Pipeline, Stage

    calls = []
    pipe = Pipeline(
        [
            Stage(name="a", run=lambda ctx: calls.append("a")),
            Stage(name="b", run=lambda ctx: calls.append("b")),
            Stage(name="c", run=lambda ctx: calls.append("c")),
        ]
    )
    pipe.run({}, log=lambda _msg: None)
    assert calls == ["a", "b", "c"]


def test_pipeline_skips_disabled_stages_silently():
    from ruff_cm.pipeline.stage import Pipeline, Stage

    calls = []
    pipe = Pipeline(
        [
            Stage(name="a", run=lambda ctx: calls.append("a")),
            Stage(name="b", run=lambda ctx: calls.append("b"), enabled=lambda ctx: False),
            Stage(name="c", run=lambda ctx: calls.append("c")),
        ]
    )
    log_lines = []
    pipe.run({}, log=log_lines.append)
    assert calls == ["a", "c"]
    assert "b" not in log_lines


def test_pipeline_emits_banner_for_each_enabled_stage():
    from ruff_cm.pipeline.stage import Pipeline, Stage

    log_lines = []
    pipe = Pipeline([
        Stage(name="phase-1", run=lambda ctx: None),
        Stage(name="phase-2", run=lambda ctx: None),
    ])
    pipe.run({}, log=log_lines.append)
    assert "phase-1" in log_lines
    assert "phase-2" in log_lines


def test_pipeline_propagates_stage_exceptions():
    import pytest
    from ruff_cm.pipeline.stage import Pipeline, Stage

    def boom(ctx):
        raise RuntimeError("kaboom")

    pipe = Pipeline([Stage(name="boom", run=boom)])
    with pytest.raises(RuntimeError, match="kaboom"):
        pipe.run({}, log=lambda _msg: None)


def test_pipeline_passes_same_ctx_to_every_stage():
    from ruff_cm.pipeline.stage import Pipeline, Stage

    def write_a(ctx):
        ctx["a"] = 1

    def read_a(ctx):
        ctx["seen"] = ctx["a"]

    pipe = Pipeline(
        [
            Stage(name="write", run=write_a),
            Stage(name="read", run=read_a),
        ]
    )
    ctx = {}
    pipe.run(ctx, log=lambda _msg: None)
    assert ctx == {"a": 1, "seen": 1}
