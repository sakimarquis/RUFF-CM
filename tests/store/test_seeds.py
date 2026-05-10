from ruff_cm.store.seeds import derive_seed, seed_namespace_metadata


def test_derive_seed_is_deterministic_across_calls():
    a = derive_seed(42, "prontoqa", "generate", "train_gen")
    b = derive_seed(42, "prontoqa", "generate", "train_gen")
    assert a == b


def test_derive_seed_changes_with_namespace():
    a = derive_seed(42, "prontoqa", "generate", "train_gen")
    b = derive_seed(42, "prontoqa", "generate", "test_gen")
    assert a != b


def test_derive_seed_changes_with_root():
    a = derive_seed(42, "x")
    b = derive_seed(43, "x")
    assert a != b


def test_derive_seed_returns_uint32():
    seed = derive_seed(0, "ns")
    assert 0 <= seed < 2**32


def test_derive_seed_accepts_mixed_part_types():
    seed_a = derive_seed(7, 1, "foo", (3,))
    seed_b = derive_seed(7, 1, "foo", (3,))
    assert seed_a == seed_b


def test_seed_namespace_metadata_returns_named_seeds():
    md = seed_namespace_metadata(
        42,
        namespaces={
            "train_seed": ("prontoqa", "generate", "train_gen"),
            "test_seed": ("prontoqa", "generate", "test_gen"),
        },
    )
    assert set(md) == {"train_seed", "test_seed"}
    assert md["train_seed"] == derive_seed(42, "prontoqa", "generate", "train_gen")
    assert md["test_seed"] != md["train_seed"]


def test_seed_namespace_metadata_passes_extras_through():
    md = seed_namespace_metadata(
        42,
        namespaces={"data_seed": ("ds", "data")},
        extras={"experiment_seed": 42, "model_name": "qwen3-4b"},
    )
    assert md["experiment_seed"] == 42
    assert md["model_name"] == "qwen3-4b"
    assert md["data_seed"] == derive_seed(42, "ds", "data")


def test_seed_namespace_metadata_overrides_collision_raises():
    import pytest

    with pytest.raises(ValueError):
        seed_namespace_metadata(
            42,
            namespaces={"data_seed": ("ds", "data")},
            extras={"data_seed": 99},
        )


def test_seed_everything_runs_without_errors():
    from ruff_cm.store.seeds import seed_everything

    seed_everything(0)
    seed_everything(2**32 - 1)


def test_seed_everything_reproduces_numpy_draws():
    import pytest

    np = pytest.importorskip("numpy")
    from ruff_cm.store.seeds import seed_everything

    seed_everything(123)
    a = np.random.rand(5)
    seed_everything(123)
    b = np.random.rand(5)
    assert (a == b).all()


def test_seed_everything_reproduces_torch_draws():
    import pytest

    torch = pytest.importorskip("torch")
    from ruff_cm.store.seeds import seed_everything

    seed_everything(456)
    a = torch.randn(5)
    seed_everything(456)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_seed_metadata_feeds_artifact_key_fingerprint():
    from ruff_cm.store import ArtifactKey
    from ruff_cm.store.seeds import seed_namespace_metadata

    base_md = seed_namespace_metadata(
        42,
        namespaces={
            "train_seed": ("ds", "generate", "train_gen"),
            "test_seed": ("ds", "generate", "test_gen"),
        },
        extras={"experiment_seed": 42, "model_name": "qwen3-4b"},
    )
    key_a = ArtifactKey("hidden", ("qwen3-4b", "ds"), base_md)
    key_a_again = ArtifactKey("hidden", ("qwen3-4b", "ds"), base_md)
    assert key_a.fingerprint() == key_a_again.fingerprint()


def test_seed_metadata_fingerprint_changes_when_root_changes():
    from ruff_cm.store import ArtifactKey
    from ruff_cm.store.seeds import seed_namespace_metadata

    md_42 = seed_namespace_metadata(
        42, namespaces={"split": ("ds", "split")}, extras={"experiment_seed": 42}
    )
    md_43 = seed_namespace_metadata(
        43, namespaces={"split": ("ds", "split")}, extras={"experiment_seed": 43}
    )
    key_42 = ArtifactKey("probe", ("ds",), md_42)
    key_43 = ArtifactKey("probe", ("ds",), md_43)
    assert key_42.fingerprint() != key_43.fingerprint()


def test_seed_metadata_fingerprint_is_order_insensitive_in_extras():
    from ruff_cm.store import ArtifactKey
    from ruff_cm.store.seeds import seed_namespace_metadata

    md_a = seed_namespace_metadata(
        7, namespaces={"split": ("ds",)}, extras={"a": 1, "b": 2}
    )
    md_b = seed_namespace_metadata(
        7, namespaces={"split": ("ds",)}, extras={"b": 2, "a": 1}
    )
    assert ArtifactKey("k", (), md_a).fingerprint() == ArtifactKey("k", (), md_b).fingerprint()
