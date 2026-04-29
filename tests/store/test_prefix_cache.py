from __future__ import annotations

from ruff_cm.store.prefix_cache import (
    load_prefix_cache,
    parse_prefix_key,
    prefix_key,
    reconstruct_trajectory,
    serialize_prefix_cache,
)


def test_prefix_key_round_trips_tuple_prefix():
    key = prefix_key(("small", "blue", 3))
    assert key == '["small","blue",3]'
    assert parse_prefix_key(key) == ("small", "blue", 3)


def test_serialize_and_load_prefix_cache():
    cache = {("small",): {"p": 0.2}, ("small", "blue"): {"p": 0.7}}
    raw = serialize_prefix_cache(cache)
    assert raw == {'["small"]': {"p": 0.2}, '["small","blue"]': {"p": 0.7}}
    assert load_prefix_cache(raw) == cache


def test_load_prefix_cache_accepts_none_as_empty_cache():
    assert load_prefix_cache(None) == {}


def test_reconstruct_trajectory_reads_each_prefix_value():
    cache = {("a",): 1, ("a", "b"): 2, ("a", "b", "c"): 3}
    assert reconstruct_trajectory(("a", "b", "c"), cache) == [1, 2, 3]
