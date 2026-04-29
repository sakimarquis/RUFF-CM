from __future__ import annotations

import json

from ruff_cm.store.prefix_cache import load_prefix_cache, serialize_prefix_cache


def old_number_game_serialize(cache):
    return {json.dumps(list(k), separators=(",", ":")): v for k, v in cache.items()}


def old_number_game_load(raw):
    return {tuple(json.loads(k)): v for k, v in raw.items()}


def test_prefix_cache_codec_matches_number_game_fixture():
    cache = {("small", "blue"): {"A": 0.4}, ("small", "blue", "round"): {"A": 0.7}}
    assert serialize_prefix_cache(cache) == old_number_game_serialize(cache)
    assert load_prefix_cache(serialize_prefix_cache(cache)) == old_number_game_load(old_number_game_serialize(cache))
