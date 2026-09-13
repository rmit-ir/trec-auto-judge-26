from judges.rag26_arena.rubrics.cache import RubricLlmCache, cache_key


def test_cache_key_is_deterministic_and_order_independent_of_call_order():
    key1 = cache_key(stage="characterize", prompt_version="v1", model_id="m", tier="flex",
                      region="ap-southeast-2", messages=[{"role": "user", "content": "hi"}])
    key2 = cache_key(stage="characterize", prompt_version="v1", model_id="m", tier="flex",
                      region="ap-southeast-2", messages=[{"role": "user", "content": "hi"}])
    assert key1 == key2


def test_cache_key_changes_with_any_component():
    base = dict(stage="characterize", prompt_version="v1", model_id="m", tier="flex",
                region="ap-southeast-2", messages=[{"role": "user", "content": "hi"}])
    base_key = cache_key(**base)
    assert cache_key(**{**base, "prompt_version": "v2"}) != base_key
    assert cache_key(**{**base, "tier": "default"}) != base_key
    assert cache_key(**{**base, "model_id": "other"}) != base_key
    assert cache_key(**{**base, "messages": [{"role": "user", "content": "bye"}]}) != base_key


def test_cache_roundtrip(tmp_path):
    cache = RubricLlmCache(tmp_path / "rubric-cache" / "cache.db")
    key = cache_key(stage="characterize", prompt_version="v1", model_id="m", tier="flex",
                     region="ap-southeast-2", messages=[{"role": "user", "content": "hi"}])
    assert cache.get(key) is None

    cache.put(key, text='{"a": 1}', input_tokens=10, output_tokens=5)
    hit = cache.get(key)
    assert hit is not None
    assert hit.text == '{"a": 1}'
    assert hit.input_tokens == 10
    assert hit.output_tokens == 5
    assert hit.cache_hit is True
    cache.close()


def test_cache_creates_parent_directory(tmp_path):
    db_path = tmp_path / "does" / "not" / "exist" / "cache.db"
    cache = RubricLlmCache(db_path)
    assert db_path.parent.exists()
    cache.close()
