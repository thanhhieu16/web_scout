from app.tools.cache import TTLCache


def test_get_returns_none_for_missing_key():
    cache = TTLCache()
    assert cache.get("missing") is None


def test_set_then_get_returns_value():
    cache = TTLCache()
    cache.set("k", "v")
    assert cache.get("k") == "v"


def test_entry_expires_after_ttl():
    clock = {"t": 0.0}
    cache = TTLCache(ttl_seconds=10.0, now=lambda: clock["t"])
    cache.set("k", "v")
    clock["t"] = 9.0
    assert cache.get("k") == "v"
    clock["t"] = 10.0
    assert cache.get("k") is None


def test_expired_entry_is_removed_from_store():
    clock = {"t": 0.0}
    cache = TTLCache(ttl_seconds=10.0, now=lambda: clock["t"])
    cache.set("k", "v")
    clock["t"] = 10.0
    cache.get("k")
    assert len(cache._store) == 0


def test_evicts_least_recently_used_when_max_size_exceeded():
    cache = TTLCache(max_size=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_get_refreshes_lru_order():
    cache = TTLCache(max_size=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")
    cache.set("c", 3)
    assert cache.get("a") == 1
    assert cache.get("b") is None
    assert cache.get("c") == 3
