import time
from collections import OrderedDict
from typing import Any


class TTLCache:
    """Process-lifetime, in-memory cache with TTL expiry and LRU eviction.

    `now` is an injectable time source (defaults to `time.monotonic`) so
    tests can advance the clock deterministically instead of sleeping.
    """

    def __init__(self, ttl_seconds: float = 1800.0, max_size: int = 500, now=time.monotonic):
        self._ttl_seconds = ttl_seconds
        self._max_size = max_size
        self._now = now
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if self._now() >= expires_at:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (self._now() + self._ttl_seconds, value)
        self._store.move_to_end(key)
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)


search_cache = TTLCache()
fetch_cache = TTLCache()
