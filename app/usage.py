import threading


class UsageCollector:
    """Collects usage from HTTP calls made outside LangChain's message stream.

    The `web_search` tool issues its own OpenRouter request, so its tokens and
    cost never reach `sum_usage`. Nodes call `drain()` once per iteration; the
    reset is what stops iteration 2 from re-counting iteration 1.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tokens = 0
        self._cost = 0.0
        self._searches = 0

    def add(self, tokens: int = 0, cost: float = 0.0, searches: int = 0) -> None:
        with self._lock:
            self._tokens += int(tokens or 0)
            self._cost += float(cost or 0.0)
            self._searches += int(searches or 0)

    def drain(self) -> tuple[int, float, int]:
        """Return the accumulated totals and reset to zero."""
        with self._lock:
            totals = (self._tokens, round(self._cost, 6), self._searches)
            self._tokens = 0
            self._cost = 0.0
            self._searches = 0
        return totals
