import threading


class SourceRegistry:
    """Hands out stable `Sn` source IDs, one per URL, for one research turn.

    `web_search` and `web_fetch` share one registry and print the ID next to
    every URL they return, so the agent cites `[S3]` by copying it instead of
    counting "the order it first used each source" — a count the agent and
    `build_sources` never reliably agreed on. A URL keeps its ID across
    research iterations because the registry lives as long as the agent.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ids: dict[str, str] = {}

    def id_for(self, url: str) -> str:
        with self._lock:
            if url not in self._ids:
                self._ids[url] = f"S{len(self._ids) + 1}"
            return self._ids[url]
