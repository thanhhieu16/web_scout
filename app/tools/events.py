from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

_INPUT_PREVIEW_CHARS = 200


class ToolEventCallback(BaseCallbackHandler):
    """Emits LangGraph custom-stream events for each tool call's lifecycle.

    The deep research agent is itself a compiled LangGraph graph, invoked
    via `agent.invoke(...)` from inside our own graph's research node. A
    lazy `get_stream_writer()` call made *during* that nested invocation
    (e.g. from inside this callback) resolves to the inner graph's own
    Pregel runtime, not our outer graph's — so it silently never reaches
    our stream. The fix is to resolve the writer once, in the outer node's
    own context *before* calling the inner agent, and hand it here.

    `writer` is optional so this class stays constructible (a no-op) for
    callers with no outer graph context — e.g. run_question's CLI path.

    Each call carries LangChain's own `run_id`, which is stable across a
    single tool invocation's start/end/error callbacks — the frontend uses
    it to update the line a "start" event created rather than appending a
    new one.
    """

    def __init__(self, writer=None) -> None:
        super().__init__()
        self._writer = writer

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        if self._writer is None:
            return
        inputs = kwargs.get("inputs")
        # Our tools take one string arg (query/url); LangChain's own input_str
        # is str(dict) — e.g. "{'query': '...'}" — so prefer the single raw
        # value from the clean `inputs` dict callback.run() also passes.
        if isinstance(inputs, dict) and len(inputs) == 1:
            preview = str(next(iter(inputs.values())))
        else:
            preview = input_str
        self._writer(
            {
                "run_id": str(run_id),
                "status": "start",
                "tool": serialized.get("name", "tool"),
                "input": preview[:_INPUT_PREVIEW_CHARS],
            }
        )

    def on_tool_end(self, output: Any, *, run_id: Any = None, **kwargs: Any) -> None:
        if self._writer is None:
            return
        self._writer({"run_id": str(run_id), "status": "done"})

    def on_tool_error(
        self, error: BaseException, *, run_id: Any = None, **kwargs: Any
    ) -> None:
        if self._writer is None:
            return
        self._writer(
            {
                "run_id": str(run_id),
                "status": "error",
                "error": str(error)[:_INPUT_PREVIEW_CHARS],
            }
        )
