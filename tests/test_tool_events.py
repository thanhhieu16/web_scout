from app.tools.events import ToolEventCallback


def test_on_tool_start_writes_tool_and_input():
    written = []
    ToolEventCallback(writer=written.append).on_tool_start(
        {"name": "web_search"}, "gia vang hom nay", run_id="r1"
    )
    assert written == [
        {"run_id": "r1", "status": "start", "tool": "web_search", "input": "gia vang hom nay"}
    ]


def test_on_tool_start_prefers_clean_inputs_dict_over_str_repr():
    """LangChain's own input_str is str(dict) (e.g. "{'query': 'x'}"); when the
    clean `inputs` dict is available (LangChain's BaseTool.run always passes
    it) and has exactly one value, that value is what should be shown."""
    written = []
    ToolEventCallback(writer=written.append).on_tool_start(
        {"name": "web_search"},
        "{'query': 'gia vang hom nay'}",
        run_id="r1",
        inputs={"query": "gia vang hom nay"},
    )
    assert written[0]["input"] == "gia vang hom nay"


def test_on_tool_start_truncates_long_input():
    written = []
    long_input = "x" * 500
    ToolEventCallback(writer=written.append).on_tool_start(
        {"name": "web_fetch"}, long_input, run_id="r1"
    )
    assert len(written[0]["input"]) == 200


def test_on_tool_start_is_a_noop_without_a_writer():
    """writer=None (no outer graph context resolved it, e.g. run_question's
    CLI path) must not raise — the callback silently drops the event."""
    ToolEventCallback(writer=None).on_tool_start({"name": "web_search"}, "q", run_id="r1")


def test_on_tool_end_writes_run_id_and_done_status():
    written = []
    ToolEventCallback(writer=written.append).on_tool_end("some output", run_id="r1")
    assert written == [{"run_id": "r1", "status": "done"}]


def test_on_tool_error_writes_run_id_and_error_message():
    written = []
    ToolEventCallback(writer=written.append).on_tool_error(
        ValueError("timed out"), run_id="r1"
    )
    assert written == [{"run_id": "r1", "status": "error", "error": "timed out"}]


def test_on_tool_error_truncates_long_message():
    written = []
    ToolEventCallback(writer=written.append).on_tool_error(
        ValueError("x" * 500), run_id="r1"
    )
    assert len(written[0]["error"]) == 200


def test_on_tool_end_and_on_tool_error_are_noops_without_a_writer():
    ToolEventCallback(writer=None).on_tool_end("out", run_id="r1")
    ToolEventCallback(writer=None).on_tool_error(ValueError("x"), run_id="r1")
