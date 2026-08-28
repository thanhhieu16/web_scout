import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import app.turn as turn  # noqa: E402
import web.server as server  # noqa: E402
from web import store  # noqa: E402


class _LinearFakeGraph:
    """Enough of LangGraph's .stream() contract for one straight-through pass."""

    def __init__(self, final_answer="Final [1]."):
        self._final_answer = final_answer

    def stream(self, state, stream_mode="updates", **kwargs):
        values = dict(state)
        steps = [
            (
                "research",
                {
                    "sources": [
                        {
                            "url": "https://s",
                            "title": "S",
                            "source_type": "secondary",
                            "excerpt": "",
                        }
                    ]
                },
            ),
            ("verify", {"sufficient": True}),
            ("answer", {"answer": self._final_answer}),
        ]
        for node, delta in steps:
            yield ("updates", {node: delta})
            values.update(delta)
            yield ("values", dict(values))


class _ToolEventGraph(_LinearFakeGraph):
    """Emits one custom tool event before the research node's update."""

    def stream(self, state, stream_mode="updates", **kwargs):
        yield ("custom", {"tool": "web_search", "input": "gia vang hom nay"})
        yield from super().stream(state, stream_mode=stream_mode)


class _RaisingGraph:
    def stream(self, state, stream_mode="updates", **kwargs):
        raise RuntimeError("boom")
        yield  # pragma: no cover - makes this a generator function


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    import json

    events = []
    for frame in text.split("\n\n"):
        if not frame.strip():
            continue
        event, data = "message", None
        for line in frame.split("\n"):
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        if data is not None:
            events.append((event, data))
    return events


client = TestClient(server.app)


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    from app.config import get_settings

    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("CONVERSATIONS_DB_PATH", db_path)
    get_settings.cache_clear()
    store.init_db(db_path)
    yield db_path
    get_settings.cache_clear()


def test_list_models_reports_shortlist_and_key_state(monkeypatch):
    from app.config import MODEL_CHOICES, get_settings

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    get_settings.cache_clear()
    try:
        resp = client.get("/api/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"] == list(MODEL_CHOICES)
        assert body["key_configured"] is True
    finally:
        get_settings.cache_clear()


def test_chat_streams_status_then_result(isolated_db, monkeypatch):
    conv_id = store.create_conversation(isolated_db)
    monkeypatch.setattr(turn, "build_graph", lambda: _LinearFakeGraph())
    monkeypatch.setattr(turn, "condense_question", lambda history, question, **k: question)
    resp = client.post("/api/chat", json={"conversation_id": conv_id, "question": "Q?"})
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    kinds = [e for e, _ in events]
    assert kinds == ["status", "status", "status", "result"]
    assert [d["node"] for e, d in events[:3]] == ["research", "verify", "answer"]
    assert events[-1][1]["answer"] == "Final [1]."


def test_chat_streams_tool_events(isolated_db, monkeypatch):
    conv_id = store.create_conversation(isolated_db)
    monkeypatch.setattr(turn, "build_graph", lambda: _ToolEventGraph())
    monkeypatch.setattr(turn, "condense_question", lambda history, question, **k: question)
    resp = client.post("/api/chat", json={"conversation_id": conv_id, "question": "Q?"})
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    assert events[0] == ("tool", {"tool": "web_search", "input": "gia vang hom nay"})


def test_chat_emits_error_event_on_failure(isolated_db, monkeypatch):
    conv_id = store.create_conversation(isolated_db)
    monkeypatch.setattr(turn, "build_graph", lambda: _RaisingGraph())
    monkeypatch.setattr(turn, "condense_question", lambda history, question, **k: question)
    resp = client.post("/api/chat", json={"conversation_id": conv_id, "question": "Q?"})
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    assert events[-1][0] == "error"
    assert "boom" in events[-1][1]["message"]
    assert store.get_conversation(isolated_db, conv_id)["messages"] == []


def test_chat_passes_history_from_stored_messages(isolated_db, monkeypatch):
    conv_id = store.create_conversation(isolated_db)
    store.append_message(isolated_db, conv_id, "What is LangGraph?", {"answer": "A framework."})

    seen = {}

    def fake_condense(history, question, **kwargs):
        seen["history"] = history
        seen["question"] = question
        return "standalone question"

    captured_state = {}

    class _CapturingGraph(_LinearFakeGraph):
        def stream(self, state, stream_mode="updates", **kwargs):
            captured_state["state"] = state
            yield from super().stream(state, stream_mode=stream_mode)

    monkeypatch.setattr(turn, "build_graph", lambda: _CapturingGraph())
    monkeypatch.setattr(turn, "condense_question", fake_condense)
    client.post("/api/chat", json={"conversation_id": conv_id, "question": "and that?"})
    assert seen["history"] == [{"question": "What is LangGraph?", "answer": "A framework."}]
    assert seen["question"] == "and that?"
    assert captured_state["state"]["question"] == "standalone question"


def test_chat_returns_404_for_missing_conversation(isolated_db):
    resp = client.post("/api/chat", json={"conversation_id": 999, "question": "Q?"})
    assert resp.status_code == 404


def test_chat_persists_message_after_result(isolated_db, monkeypatch):
    conv_id = store.create_conversation(isolated_db)
    monkeypatch.setattr(turn, "build_graph", lambda: _LinearFakeGraph())
    monkeypatch.setattr(turn, "condense_question", lambda history, question, **k: question)
    client.post("/api/chat", json={"conversation_id": conv_id, "question": "Q?"})
    conv = store.get_conversation(isolated_db, conv_id)
    assert len(conv["messages"]) == 1
    assert conv["messages"][0]["question"] == "Q?"
    assert conv["messages"][0]["out"]["answer"] == "Final [1]."


def test_create_conversation_returns_default_title(isolated_db):
    resp = client.post("/api/conversations")
    assert resp.status_code == 200
    body = resp.json()
    assert body["title"] == store.DEFAULT_TITLE
    assert isinstance(body["id"], int)


def test_list_conversations_empty_initially(isolated_db):
    resp = client.get("/api/conversations")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_conversation_returns_404_for_missing_id(isolated_db):
    resp = client.get("/api/conversations/999")
    assert resp.status_code == 404


def test_rename_conversation(isolated_db):
    conv_id = store.create_conversation(isolated_db)
    resp = client.patch(f"/api/conversations/{conv_id}", json={"title": "  New title  "})
    assert resp.status_code == 200
    assert resp.json()["title"] == "New title"
    assert store.list_conversations(isolated_db)[0]["title"] == "New title"


def test_rename_conversation_rejects_empty_title(isolated_db):
    conv_id = store.create_conversation(isolated_db)
    resp = client.patch(f"/api/conversations/{conv_id}", json={"title": "   "})
    assert resp.status_code == 400


def test_rename_conversation_404_for_missing_id(isolated_db):
    resp = client.patch("/api/conversations/999", json={"title": "x"})
    assert resp.status_code == 404


def test_delete_conversation(isolated_db):
    conv_id = store.create_conversation(isolated_db)
    resp = client.delete(f"/api/conversations/{conv_id}")
    assert resp.status_code == 204
    assert store.list_conversations(isolated_db) == []


def test_delete_conversation_404_for_missing_id(isolated_db):
    resp = client.delete("/api/conversations/999")
    assert resp.status_code == 404


def test_report_renders_markdown():
    resp = client.post(
        "/api/report",
        json={
            "question": "Q?",
            "out": {
                "answer": "A.",
                "sources": [],
                "findings": [],
                "sufficient": True,
                "iteration": 1,
                "search_calls": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
            },
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/markdown")
    assert "# WebScout Report" in resp.text
    assert "A." in resp.text


class _FakeRun:
    def __init__(self, name, run_type, children=None, tokens=10, cost=0.001, error=None):
        import uuid
        from datetime import UTC, datetime

        self.id = uuid.uuid4()
        self.name = name
        self.run_type = run_type
        self.start_time = datetime(2026, 1, 1, tzinfo=UTC)
        self.end_time = datetime(2026, 1, 1, 0, 0, 5, tzinfo=UTC)
        self.total_tokens = tokens
        self.total_cost = cost
        self.error = error
        self.inputs = {"question": "Q?"}
        self.outputs = {"answer": "A" * 3000}
        self.child_runs = children or []


class _FakeLangsmithClient:
    def __init__(self, run):
        self._run = run

    def read_run(self, run_id, load_child_runs=False):
        return self._run


def test_get_trace_returns_simplified_nested_run(monkeypatch):
    child = _FakeRun("web_search", "tool")
    root = _FakeRun("research", "chain", children=[child])
    monkeypatch.setattr(server, "Client", lambda: _FakeLangsmithClient(root))

    resp = client.get(f"/api/trace/{root.id}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "research"
    assert body["run_type"] == "chain"
    assert body["duration_seconds"] == 5.0
    assert body["total_tokens"] == 10
    assert body["total_cost"] == 0.001
    assert len(body["outputs_preview"]) == 2001  # 2000 chars + ellipsis
    assert len(body["children"]) == 1
    assert body["children"][0]["name"] == "web_search"


def test_get_trace_returns_502_on_langsmith_failure(monkeypatch):
    class _RaisingClient:
        def read_run(self, run_id, load_child_runs=False):
            raise RuntimeError("boom")

    monkeypatch.setattr(server, "Client", lambda: _RaisingClient())

    resp = client.get("/api/trace/does-not-matter")

    assert resp.status_code == 502


def test_index_page_served_at_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "<title>WebScout Chat</title>" in resp.text


def test_static_assets_served():
    resp = client.get("/static/app.js")
    assert resp.status_code == 200
    resp = client.get("/static/style.css")
    assert resp.status_code == 200
