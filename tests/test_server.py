import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import web.server as server  # noqa: E402


class _LinearFakeGraph:
    """Enough of LangGraph's .stream() contract for one straight-through pass."""

    def __init__(self, final_answer="Final [1]."):
        self._final_answer = final_answer

    def stream(self, state, stream_mode="updates"):
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


class _RaisingGraph:
    def stream(self, state, stream_mode="updates"):
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


def test_chat_streams_status_then_result(monkeypatch):
    monkeypatch.setattr(server, "build_graph", lambda: _LinearFakeGraph())
    monkeypatch.setattr(server, "condense_question", lambda history, question, **k: question)
    resp = client.post("/api/chat", json={"question": "Q?", "history": []})
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    kinds = [e for e, _ in events]
    assert kinds == ["status", "status", "status", "result"]
    assert [d["node"] for e, d in events[:3]] == ["research", "verify", "answer"]
    assert events[-1][1]["answer"] == "Final [1]."


def test_chat_emits_error_event_on_failure(monkeypatch):
    monkeypatch.setattr(server, "build_graph", lambda: _RaisingGraph())
    monkeypatch.setattr(server, "condense_question", lambda history, question, **k: question)
    resp = client.post("/api/chat", json={"question": "Q?", "history": []})
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    assert events[-1][0] == "error"
    assert "boom" in events[-1][1]["message"]


def test_chat_passes_condensed_question_and_history(monkeypatch):
    seen = {}

    def fake_condense(history, question, **kwargs):
        seen["history"] = history
        seen["question"] = question
        return "standalone question"

    captured_state = {}

    class _CapturingGraph(_LinearFakeGraph):
        def stream(self, state, stream_mode="updates"):
            captured_state["state"] = state
            yield from super().stream(state, stream_mode=stream_mode)

    monkeypatch.setattr(server, "build_graph", lambda: _CapturingGraph())
    monkeypatch.setattr(server, "condense_question", fake_condense)
    client.post(
        "/api/chat",
        json={
            "question": "and that?",
            "history": [{"question": "What is LangGraph?", "answer": "A framework."}],
        },
    )
    assert seen["history"] == [{"question": "What is LangGraph?", "answer": "A framework."}]
    assert seen["question"] == "and that?"
    assert captured_state["state"]["question"] == "standalone question"


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
