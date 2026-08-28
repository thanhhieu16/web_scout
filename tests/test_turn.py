import pytest

from app import turn
from web import store


class _LinearFakeGraph:
    """Enough of LangGraph's .stream() contract for one straight-through pass."""

    def __init__(self, final_answer="Final [1]."):
        self._final_answer = final_answer

    def stream(self, state, stream_mode="updates", **kwargs):
        values = dict(state)
        steps = [
            ("research", {"sources": []}),
            ("verify", {"sufficient": True}),
            ("answer", {"answer": self._final_answer}),
        ]
        for node, delta in steps:
            yield ("updates", {node: delta})
            values.update(delta)
            yield ("values", dict(values))


class _RaisingGraph:
    def stream(self, state, stream_mode="updates", **kwargs):
        raise RuntimeError("boom")
        yield  # pragma: no cover - makes this a generator function


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "test.db")
    store.init_db(path)
    return path


def test_run_chat_turn_streams_status_then_persists_result(db_path, monkeypatch):
    conv_id = store.create_conversation(db_path)
    monkeypatch.setattr(turn, "build_graph", lambda: _LinearFakeGraph())
    monkeypatch.setattr(turn, "condense_question", lambda history, question, **k: question)

    events = list(turn.run_chat_turn(db_path, conv_id, "Q?"))

    kinds = [k for k, _ in events]
    assert kinds == ["status", "status", "status", "result"]
    assert [p["node"] for _, p in events[:3]] == ["research", "verify", "answer"]
    assert events[-1][1]["answer"] == "Final [1]."
    stored = store.get_conversation(db_path, conv_id)
    assert stored["messages"][0]["question"] == "Q?"
    assert stored["messages"][0]["out"]["answer"] == "Final [1]."


def test_run_chat_turn_yields_error_without_persisting(db_path, monkeypatch):
    conv_id = store.create_conversation(db_path)
    monkeypatch.setattr(turn, "build_graph", lambda: _RaisingGraph())
    monkeypatch.setattr(turn, "condense_question", lambda history, question, **k: question)

    events = list(turn.run_chat_turn(db_path, conv_id, "Q?"))

    assert events[-1][0] == "error"
    assert "boom" in events[-1][1]["message"]
    assert store.get_conversation(db_path, conv_id)["messages"] == []


def test_run_chat_turn_condenses_using_stored_history(db_path, monkeypatch):
    conv_id = store.create_conversation(db_path)
    store.append_message(db_path, conv_id, "What is LangGraph?", {"answer": "A framework."})

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

    list(turn.run_chat_turn(db_path, conv_id, "and that?"))

    assert seen["history"] == [{"question": "What is LangGraph?", "answer": "A framework."}]
    assert seen["question"] == "and that?"
    assert captured_state["state"]["question"] == "standalone question"


def test_run_chat_turn_raises_keyerror_for_missing_conversation(db_path):
    with pytest.raises(KeyError):
        list(turn.run_chat_turn(db_path, 999, "Q?"))


def test_run_chat_turn_still_yields_result_when_persist_fails(db_path, monkeypatch, capsys):
    conv_id = store.create_conversation(db_path)
    monkeypatch.setattr(turn, "build_graph", lambda: _LinearFakeGraph())
    monkeypatch.setattr(turn, "condense_question", lambda history, question, **k: question)
    monkeypatch.setattr(
        turn.store, "append_message", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("locked"))
    )

    events = list(turn.run_chat_turn(db_path, conv_id, "Q?"))

    assert events[-1][0] == "result"
    assert events[-1][1]["answer"] == "Final [1]."
    assert "failed to persist" in capsys.readouterr().err
