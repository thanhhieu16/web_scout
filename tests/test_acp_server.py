import asyncio

import pytest

pytest.importorskip("acp")

from acp import text_block  # noqa: E402

import app.acp_server as acp_server  # noqa: E402
from web import store  # noqa: E402


class _FakeConn:
    def __init__(self):
        self.updates = []

    async def session_update(self, session_id, update):
        self.updates.append((session_id, update))


class _LinearFakeGraph:
    def __init__(self, final_answer="Final [1]."):
        self._final_answer = final_answer

    def stream(self, state, stream_mode="updates"):
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


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    from app.config import get_settings

    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("CONVERSATIONS_DB_PATH", db_path)
    get_settings.cache_clear()
    store.init_db(db_path)
    yield db_path
    get_settings.cache_clear()


def _fake_turn(status_nodes, result):
    def gen(db_path, conversation_id, question, max_iterations=None):
        for node in status_nodes:
            yield ("status", {"node": node})
        yield ("result", result)

    return gen


def test_module_import_disables_langsmith_tracing():
    import os

    assert os.environ.get("LANGSMITH_TRACING") == "false"


def test_new_session_creates_a_conversation_row(isolated_db):
    agent = acp_server.WebScoutAcpAgent()
    resp = asyncio.run(agent.new_session(cwd="."))
    assert store.get_conversation(isolated_db, int(resp.session_id)) is not None


def test_prompt_emits_plan_updates_then_final_answer(isolated_db, monkeypatch):
    conv_id = store.create_conversation(isolated_db)
    monkeypatch.setattr(
        acp_server,
        "run_chat_turn",
        _fake_turn(["research", "verify", "answer"], {"answer": "Final [1]."}),
    )
    agent = acp_server.WebScoutAcpAgent()
    conn = _FakeConn()
    agent.on_connect(conn)

    resp = asyncio.run(agent.prompt(session_id=str(conv_id), prompt=[text_block("Q?")]))

    assert resp.stop_reason == "end_turn"
    plan_updates = [u for _, u in conn.updates if u.session_update == "plan"]
    # 1 bootstrap ("Research" in_progress) + 3 status events = 4 plan updates
    assert len(plan_updates) == 4
    assert [e.status for e in plan_updates[-1].entries] == ["completed", "completed", "completed"]
    message_updates = [u for _, u in conn.updates if u.session_update == "agent_message_chunk"]
    assert message_updates[-1].content.text == "Final [1]."


def test_prompt_surfaces_a_failed_turn_as_a_message_not_a_crash(isolated_db, monkeypatch):
    conv_id = store.create_conversation(isolated_db)

    def failing_turn(db_path, conversation_id, question, max_iterations=None):
        yield ("error", {"message": "boom"})

    monkeypatch.setattr(acp_server, "run_chat_turn", failing_turn)
    agent = acp_server.WebScoutAcpAgent()
    conn = _FakeConn()
    agent.on_connect(conn)

    resp = asyncio.run(agent.prompt(session_id=str(conv_id), prompt=[text_block("Q?")]))

    assert resp.stop_reason == "end_turn"
    message_updates = [u for _, u in conn.updates if u.session_update == "agent_message_chunk"]
    assert "boom" in message_updates[-1].content.text


def test_prompt_runs_the_real_run_chat_turn_through_to_sqlite(isolated_db, monkeypatch):
    from app import turn

    conv_id = store.create_conversation(isolated_db)
    monkeypatch.setattr(turn, "build_graph", lambda: _LinearFakeGraph())
    monkeypatch.setattr(turn, "condense_question", lambda history, question, **k: question)

    agent = acp_server.WebScoutAcpAgent()
    conn = _FakeConn()
    agent.on_connect(conn)

    resp = asyncio.run(agent.prompt(session_id=str(conv_id), prompt=[text_block("Q?")]))

    assert resp.stop_reason == "end_turn"
    plan_updates = [u for _, u in conn.updates if u.session_update == "plan"]
    assert [e.status for e in plan_updates[-1].entries] == ["completed", "completed", "completed"]
    message_updates = [u for _, u in conn.updates if u.session_update == "agent_message_chunk"]
    assert message_updates[-1].content.text == "Final [1]."
    stored = store.get_conversation(isolated_db, conv_id)
    assert stored["messages"][0]["out"]["answer"] == "Final [1]."
