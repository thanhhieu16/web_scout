# WebScout ACP Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose WebScout as a third client surface (alongside the CLI and the web chat UI) that speaks Agent Client Protocol over stdio, so Zed editor can drive a research turn and see live per-node progress.

**Architecture:** Extract the chat-turn logic currently inlined in `web/server.py`'s `/api/chat` handler into a shared `app/turn.py::run_chat_turn()` generator. Add `app/acp_server.py`, a new `acp.Agent`-shaped class that calls the same `run_chat_turn()`, mapping its yielded events onto ACP `session/update` notifications (`plan` for node progress, `agent_message_chunk` for the final answer).

**Tech Stack:** `agent-client-protocol` (PyPI package `acp`), added as a new optional dependency group — same pattern as the existing `web` group.

**Spec:** [docs/superpowers/specs/2026-08-27-webscout-acp-integration-design.md](../specs/2026-08-27-webscout-acp-integration-design.md)

## Global Constraints

- No change to `app/graph.py`, the node contracts, or `stream_pipeline`'s existing contract (`("status", node_name)` then one `("result", out)`).
- ACP progress reporting is node-level only (`research`/`verify`/`answer`) — no per-tool-call visibility.
- No session resume in this plan: every ACP `session/new` creates a brand-new conversation row; no `session/load`.
- No per-session model or `max_iterations` override via ACP — always the process-global `config.yaml` defaults.
- A failed turn must never be persisted to `data/webscout.db` — this invariant already holds in `web/server.py` and must survive the `run_chat_turn` extraction unchanged.
- The `acp` dependency group must never be required by the default `uv sync` or by CI's offline test run — mirrors how the `web` group is isolated today.
- `tests/test_acp_server.py` must open with `pytest.importorskip("acp")`, mirroring `tests/test_server.py`'s `pytest.importorskip("fastapi")`.

---

### Task 1: Extract `app/turn.py` and refactor `web/server.py` to use it

**Files:**
- Create: `app/turn.py`
- Modify: `web/server.py:1-14` (imports), `web/server.py`'s `chat()` handler (currently lines 112-141)
- Modify: `tests/test_server.py` (monkeypatch targets only — 4 tests patch `server.build_graph`/`server.condense_question`)
- Test: `tests/test_turn.py`

**Interfaces:**
- Produces: `app.turn.run_chat_turn(db_path: str, conversation_id: int, question: str, max_iterations: int | None = None) -> Iterator[tuple[str, dict]]` — yields `("status", {"node": name})` zero or more times, then exactly one `("result", out_dict)` or one `("error", {"message": str})`. Raises `KeyError` if `conversation_id` doesn't exist in `db_path` (only if the generator is actually iterated — this is a generator function, so nothing runs until the first `next()`). On `"result"`, persists via `store.append_message` before yielding. On `"error"`, persists nothing.
- Consumes (from existing code, unchanged): `app.conversation.condense_question(history, question, settings=None, model=None) -> str`, `app.graph.build_graph() -> CompiledGraph`, `app.main.stream_pipeline(question, graph=None, max_iterations=None) -> Iterator[tuple[str, ...]]`, `web.store.get_conversation/append_message`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_turn.py`:

```python
import pytest

from app import turn
from web import store


class _LinearFakeGraph:
    """Enough of LangGraph's .stream() contract for one straight-through pass."""

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


class _RaisingGraph:
    def stream(self, state, stream_mode="updates"):
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
        def stream(self, state, stream_mode="updates"):
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_turn.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.turn'`

- [ ] **Step 3: Write `app/turn.py`**

```python
from collections.abc import Iterator

from app.conversation import condense_question
from app.graph import build_graph
from app.main import stream_pipeline
from web import store


def run_chat_turn(
    db_path: str,
    conversation_id: int,
    question: str,
    max_iterations: int | None = None,
) -> Iterator[tuple[str, dict]]:
    """Run one conversation turn, yielding progress then the final result.

    Mirrors `stream_pipeline`'s contract: yields ("status", {"node": name})
    zero or more times, then exactly one ("result", out_dict) or one
    ("error", {"message": str}) as the last item. On "result", the turn is
    persisted via `store.append_message` before it's yielded. On "error",
    nothing is persisted — a failed turn must never appear in history.

    Raises KeyError if `conversation_id` doesn't exist in `db_path` (only
    once the generator is actually iterated, since this is a generator
    function — nothing in this body runs until the first `next()`).
    """
    conversation = store.get_conversation(db_path, conversation_id)
    if conversation is None:
        raise KeyError(f"conversation {conversation_id} not found")
    try:
        history = [
            {"question": m["question"], "answer": m["out"].get("answer", "")}
            for m in conversation["messages"]
        ]
        condensed = condense_question(history, question)
        graph = build_graph()
        for kind, payload in stream_pipeline(
            condensed, graph=graph, max_iterations=max_iterations
        ):
            if kind == "status":
                yield ("status", {"node": payload})
            else:
                store.append_message(db_path, conversation_id, question, payload)
                yield ("result", payload)
    except Exception as exc:
        yield ("error", {"message": str(exc)})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_turn.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Refactor `web/server.py` to call `run_chat_turn`**

In `web/server.py`, replace these two import lines:

```python
from app.conversation import condense_question
from app.graph import build_graph
```

with:

```python
from app.turn import run_chat_turn
```

and change:

```python
from app.main import render_report_markdown, stream_pipeline
```

to:

```python
from app.main import render_report_markdown
```

Replace the `chat()` function body (currently the block starting `def chat(body: ChatRequest):` through the end of `gen()`) with:

```python
@app.post("/api/chat")
def chat(body: ChatRequest):
    db_path = _db_path()
    conversation = store.get_conversation(db_path, body.conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation not found")

    if body.model:
        override_model(body.model)

    def gen():
        try:
            for kind, payload in run_chat_turn(
                db_path, body.conversation_id, body.question, body.max_iterations
            ):
                if kind == "status":
                    yield _sse("status", {"node": payload["node"]})
                elif kind == "result":
                    yield _sse("result", payload)
                else:
                    yield _sse("error", payload)
        except Exception as exc:
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(gen(), media_type="text/event-stream")
```

The pre-check (`store.get_conversation` + `404` before the `StreamingResponse` is built) stays — it's what turns a missing conversation into a real HTTP `404` instead of a `200` response carrying an SSE `error` event. `run_chat_turn`'s own `KeyError` guard is a safety net for callers that don't pre-check (Task 2's ACP agent doesn't, since it always uses session ids it created itself).

- [ ] **Step 6: Update `tests/test_server.py`'s monkeypatch targets**

The refactor moves `build_graph` and `condense_question` out of `web.server`'s namespace and into `app.turn`'s. Four tests patch these — update each to patch the new location. Add near the top of the file (after the existing `import web.server as server` line):

```python
import app.turn as turn  # noqa: E402
```

Then in each of these four tests, change `monkeypatch.setattr(server, "build_graph", ...)` to `monkeypatch.setattr(turn, "build_graph", ...)`, and `monkeypatch.setattr(server, "condense_question", ...)` to `monkeypatch.setattr(turn, "condense_question", ...)`:

- `test_chat_streams_status_then_result`
- `test_chat_emits_error_event_on_failure`
- `test_chat_passes_history_from_stored_messages`
- `test_chat_persists_message_after_result`

No other change to these tests — same assertions, same fake graph classes, same expected behavior.

- [ ] **Step 7: Run the full suite to verify nothing broke**

Run: `uv run pytest tests/test_turn.py tests/test_server.py -v`
Expected: PASS, all tests (the four updated ones plus the new `test_turn.py` file)

Run: `uv run ruff check .`
Expected: no errors (catches the now-unused imports if Step 5 missed one)

- [ ] **Step 8: Commit**

```bash
git add app/turn.py web/server.py tests/test_turn.py tests/test_server.py
git commit -m "refactor: extract chat-turn logic into app/turn.py

web/server.py's /api/chat handler inlined load-history -> condense ->
stream-graph -> persist. Pulled into a shared run_chat_turn() generator so
the upcoming ACP agent (app/acp_server.py) can reuse it instead of
duplicating the same glue a third time."
```

---

### Task 2: Add `app/acp_server.py` (Zed integration)

**Files:**
- Create: `app/acp_server.py`
- Modify: `pyproject.toml` (new dependency group + console script)
- Modify: `README.md` (usage section)
- Test: `tests/test_acp_server.py`

**Interfaces:**
- Consumes: `app.turn.run_chat_turn` (Task 1), `app.main.require_openrouter_key`, `app.config.get_settings`, `web.store.init_db/create_conversation`.
- Produces: `app.acp_server.WebScoutAcpAgent` (class), `app.acp_server.main()` (the `webscout-acp` console-script entry point).

- [ ] **Step 1: Add the `acp` dependency group and console script**

In `pyproject.toml`, add `webscout-acp` under `[project.scripts]`:

```toml
[project.scripts]
webscout = "app.main:main"
webscout-web = "web.server:main"
webscout-acp = "app.acp_server:main"
```

Add a new `acp` group under `[dependency-groups]`:

```toml
[dependency-groups]
dev = ["pytest>=8.0", "ruff>=0.9"]
web = ["fastapi>=0.115", "uvicorn[standard]>=0.30"]
acp = ["agent-client-protocol>=0.12"]
```

- [ ] **Step 2: Install the group and verify it resolves**

Run: `uv sync --group acp`
Expected: installs `agent-client-protocol` (package name `acp` when imported) with no errors

- [ ] **Step 3: Write the failing test**

Create `tests/test_acp_server.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/test_acp_server.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.acp_server'`

- [ ] **Step 5: Write `app/acp_server.py`**

```python
import asyncio

from acp import (
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    plan_entry,
    run_agent,
    update_agent_message_text,
    update_plan,
)

from app.config import get_settings
from app.main import require_openrouter_key
from app.turn import run_chat_turn
from web import store

_STAGE_LABELS = {"research": "Research", "verify": "Verify", "answer": "Answer"}


def _stage_label(node: str, occurrence: int) -> str:
    base = _STAGE_LABELS.get(node, node)
    return base if occurrence == 1 else f"{base} — vòng {occurrence}"


def _db_path() -> str:
    path = get_settings().conversations_db_path
    store.init_db(path)
    return path


class WebScoutAcpAgent:
    """ACP agent exposing WebScout's research pipeline over stdio, for Zed."""

    def __init__(self) -> None:
        self.conn = None

    def on_connect(self, conn) -> None:
        self.conn = conn

    async def initialize(
        self, protocol_version, client_capabilities=None, client_info=None, **kwargs
    ) -> InitializeResponse:
        return InitializeResponse(protocol_version=protocol_version)

    async def new_session(
        self, cwd, additional_directories=None, mcp_servers=None, **kwargs
    ) -> NewSessionResponse:
        conversation_id = store.create_conversation(_db_path())
        return NewSessionResponse(session_id=str(conversation_id))

    async def prompt(self, session_id, prompt, **kwargs) -> PromptResponse:
        question = "".join(
            block.text for block in prompt if getattr(block, "type", None) == "text"
        )
        db_path = _db_path()
        conversation_id = int(session_id)

        entries = [plan_entry(_stage_label("research", 1), status="in_progress")]
        await self.conn.session_update(session_id, update_plan(entries))
        occurrences = {"research": 1}
        bootstrap_open = True

        answer_text = "(không có câu trả lời)"
        turn_events = run_chat_turn(db_path, conversation_id, question)
        sentinel = object()
        while True:
            item = await asyncio.to_thread(next, turn_events, sentinel)
            if item is sentinel:
                break
            kind, payload = item
            if kind == "status":
                # stream_pipeline only reports node COMPLETIONS, never starts, and
                # verify can loop back to research an unknown number of times, so
                # the next node to run can't be predicted here. Every event after
                # the bootstrap entry simply appends a new completed entry.
                node = payload["node"]
                n = occurrences.get(node, 0) + 1
                occurrences[node] = n
                label = _stage_label(node, n)
                if bootstrap_open:
                    entries[-1] = plan_entry(label, status="completed")
                    bootstrap_open = False
                else:
                    entries.append(plan_entry(label, status="completed"))
                await self.conn.session_update(session_id, update_plan(entries))
            elif kind == "result":
                answer_text = payload["answer"]
            else:
                answer_text = f"⚠ Lỗi: {payload['message']}"

        await self.conn.session_update(session_id, update_agent_message_text(answer_text))
        return PromptResponse(stop_reason="end_turn")


def main() -> None:
    """Entry point for the `webscout-acp` console script."""
    require_openrouter_key()
    asyncio.run(run_agent(WebScoutAcpAgent()))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_acp_server.py -v`
Expected: PASS (3 tests)

- [ ] **Step 7: Run the full offline suite plus ruff**

Run: `uv run pytest -m "not integration"`
Expected: PASS — the `acp` group is installed in this dev environment right now, so `test_acp_server.py` runs for real here; confirm it also *skips cleanly* by temporarily checking `python -c "import acp"` fails in an env without the group (not required to demonstrate in this task, just keep the `importorskip` guard as written).

Run: `uv run ruff check .`
Expected: no errors

- [ ] **Step 8: Add a README section**

In `README.md`, after the existing "**Browser chat UI**" paragraph (ends `...conversation history lives in the browser tab only.`) and before the `### What a run looks like` heading, insert:

```markdown
**Zed editor (ACP)** — drive WebScout as an agent inside [Zed](https://zed.dev/), with
per-node progress shown as a plan panel:

```powershell
uv sync --group acp
```

Then point Zed's agent-server config at the installed `webscout-acp` executable (its
path depends on your venv — usually `.venv/Scripts/webscout-acp.exe` on Windows). See
Zed's own [ACP documentation](https://agentclientprotocol.com/) for the exact
`settings.json` shape. Each Zed session starts a fresh conversation in the same
`data/webscout.db` used by the browser chat UI — no session resume yet, and no
per-session model override (both use `config.yaml`'s configured models).
```

Also update the "Project layout" section's `app/` block to add one line after `main.py`:

```text
app/
  main.py        CLI: run_pipeline (graph), run_question (agent only), markdown report
  turn.py        run_chat_turn: shared chat-turn logic (web UI + ACP agent)
  acp_server.py  ACP agent (Zed integration) over stdio — `webscout-acp` console script
  graph.py       LangGraph product loop (research -> verify -> answer)
  ...
```

(keep the rest of that block as-is — this only adds the two new lines in the right place)

- [ ] **Step 9: Commit**

```bash
git add app/acp_server.py pyproject.toml README.md tests/test_acp_server.py uv.lock
git commit -m "feat: add ACP (Agent Client Protocol) integration for Zed

New app/acp_server.py exposes the research pipeline as an ACP agent over
stdio, reusing app/turn.py's run_chat_turn (same conversation store the
web UI uses). Node-level status maps to ACP plan updates; the final
answer streams as one agent_message_chunk. New optional 'acp' dependency
group, isolated from the default install like the existing 'web' group."
```

---

## Manual verification (not automated)

After Task 2, verify against a real Zed install (this is the acceptance check for the end-to-end wiring — no automated test drives a real stdio round-trip, per the spec's Testing section):

1. `uv sync --group acp`
2. Add `webscout-acp`'s path to Zed's agent-servers config.
3. Open Zed, start a WebScout agent session, ask a question.
4. Confirm: a plan panel appears and updates through Research/Verify/Answer, the final answer renders as a message, and the conversation shows up in the web UI's sidebar (`uv run uvicorn web.server:app --reload` → `http://127.0.0.1:8000/`) since both share `data/webscout.db`.
