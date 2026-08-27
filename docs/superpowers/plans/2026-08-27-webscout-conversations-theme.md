# WebScout Conversations, Theme Toggle, Status Trace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add server-side persisted multi-conversation support, a light/dark theme toggle, and a per-turn status trace to the existing WebScout chat UI.

**Architecture:** A new `web/store.py` (stdlib `sqlite3`, no new dependency) owns a `data/webscout.db` file with `conversations`/`messages` tables. `web/server.py` gains five CRUD routes and `POST /api/chat` switches from client-supplied `history` to a required `conversation_id`, loading history from the DB and persisting each turn after the pipeline finishes. The frontend (`web/static/*`) gets a conversation list in the sidebar, a `data-theme` attribute toggle backed by a second CSS token palette, and a per-turn chip trail showing which LangGraph node ran.

**Tech Stack:** FastAPI, stdlib `sqlite3`, vanilla JS/CSS (unchanged from the existing chat UI — no new frontend dependency).

**Spec:** [docs/superpowers/specs/2026-08-27-webscout-conversations-theme-design.md](../specs/2026-08-27-webscout-conversations-theme-design.md)

## Global Constraints

- No new Python dependency — `sqlite3` is stdlib. The `web` dependency group (`fastapi`, `uvicorn`) is unchanged.
- DB file: `data/webscout.db`, path from `Settings.conversations_db_path`, default `str(REPO_ROOT / "data" / "webscout.db")`. `data/` is gitignored.
- Every sqlite connection: `sqlite3.connect(db_path, timeout=5)` immediately followed by `conn.execute("PRAGMA foreign_keys = ON")`.
- Timestamps: `datetime.now(timezone.utc).isoformat()`, stored as TEXT.
- Default conversation title: `"Cuộc hội thoại mới"` (exported as `store.DEFAULT_TITLE`).
- Auto-title rule: only when a conversation's message count is 0 **and** its title still equals `store.DEFAULT_TITLE` — set to `question[:40] + ("…" if len(question) > 40 else "")`.
- `store.py` functions never raise a bare `sqlite3.Error` up to callers for a missing id — they return `None`/`False` or raise `KeyError`, per function, so routes can turn that into a clean `404` (never a `500`).
- Theme persistence key: `localStorage["webscout-theme"]`, values `"light"` or `"dark"`. Absent key → follow `prefers-color-scheme` without writing to storage.
- No new frontend test tooling — the project has none today (see `CLAUDE.md`'s Testing conventions). Frontend-only tasks are verified by running the app and checking in a browser, same as the existing chat UI's tasks.
- `tests/test_server.py` stays `pytest.importorskip("fastapi")`-gated; nothing in this plan changes that.

---

### Task 1: `web/store.py` — SQLite conversation store

**Files:**
- Create: `web/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: nothing from this plan (only stdlib `sqlite3`, `json`, `datetime`, `pathlib`).
- Produces (used by Task 2):
  - `DEFAULT_TITLE: str`
  - `init_db(db_path: str) -> None`
  - `create_conversation(db_path: str, title: str = DEFAULT_TITLE) -> int`
  - `list_conversations(db_path: str) -> list[dict]` — each `{"id": int, "title": str, "updated_at": str}`
  - `get_conversation(db_path: str, conversation_id: int) -> dict | None` — `{"id": int, "title": str, "messages": [{"question": str, "out": dict}, ...]}` or `None`
  - `append_message(db_path: str, conversation_id: int, question: str, out: dict) -> None` — raises `KeyError` if `conversation_id` doesn't exist
  - `rename_conversation(db_path: str, conversation_id: int, title: str) -> bool`
  - `delete_conversation(db_path: str, conversation_id: int) -> bool`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_store.py`:

```python
import pytest

from web import store


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "test.db")
    store.init_db(path)
    return path


def test_create_conversation_returns_id_with_default_title(db_path):
    conv_id = store.create_conversation(db_path)
    conversations = store.list_conversations(db_path)
    assert len(conversations) == 1
    assert conversations[0]["id"] == conv_id
    assert conversations[0]["title"] == store.DEFAULT_TITLE
    assert conversations[0]["updated_at"]


def test_list_conversations_orders_by_updated_at_desc(db_path):
    first = store.create_conversation(db_path)
    second = store.create_conversation(db_path)
    store.append_message(db_path, first, "hi", {"answer": "hello"})
    ids = [c["id"] for c in store.list_conversations(db_path)]
    assert ids == [first, second]


def test_get_conversation_returns_messages_in_order(db_path):
    conv_id = store.create_conversation(db_path)
    store.append_message(db_path, conv_id, "q1", {"answer": "a1"})
    store.append_message(db_path, conv_id, "q2", {"answer": "a2"})
    conv = store.get_conversation(db_path, conv_id)
    assert conv["messages"] == [
        {"question": "q1", "out": {"answer": "a1"}},
        {"question": "q2", "out": {"answer": "a2"}},
    ]


def test_get_conversation_returns_none_for_missing_id(db_path):
    assert store.get_conversation(db_path, 999) is None


def test_append_message_sets_title_from_first_question(db_path):
    conv_id = store.create_conversation(db_path)
    store.append_message(db_path, conv_id, "What is LangGraph?", {"answer": "..."})
    conv = store.get_conversation(db_path, conv_id)
    assert conv["title"] == "What is LangGraph?"


def test_append_message_truncates_long_first_question(db_path):
    conv_id = store.create_conversation(db_path)
    long_q = "x" * 60
    store.append_message(db_path, conv_id, long_q, {"answer": "..."})
    conv = store.get_conversation(db_path, conv_id)
    assert conv["title"] == "x" * 40 + "…"


def test_append_message_does_not_overwrite_title_on_second_message(db_path):
    conv_id = store.create_conversation(db_path)
    store.append_message(db_path, conv_id, "first question", {"answer": "a1"})
    store.append_message(db_path, conv_id, "second question", {"answer": "a2"})
    conv = store.get_conversation(db_path, conv_id)
    assert conv["title"] == "first question"


def test_append_message_does_not_overwrite_a_manually_renamed_title(db_path):
    conv_id = store.create_conversation(db_path)
    store.rename_conversation(db_path, conv_id, "My custom title")
    store.append_message(db_path, conv_id, "first question", {"answer": "a1"})
    conv = store.get_conversation(db_path, conv_id)
    assert conv["title"] == "My custom title"


def test_append_message_raises_for_missing_conversation(db_path):
    with pytest.raises(KeyError):
        store.append_message(db_path, 999, "q", {"answer": "a"})


def test_rename_conversation_returns_true_on_success(db_path):
    conv_id = store.create_conversation(db_path)
    assert store.rename_conversation(db_path, conv_id, "New title") is True
    assert store.list_conversations(db_path)[0]["title"] == "New title"


def test_rename_conversation_returns_false_for_missing_id(db_path):
    assert store.rename_conversation(db_path, 999, "x") is False


def test_delete_conversation_returns_true_and_cascades_messages(db_path):
    conv_id = store.create_conversation(db_path)
    store.append_message(db_path, conv_id, "q", {"answer": "a"})
    assert store.delete_conversation(db_path, conv_id) is True
    assert store.list_conversations(db_path) == []
    assert store.get_conversation(db_path, conv_id) is None


def test_delete_conversation_returns_false_for_missing_id(db_path):
    assert store.delete_conversation(db_path, 999) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_store.py -v`
Expected: `ModuleNotFoundError: No module named 'web.store'` (or every test erroring the same way).

- [ ] **Step 3: Write `web/store.py`**

```python
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_TITLE = "Cuộc hội thoại mới"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=5)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                question TEXT NOT NULL,
                answer_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def create_conversation(db_path: str, title: str = DEFAULT_TITLE) -> int:
    now = _now()
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)",
            (title, now, now),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def list_conversations(db_path: str) -> list[dict]:
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, title, updated_at FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
        return [{"id": r[0], "title": r[1], "updated_at": r[2]} for r in rows]
    finally:
        conn.close()


def get_conversation(db_path: str, conversation_id: int) -> dict | None:
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT id, title FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if row is None:
            return None
        messages = conn.execute(
            "SELECT question, answer_json FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        ).fetchall()
        return {
            "id": row[0],
            "title": row[1],
            "messages": [{"question": q, "out": json.loads(a)} for q, a in messages],
        }
    finally:
        conn.close()


def append_message(db_path: str, conversation_id: int, question: str, out: dict) -> None:
    now = _now()
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT title FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"conversation {conversation_id} not found")
        count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (conversation_id,)
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO messages (conversation_id, question, answer_json, created_at) "
            "VALUES (?, ?, ?, ?)",
            (conversation_id, question, json.dumps(out), now),
        )
        if count == 0 and row[0] == DEFAULT_TITLE:
            title = question[:40] + ("…" if len(question) > 40 else "")
            conn.execute(
                "UPDATE conversations SET updated_at = ?, title = ? WHERE id = ?",
                (now, title, conversation_id),
            )
        else:
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
        conn.commit()
    finally:
        conn.close()


def rename_conversation(db_path: str, conversation_id: int, title: str) -> bool:
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id)
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def delete_conversation(db_path: str, conversation_id: int) -> bool:
    conn = _connect(db_path)
    try:
        cur = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_store.py -v`
Expected: all 12 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add web/store.py tests/test_store.py
git commit -m "feat: add SQLite-backed conversation store"
```

---

### Task 2: Conversation API + `POST /api/chat` persistence

**Files:**
- Modify: `app/config.py` (add `conversations_db_path` setting)
- Modify: `.gitignore` (add `data/`)
- Modify: `web/server.py` (new routes, `ChatRequest` contract change, persistence wiring)
- Modify: `tests/test_server.py` (update 3 existing tests for the new `ChatRequest` shape, add conversation CRUD + persistence tests)
- Modify: `tests/test_config.py` (one new test for the default db path)

**Interfaces:**
- Consumes: `web.store` (`DEFAULT_TITLE`, `init_db`, `create_conversation`, `list_conversations`, `get_conversation`, `append_message`, `rename_conversation`, `delete_conversation`) from Task 1.
- Produces (used by Task 3): the five conversation routes and the new `POST /api/chat` contract described below — no Python symbols later tasks import (Task 3 only talks to these routes over HTTP).

- [ ] **Step 1: Add `conversations_db_path` to `Settings`**

In `app/config.py`, the `Settings` class currently ends with (around line 113):

```python
    max_iterations: int = 3
    skills_enabled: bool = False
    search: SearchConfig = SearchConfig()
    fetch: FetchConfig = FetchConfig()
```

Change it to:

```python
    max_iterations: int = 3
    skills_enabled: bool = False
    search: SearchConfig = SearchConfig()
    fetch: FetchConfig = FetchConfig()
    conversations_db_path: str = Field(
        default_factory=lambda: str(REPO_ROOT / "data" / "webscout.db")
    )
```

(`Field` is already imported at the top of the file; `REPO_ROOT` is already defined above the `Settings` class.)

- [ ] **Step 2: Add the failing config test**

In `tests/test_config.py`, add at the end of the file:

```python
def test_conversations_db_path_defaults_under_repo_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CONVERSATIONS_DB_PATH", raising=False)
    from app.config import REPO_ROOT

    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.conversations_db_path == str(REPO_ROOT / "data" / "webscout.db")
```

Run: `uv run pytest tests/test_config.py::test_conversations_db_path_defaults_under_repo_root -v`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'conversations_db_path'` (Step 1 hasn't landed yet if you're doing this test-first — if you already applied Step 1's edit, this will instead PASS immediately, which is fine; either order is acceptable for a one-line settings field).

- [ ] **Step 3: Run it again to confirm it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: all tests PASS, including the new one.

- [ ] **Step 4: Ignore the new data directory**

Add a line to `.gitignore` (anywhere; grouping it near `.env` reads naturally):

```
data/
```

- [ ] **Step 5: Write the failing server tests**

In `tests/test_server.py`, add this import near the top (after the existing `import web.server as server`):

```python
from web import store  # noqa: E402
```

Add this fixture right after the `client = TestClient(server.app)` line:

```python
@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    from app.config import get_settings

    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("CONVERSATIONS_DB_PATH", db_path)
    get_settings.cache_clear()
    yield db_path
    get_settings.cache_clear()
```

Replace the three existing tests that build a `ChatRequest` body — `test_chat_streams_status_then_result`, `test_chat_emits_error_event_on_failure`, and `test_chat_passes_condensed_question_and_history` — with:

```python
def test_chat_streams_status_then_result(isolated_db, monkeypatch):
    conv_id = store.create_conversation(isolated_db)
    monkeypatch.setattr(server, "build_graph", lambda: _LinearFakeGraph())
    monkeypatch.setattr(server, "condense_question", lambda history, question, **k: question)
    resp = client.post("/api/chat", json={"conversation_id": conv_id, "question": "Q?"})
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    kinds = [e for e, _ in events]
    assert kinds == ["status", "status", "status", "result"]
    assert [d["node"] for e, d in events[:3]] == ["research", "verify", "answer"]
    assert events[-1][1]["answer"] == "Final [1]."


def test_chat_emits_error_event_on_failure(isolated_db, monkeypatch):
    conv_id = store.create_conversation(isolated_db)
    monkeypatch.setattr(server, "build_graph", lambda: _RaisingGraph())
    monkeypatch.setattr(server, "condense_question", lambda history, question, **k: question)
    resp = client.post("/api/chat", json={"conversation_id": conv_id, "question": "Q?"})
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    assert events[-1][0] == "error"
    assert "boom" in events[-1][1]["message"]


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
        def stream(self, state, stream_mode="updates"):
            captured_state["state"] = state
            yield from super().stream(state, stream_mode=stream_mode)

    monkeypatch.setattr(server, "build_graph", lambda: _CapturingGraph())
    monkeypatch.setattr(server, "condense_question", fake_condense)
    client.post("/api/chat", json={"conversation_id": conv_id, "question": "and that?"})
    assert seen["history"] == [{"question": "What is LangGraph?", "answer": "A framework."}]
    assert seen["question"] == "and that?"
    assert captured_state["state"]["question"] == "standalone question"


def test_chat_returns_404_for_missing_conversation(isolated_db):
    resp = client.post("/api/chat", json={"conversation_id": 999, "question": "Q?"})
    assert resp.status_code == 404


def test_chat_persists_message_after_result(isolated_db, monkeypatch):
    conv_id = store.create_conversation(isolated_db)
    monkeypatch.setattr(server, "build_graph", lambda: _LinearFakeGraph())
    monkeypatch.setattr(server, "condense_question", lambda history, question, **k: question)
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
```

Delete the old `test_chat_passes_condensed_question_and_history` test entirely — `test_chat_passes_history_from_stored_messages` above replaces it (history now comes from the DB, not a client-supplied `history` field).

Run: `uv run pytest tests/test_server.py -v`
Expected: every test using the old `{"question": ..., "history": [...]}` body shape now FAILs (422 — `conversation_id` is a required field FastAPI hasn't been given), and the brand-new tests FAIL with 404s from routes that don't exist yet.

- [ ] **Step 6: Rewrite `web/server.py`**

Replace the full contents of `web/server.py` with:

```python
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import MODEL_CHOICES, get_settings, override_model
from app.conversation import condense_question
from app.graph import build_graph
from app.main import render_report_markdown, stream_pipeline
from web import store

STATIC_DIR = Path(__file__).parent / "static"


def _db_path() -> str:
    """Read the configured DB path and ensure its schema exists.

    Called per-request (not once at import time) so tests can point
    CONVERSATIONS_DB_PATH at a tmp file via monkeypatch + cache_clear()
    before the first conversation route runs, instead of the real
    data/webscout.db getting created as a side effect of merely
    importing this module.
    """
    path = get_settings().conversations_db_path
    store.init_db(path)
    return path


class ChatRequest(BaseModel):
    conversation_id: int
    question: str
    model: str | None = None
    max_iterations: int | None = Field(default=None, ge=1, le=10)


class ReportRequest(BaseModel):
    question: str
    out: dict


class RenameRequest(BaseModel):
    title: str


app = FastAPI(title="WebScout Chat")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/api/models")
def list_models():
    s = get_settings()
    return {
        "choices": list(MODEL_CHOICES),
        "current": s.researcher.model,
        "key_configured": bool(s.openrouter_api_key),
    }


@app.get("/api/conversations")
def list_conversations():
    return store.list_conversations(_db_path())


@app.post("/api/conversations")
def create_conversation():
    db_path = _db_path()
    conv_id = store.create_conversation(db_path)
    return {"id": conv_id, "title": store.DEFAULT_TITLE}


@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: int):
    conv = store.get_conversation(_db_path(), conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return conv


@app.patch("/api/conversations/{conversation_id}")
def rename_conversation(conversation_id: int, body: RenameRequest):
    title = body.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="title cannot be empty")
    ok = store.rename_conversation(_db_path(), conversation_id, title)
    if not ok:
        raise HTTPException(status_code=404, detail="conversation not found")
    return {"id": conversation_id, "title": title}


@app.delete("/api/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: int):
    ok = store.delete_conversation(_db_path(), conversation_id)
    if not ok:
        raise HTTPException(status_code=404, detail="conversation not found")
    return Response(status_code=204)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


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
            history = [
                {"question": m["question"], "answer": m["out"].get("answer", "")}
                for m in conversation["messages"]
            ]
            question = condense_question(history, body.question)
            graph = build_graph()
            for kind, payload in stream_pipeline(
                question, graph=graph, max_iterations=body.max_iterations
            ):
                if kind == "status":
                    yield _sse("status", {"node": payload})
                else:
                    store.append_message(db_path, body.conversation_id, body.question, payload)
                    yield _sse("result", payload)
        except Exception as exc:
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/report")
def report(body: ReportRequest):
    md = render_report_markdown(body.question, body.out)
    return Response(
        md,
        media_type="text/markdown",
        headers={"Content-Disposition": 'attachment; filename="report.md"'},
    )


def main() -> None:
    """Entry point for the `webscout-web` console script."""
    import uvicorn

    uvicorn.run("web.server:app", host="127.0.0.1", port=8000, reload=False)
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_server.py tests/test_config.py tests/test_store.py -v`
Expected: all PASS.

- [ ] **Step 8: Run the full offline suite**

Run: `uv run pytest -m "not integration" -v`
Expected: all PASS (no regressions elsewhere).

- [ ] **Step 9: Commit**

```bash
git add app/config.py .gitignore web/server.py tests/test_server.py tests/test_config.py
git commit -m "feat: persist conversations server-side, add conversation CRUD API"
```

---

### Task 3: Sidebar conversation list (new / select / rename / delete / replay)

**Files:**
- Modify: `web/static/index.html`
- Modify: `web/static/style.css`
- Modify: `web/static/app.js`

**Interfaces:**
- Consumes: the five conversation routes + the new `POST /api/chat` contract from Task 2 (`GET /api/conversations`, `POST /api/conversations`, `GET /api/conversations/{id}`, `PATCH /api/conversations/{id}`, `DELETE /api/conversations/{id}`).
- Produces (used by Task 4 and Task 5): DOM ids `#sidebar-header`, `#new-conversation`, `#conversation-list`, `#settings`; JS globals `activeConversationId`, `conversations`, and the functions `loadConversations()`, `selectConversation(id)`, `createConversation()`, `renderConversationList()`; the `sendQuestion` function shape Task 5 will edit further (a `thinking`/`trace`/`status` triple is introduced in Task 5, not here — this task still uses the single `addBubble(...)` placeholder bubble).

- [ ] **Step 1: Replace `web/static/index.html`**

```html
<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <title>WebScout Chat</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link
    href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap"
    rel="stylesheet"
  />
  <link rel="stylesheet" href="/static/style.css" />
</head>
<body>
  <div id="app">
    <aside id="sidebar">
      <div id="sidebar-header">
        <div id="wordmark">
          <span class="eyebrow">Case File</span>
          <h1>WebScout</h1>
        </div>
      </div>
      <button id="new-conversation" type="button">+ Cuộc hội thoại mới</button>
      <div id="conversation-list"></div>
      <div id="settings">
        <div class="field">
          <label for="model-select">Model</label>
          <select id="model-select"></select>
        </div>
        <div class="field">
          <label for="max-iterations">Max iterations</label>
          <input id="max-iterations" type="number" min="1" max="10" value="3" />
        </div>
      </div>
      <div id="key-banner" class="banner hidden">
        OPENROUTER_API_KEY chưa được cấu hình. Xem README để thiết lập .env.
      </div>
    </aside>
    <main id="chat">
      <div id="log"></div>
      <form id="composer">
        <textarea
          id="question"
          rows="2"
          placeholder="Hỏi gì đó... (Enter để gửi, Shift+Enter xuống dòng)"
        ></textarea>
        <button id="send" type="submit">Gửi</button>
      </form>
    </main>
  </div>
  <script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Add sidebar-list CSS to `web/static/style.css`**

Insert this block right after the existing `.field label { ... }` rule (after line 77 in the current file) and before the `#sidebar select,` rule:

```css
#sidebar-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 8px;
}

#new-conversation {
  padding: 8px 10px;
  background: var(--accent-soft);
  color: var(--accent);
  border: 1px solid var(--accent);
  border-radius: 4px;
  font-family: var(--font-body);
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  text-align: left;
}

#new-conversation:hover {
  background: var(--accent);
  color: var(--bg);
}

#conversation-list {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.conversation-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 7px 8px;
  border-radius: 4px;
  border-left: 3px solid transparent;
  cursor: pointer;
}

.conversation-item:hover {
  background: var(--panel-2);
}

.conversation-item.active {
  border-left-color: var(--accent);
  background: var(--panel-2);
}

.conversation-title {
  flex: 1;
  min-width: 0;
  font-size: 13px;
  color: var(--text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.conversation-actions {
  display: flex;
  gap: 2px;
  flex-shrink: 0;
}

.conversation-actions button {
  width: 20px;
  height: 20px;
  padding: 0;
  background: transparent;
  border: none;
  color: var(--text-dim);
  cursor: pointer;
  font-size: 12px;
  border-radius: 3px;
}

.conversation-actions button:hover {
  color: var(--text);
  background: var(--border);
}

#settings {
  display: flex;
  flex-direction: column;
  gap: 10px;
  flex-shrink: 0;
}
```

Then, inside the existing `@media (max-width: 720px)` block at the bottom of the file, add one more rule (anywhere inside the block):

```css
  #conversation-list {
    width: 100%;
    max-height: 160px;
  }
```

- [ ] **Step 3: Rewrite `web/static/app.js`**

Replace the full contents of `web/static/app.js` with:

```js
const logEl = document.getElementById("log");
const formEl = document.getElementById("composer");
const questionEl = document.getElementById("question");
const sendEl = document.getElementById("send");
const modelSelectEl = document.getElementById("model-select");
const maxIterEl = document.getElementById("max-iterations");
const bannerEl = document.getElementById("key-banner");
const newConversationEl = document.getElementById("new-conversation");
const conversationListEl = document.getElementById("conversation-list");

let currentModel = null;
let turnCounter = 0;
let activeConversationId = null;
let conversations = []; // [{id, title, updated_at}]

const STATUS_LABELS = {
  research: "Đang research...",
  verify: "Đang verify...",
  answer: "Đang trả lời...",
};

function safeHref(url) {
  try {
    const parsed = new URL(url, window.location.href);
    return parsed.protocol === "http:" || parsed.protocol === "https:" ? parsed.href : "#";
  } catch {
    return "#";
  }
}

function addBubble(role, text, extraClass) {
  const div = document.createElement("div");
  div.className = `bubble ${role}${extraClass ? " " + extraClass : ""}`;
  div.textContent = text;
  logEl.appendChild(div);
  logEl.scrollTop = logEl.scrollHeight;
  return div;
}

function renderResult(bubble, question, out) {
  bubble.textContent = "";
  bubble.classList.remove("error", "pending");

  const turnId = `t${turnCounter++}`;
  const sources = out.sources || [];
  const sourceIndexByUrl = new Map(sources.map((s, i) => [s.url, i]));

  const eyebrow = document.createElement("div");
  eyebrow.className = "eyebrow";
  eyebrow.textContent = "Finding Report";
  bubble.appendChild(eyebrow);

  const answerText = document.createElement("div");
  answerText.className = "answer-text";
  answerText.textContent = out.answer || "";
  bubble.appendChild(answerText);

  if (out.findings && out.findings.length) {
    const findings = document.createElement("div");
    findings.className = "findings";
    const heading = document.createElement("div");
    heading.className = "ledger-heading";
    heading.textContent = "Findings";
    findings.appendChild(heading);
    out.findings.forEach((f) => {
      const conf = (f.confidence || "unknown").toLowerCase();
      const row = document.createElement("div");
      row.className = `finding-row confidence-${conf}`;
      const dot = document.createElement("span");
      dot.className = "confidence-dot";
      row.appendChild(dot);
      const claim = document.createElement("span");
      claim.className = "finding-claim";
      claim.textContent = f.claim || "";
      row.appendChild(claim);
      (f.source_urls || []).forEach((url) => {
        const idx = sourceIndexByUrl.get(url);
        if (idx === undefined) return;
        const tab = document.createElement("a");
        tab.className = "citation-tab";
        tab.href = `#${turnId}-source-${idx}`;
        tab.textContent = `S${idx + 1}`;
        tab.addEventListener("click", (e) => {
          e.preventDefault();
          const target = document.getElementById(`${turnId}-source-${idx}`);
          if (!target) return;
          target.scrollIntoView({ behavior: "smooth", block: "center" });
          target.classList.add("flash");
          setTimeout(() => target.classList.remove("flash"), 900);
        });
        row.appendChild(tab);
      });
      findings.appendChild(row);
    });
    bubble.appendChild(findings);
  }

  if (sources.length) {
    const sourcesEl = document.createElement("div");
    sourcesEl.className = "sources";
    const heading = document.createElement("div");
    heading.className = "ledger-heading";
    heading.textContent = "Sources";
    sourcesEl.appendChild(heading);
    sources.forEach((s, i) => {
      const item = document.createElement("div");
      item.className = "source-item";
      item.id = `${turnId}-source-${i}`;
      const num = document.createElement("span");
      num.className = "source-num";
      num.textContent = `S${i + 1}`;
      item.appendChild(num);
      const link = document.createElement("a");
      link.href = safeHref(s.url);
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = s.title || s.url;
      item.appendChild(link);
      sourcesEl.appendChild(item);
    });
    bubble.appendChild(sourcesEl);
  }

  const metrics = document.createElement("div");
  metrics.className = "metrics";
  metrics.textContent =
    `iterations: ${out.iteration ?? 0} | searches: ${out.search_calls ?? 0} | ` +
    `sources: ${(out.sources || []).length} | tokens: ${out.total_tokens ?? 0} | ` +
    `est_cost: $${(out.total_cost ?? 0).toFixed(4)}`;
  bubble.appendChild(metrics);

  const download = document.createElement("button");
  download.type = "button";
  download.className = "download";
  download.textContent = "Tải report.md";
  download.addEventListener("click", () => downloadReport(question, out));
  bubble.appendChild(download);
}

async function downloadReport(question, out) {
  const resp = await fetch("/api/report", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, out }),
  });
  const blob = await resp.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "report.md";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function parseSseFrame(frame) {
  const lines = frame.split("\n");
  let event = "message";
  let data = "";
  for (const line of lines) {
    if (line.startsWith("event: ")) event = line.slice(7);
    if (line.startsWith("data: ")) data = line.slice(6);
  }
  return data ? { event, data: JSON.parse(data) } : null;
}

function renderConversationList() {
  conversationListEl.replaceChildren();
  conversations.forEach((c) => {
    const item = document.createElement("div");
    item.className = `conversation-item${c.id === activeConversationId ? " active" : ""}`;

    const title = document.createElement("span");
    title.className = "conversation-title";
    title.textContent = c.title;
    title.addEventListener("click", () => selectConversation(c.id));
    item.appendChild(title);

    const actions = document.createElement("span");
    actions.className = "conversation-actions";

    const renameBtn = document.createElement("button");
    renameBtn.type = "button";
    renameBtn.textContent = "✎";
    renameBtn.title = "Đổi tên";
    renameBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      renameConversation(c.id, c.title);
    });
    actions.appendChild(renameBtn);

    const deleteBtn = document.createElement("button");
    deleteBtn.type = "button";
    deleteBtn.textContent = "×";
    deleteBtn.title = "Xóa";
    deleteBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      deleteConversation(c.id);
    });
    actions.appendChild(deleteBtn);

    item.appendChild(actions);
    conversationListEl.appendChild(item);
  });
}

async function loadConversations() {
  const resp = await fetch("/api/conversations");
  conversations = await resp.json();
  renderConversationList();
}

async function createConversation() {
  const resp = await fetch("/api/conversations", { method: "POST" });
  const conv = await resp.json();
  conversations.unshift({ id: conv.id, title: conv.title, updated_at: null });
  await selectConversation(conv.id);
}

async function selectConversation(id) {
  activeConversationId = id;
  renderConversationList();
  const resp = await fetch(`/api/conversations/${id}`);
  const data = await resp.json();
  logEl.replaceChildren();
  data.messages.forEach((m) => {
    addBubble("user", m.question);
    const bubble = document.createElement("div");
    bubble.className = "bubble assistant";
    logEl.appendChild(bubble);
    renderResult(bubble, m.question, m.out);
  });
  logEl.scrollTop = logEl.scrollHeight;
}

async function renameConversation(id, currentTitle) {
  const next = window.prompt("Đổi tên hội thoại:", currentTitle);
  if (next === null) return;
  const title = next.trim();
  if (!title) return;
  const resp = await fetch(`/api/conversations/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  if (!resp.ok) return;
  const updated = await resp.json();
  const conv = conversations.find((c) => c.id === id);
  if (conv) conv.title = updated.title;
  renderConversationList();
}

async function deleteConversation(id) {
  if (!window.confirm("Xóa hội thoại này?")) return;
  const resp = await fetch(`/api/conversations/${id}`, { method: "DELETE" });
  if (!resp.ok) return;
  conversations = conversations.filter((c) => c.id !== id);
  if (activeConversationId === id) {
    if (conversations.length) {
      await selectConversation(conversations[0].id);
    } else {
      await createConversation();
    }
  } else {
    renderConversationList();
  }
}

async function sendQuestion(question) {
  addBubble("user", question);
  const thinking = addBubble("assistant", STATUS_LABELS.research, "pending");

  const body = {
    conversation_id: activeConversationId,
    question,
    model: modelSelectEl.value !== currentModel ? modelSelectEl.value : null,
    max_iterations: Number(maxIterEl.value) || null,
  };

  questionEl.disabled = true;
  sendEl.disabled = true;

  try {
    let resp;
    try {
      resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    } catch (err) {
      thinking.textContent = `Lỗi kết nối: ${err.message}`;
      thinking.classList.remove("pending");
      thinking.classList.add("error");
      return;
    }

    if (!resp.ok) {
      let detail = "";
      try {
        detail = await resp.text();
      } catch {
        // best-effort only; fall back to the status line below
      }
      thinking.textContent = `Lỗi: ${resp.status} ${resp.statusText}${detail ? " — " + detail : ""}`;
      thinking.classList.remove("pending");
      thinking.classList.add("error");
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let sawTerminalEvent = false;

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const frames = buffer.split("\n\n");
        buffer = frames.pop();
        for (const frame of frames) {
          const parsed = parseSseFrame(frame);
          if (!parsed) continue;
          const { event, data } = parsed;
          if (event === "status") {
            thinking.textContent = STATUS_LABELS[data.node] || `Đang ${data.node}...`;
          } else if (event === "result") {
            renderResult(thinking, question, data);
            sawTerminalEvent = true;
          } else if (event === "error") {
            thinking.textContent = `Lỗi: ${data.message}`;
            thinking.classList.remove("pending");
            thinking.classList.add("error");
            sawTerminalEvent = true;
          }
        }
      }
    } catch (err) {
      thinking.textContent = `Lỗi: ${err.message}`;
      thinking.classList.remove("pending");
      thinking.classList.add("error");
      return;
    }

    if (!sawTerminalEvent) {
      thinking.textContent = "Lỗi: kết nối bị ngắt trước khi có kết quả.";
      thinking.classList.remove("pending");
      thinking.classList.add("error");
    } else {
      await loadConversations();
    }
  } finally {
    questionEl.disabled = false;
    sendEl.disabled = false;
    questionEl.focus();
  }
}

formEl.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = questionEl.value.trim();
  if (!question || !activeConversationId) return;
  questionEl.value = "";
  sendQuestion(question);
});

questionEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    formEl.requestSubmit();
  }
});

newConversationEl.addEventListener("click", () => {
  createConversation();
});

async function loadModels() {
  const resp = await fetch("/api/models");
  const data = await resp.json();
  currentModel = data.current;
  modelSelectEl.innerHTML = "";
  for (const choice of data.choices) {
    const opt = document.createElement("option");
    opt.value = choice;
    opt.textContent = choice;
    if (choice === data.current) opt.selected = true;
    modelSelectEl.appendChild(opt);
  }
  if (!data.key_configured) {
    bannerEl.classList.remove("hidden");
    questionEl.disabled = true;
    sendEl.disabled = true;
  }
}

async function init() {
  await loadModels();
  try {
    await loadConversations();
    if (conversations.length) {
      await selectConversation(conversations[0].id);
    } else {
      await createConversation();
    }
  } catch (err) {
    bannerEl.textContent = "Không kết nối được máy chủ. Thử tải lại trang.";
    bannerEl.classList.remove("hidden");
    questionEl.disabled = true;
    sendEl.disabled = true;
  }
}

init();
```

- [ ] **Step 4: Run the offline suite (regression check)**

Run: `uv run pytest -m "not integration" -v`
Expected: all PASS (this task touches no Python, but confirms nothing else broke).

- [ ] **Step 5: Manual verification**

No JS test tooling exists in this project (see `CLAUDE.md`'s Testing conventions) — verify by running the app:

```bash
uv run uvicorn web.server:app --port 8000
```

Open `http://127.0.0.1:8000/` and check:
- A conversation is auto-created on first load (empty DB) and appears selected in the sidebar.
- Sending a question shows the reply, then the sidebar item's title updates to the question text (first message auto-titles).
- "+ Cuộc hội thoại mới" creates and switches to a new, empty conversation.
- Clicking an older conversation in the list replays its stored messages (including their sources/findings/citation tabs) into the log.
- ✎ renames via a prompt dialog; the sidebar list updates immediately.
- × deletes after a confirm dialog; deleting the active conversation selects the next one (or creates a fresh one if none remain).

Stop the server (`Ctrl+C` in that shell, or the harness's task-stop) when done.

- [ ] **Step 6: Commit**

```bash
git add web/static/index.html web/static/style.css web/static/app.js
git commit -m "feat: add sidebar conversation list (new/select/rename/delete)"
```

---

### Task 4: Light/dark theme toggle

**Files:**
- Modify: `web/static/index.html`
- Modify: `web/static/style.css`
- Modify: `web/static/app.js`

**Interfaces:**
- Consumes: `#sidebar-header` (from Task 3) as the mount point for the toggle button.
- Produces: nothing later tasks depend on (Task 5 doesn't touch theming).

- [ ] **Step 1: Add the toggle button to `web/static/index.html`**

Find:

```html
      <div id="sidebar-header">
        <div id="wordmark">
          <span class="eyebrow">Case File</span>
          <h1>WebScout</h1>
        </div>
      </div>
```

Replace with:

```html
      <div id="sidebar-header">
        <div id="wordmark">
          <span class="eyebrow">Case File</span>
          <h1>WebScout</h1>
        </div>
        <button id="theme-toggle" type="button" aria-label="Đổi giao diện sáng/tối">☾</button>
      </div>
```

- [ ] **Step 2: Add the light palette + toggle button styles to `web/static/style.css`**

Insert this block immediately after the `:root { ... }` block at the top of the file (after its closing `}`, before the `* { box-sizing: border-box; }` rule):

```css
:root[data-theme="light"] {
  --bg: #f3efe6;
  --panel: #ffffff;
  --panel-2: #faf7f0;
  --border: #ddd5c4;
  --text: #1c1a16;
  --text-dim: #6b6559;
  --accent: #9c5f1f;
  --accent-soft: rgba(156, 95, 31, 0.12);
  --conf-high: #3f8f6c;
  --conf-medium: #a67a1f;
  --conf-low: #a83f38;
}
```

Then add the toggle button style anywhere after the `#sidebar-header { ... }` rule added in Task 3:

```css
#theme-toggle {
  flex-shrink: 0;
  width: 30px;
  height: 30px;
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 4px;
  color: var(--text);
  cursor: pointer;
  font-size: 14px;
}

#theme-toggle:hover {
  border-color: var(--accent);
}
```

- [ ] **Step 3: Add theme logic to `web/static/app.js`**

Find this line near the top of the file:

```js
const conversationListEl = document.getElementById("conversation-list");
```

Add right after it:

```js
const themeToggleEl = document.getElementById("theme-toggle");
```

Add these two functions anywhere above the `init()` function (e.g. right after `safeHref`):

```js
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  themeToggleEl.textContent = theme === "light" ? "☀" : "☾";
}

function initTheme() {
  const stored = localStorage.getItem("webscout-theme");
  if (stored === "light" || stored === "dark") {
    applyTheme(stored);
    return;
  }
  const prefersLight = window.matchMedia("(prefers-color-scheme: light)").matches;
  applyTheme(prefersLight ? "light" : "dark");
}

themeToggleEl.addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme") || "dark";
  const next = current === "light" ? "dark" : "light";
  localStorage.setItem("webscout-theme", next);
  applyTheme(next);
});
```

Find the bottom of the file:

```js
init();
```

Replace with:

```js
initTheme();
init();
```

- [ ] **Step 4: Run the offline suite (regression check)**

Run: `uv run pytest -m "not integration" -v`
Expected: all PASS.

- [ ] **Step 5: Manual verification**

Run the app (`uv run uvicorn web.server:app --port 8000`), open `http://127.0.0.1:8000/`, and check:
- The sun/moon button toggles the whole page between the dark "Case File" look and a light paper look — same fonts, same layout, same amber accent family, only the palette changes.
- Reloading the page keeps the chosen theme (`localStorage["webscout-theme"]`).
- With `localStorage` cleared, the page follows the OS/browser light-dark preference.

Stop the server when done.

- [ ] **Step 6: Commit**

```bash
git add web/static/index.html web/static/style.css web/static/app.js
git commit -m "feat: add light/dark theme toggle"
```

---

### Task 5: Status trace chips + CLAUDE.md update

**Files:**
- Modify: `web/static/style.css`
- Modify: `web/static/app.js`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: the `sendQuestion` and `renderResult` functions as they stand after Task 3 (this task edits both).
- Produces: nothing later — this is the plan's last task.

- [ ] **Step 1: Add trace/chip styles to `web/static/style.css`**

Insert this block anywhere after the `.bubble.assistant.pending { ... }` rule (reuses the existing `@keyframes pulse` defined a few lines below it — no new keyframes needed):

```css
.trace {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  margin-bottom: 10px;
}

.chip {
  position: relative;
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--text-dim);
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 3px;
  padding: 2px 6px;
}

.chip:not(:last-child)::after {
  content: "→";
  position: absolute;
  top: 50%;
  right: -14px;
  transform: translateY(-50%);
  color: var(--text-dim);
  font-size: 10px;
}

.chip.active {
  color: var(--accent);
  border-color: var(--accent);
  background: var(--accent-soft);
}

@media (prefers-reduced-motion: no-preference) {
  .chip.active {
    animation: pulse 1.6s ease-in-out infinite;
  }
}
```

- [ ] **Step 2: Rewrite the pending-bubble and status-handling logic in `web/static/app.js`**

Find:

```js
function renderResult(bubble, question, out) {
  bubble.textContent = "";
  bubble.classList.remove("error", "pending");

  const turnId = `t${turnCounter++}`;
```

Replace with:

```js
function renderResult(bubble, question, out) {
  const existingTrace = bubble.querySelector(".trace");
  bubble.textContent = "";
  bubble.classList.remove("error", "pending");
  if (existingTrace) bubble.appendChild(existingTrace);

  const turnId = `t${turnCounter++}`;
```

Find:

```js
async function sendQuestion(question) {
  addBubble("user", question);
  const thinking = addBubble("assistant", STATUS_LABELS.research, "pending");
```

Replace with:

```js
function startAssistantBubble() {
  const bubble = document.createElement("div");
  bubble.className = "bubble assistant pending";
  const trace = document.createElement("div");
  trace.className = "trace";
  bubble.appendChild(trace);
  const status = document.createElement("div");
  status.className = "status-text";
  status.textContent = STATUS_LABELS.research;
  bubble.appendChild(status);
  logEl.appendChild(bubble);
  logEl.scrollTop = logEl.scrollHeight;
  return { bubble, trace, status };
}

function addTraceChip(trace, node) {
  const prevActive = trace.querySelector(".chip.active");
  if (prevActive) prevActive.classList.remove("active");
  const chip = document.createElement("span");
  chip.className = "chip active";
  chip.textContent = node;
  trace.appendChild(chip);
}

function settleTrace(trace) {
  const active = trace.querySelector(".chip.active");
  if (active) active.classList.remove("active");
}

async function sendQuestion(question) {
  addBubble("user", question);
  const { bubble: thinking, trace, status } = startAssistantBubble();
```

Find the whole SSE-handling block:

```js
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const frames = buffer.split("\n\n");
        buffer = frames.pop();
        for (const frame of frames) {
          const parsed = parseSseFrame(frame);
          if (!parsed) continue;
          const { event, data } = parsed;
          if (event === "status") {
            thinking.textContent = STATUS_LABELS[data.node] || `Đang ${data.node}...`;
          } else if (event === "result") {
            renderResult(thinking, question, data);
            sawTerminalEvent = true;
          } else if (event === "error") {
            thinking.textContent = `Lỗi: ${data.message}`;
            thinking.classList.remove("pending");
            thinking.classList.add("error");
            sawTerminalEvent = true;
          }
        }
      }
    } catch (err) {
      thinking.textContent = `Lỗi: ${err.message}`;
      thinking.classList.remove("pending");
      thinking.classList.add("error");
      return;
    }

    if (!sawTerminalEvent) {
      thinking.textContent = "Lỗi: kết nối bị ngắt trước khi có kết quả.";
      thinking.classList.remove("pending");
      thinking.classList.add("error");
    } else {
      await loadConversations();
    }
```

Replace with:

```js
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const frames = buffer.split("\n\n");
        buffer = frames.pop();
        for (const frame of frames) {
          const parsed = parseSseFrame(frame);
          if (!parsed) continue;
          const { event, data } = parsed;
          if (event === "status") {
            addTraceChip(trace, data.node);
            status.textContent = STATUS_LABELS[data.node] || `Đang ${data.node}...`;
          } else if (event === "result") {
            settleTrace(trace);
            renderResult(thinking, question, data);
            sawTerminalEvent = true;
          } else if (event === "error") {
            settleTrace(trace);
            status.textContent = `Lỗi: ${data.message}`;
            thinking.classList.remove("pending");
            thinking.classList.add("error");
            sawTerminalEvent = true;
          }
        }
      }
    } catch (err) {
      settleTrace(trace);
      status.textContent = `Lỗi: ${err.message}`;
      thinking.classList.remove("pending");
      thinking.classList.add("error");
      return;
    }

    if (!sawTerminalEvent) {
      settleTrace(trace);
      status.textContent = "Lỗi: kết nối bị ngắt trước khi có kết quả.";
      thinking.classList.remove("pending");
      thinking.classList.add("error");
    } else {
      await loadConversations();
    }
```

Finally, the two earlier error branches (connection failure and non-ok response) still reference `thinking.textContent` directly — find:

```js
    } catch (err) {
      thinking.textContent = `Lỗi kết nối: ${err.message}`;
      thinking.classList.remove("pending");
      thinking.classList.add("error");
      return;
    }

    if (!resp.ok) {
      let detail = "";
      try {
        detail = await resp.text();
      } catch {
        // best-effort only; fall back to the status line below
      }
      thinking.textContent = `Lỗi: ${resp.status} ${resp.statusText}${detail ? " — " + detail : ""}`;
      thinking.classList.remove("pending");
      thinking.classList.add("error");
      return;
    }
```

Replace with:

```js
    } catch (err) {
      status.textContent = `Lỗi kết nối: ${err.message}`;
      thinking.classList.remove("pending");
      thinking.classList.add("error");
      return;
    }

    if (!resp.ok) {
      let detail = "";
      try {
        detail = await resp.text();
      } catch {
        // best-effort only; fall back to the status line below
      }
      status.textContent = `Lỗi: ${resp.status} ${resp.statusText}${detail ? " — " + detail : ""}`;
      thinking.classList.remove("pending");
      thinking.classList.add("error");
      return;
    }
```

- [ ] **Step 3: Run the offline suite (regression check)**

Run: `uv run pytest -m "not integration" -v`
Expected: all PASS.

- [ ] **Step 4: Manual verification**

Run the app (`uv run uvicorn web.server:app --port 8000`), open `http://127.0.0.1:8000/`, and check:
- Sending a question shows chips accumulating left to right (`research`, then `verify`, then possibly `research`/`verify` again, then `answer`), with the current one visibly pulsing (amber) and settled ones dim.
- After the answer renders, the chip trail stays visible above the "Finding Report" content — it isn't wiped.
- Deliberately trigger an error (e.g. stop the server mid-request) and confirm the chip trail up to that point stays visible next to the error text.

Stop the server when done.

- [ ] **Step 5: Update `CLAUDE.md`**

In the "### Web chat UI" section (currently a single paragraph), add a second paragraph right after the existing one:

```markdown
Conversations persist server-side in `data/webscout.db` via `web/store.py` (stdlib `sqlite3`, no new dependency) — `POST /api/chat` now takes a `conversation_id` instead of a client-supplied `history` array, loading prior turns from the DB and appending the new one after the pipeline finishes. The chat UI's light/dark theme is a single CSS token system (`:root` = dark, `:root[data-theme="light"]` overrides) toggled client-side via `localStorage["webscout-theme"]`; per-turn node status accumulates as visible trace chips instead of being overwritten.
```

- [ ] **Step 6: Commit**

```bash
git add web/static/style.css web/static/app.js CLAUDE.md
git commit -m "feat: add persistent status trace chips, document conversations architecture"
```
