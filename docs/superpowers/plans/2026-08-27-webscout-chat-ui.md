# WebScout Chat UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a browser chat UI (FastAPI + Server-Sent Events + vanilla JS) over the existing LangGraph pipeline, with conversation-aware follow-up questions, live per-node status, model/max_iterations picker, and markdown report download.

**Architecture:** `app/main.py`'s graph-driving loop becomes a generator (`stream_pipeline`) that both the existing CLI and a new FastAPI app (`web/server.py`) consume — one implementation of "drive the graph, report progress" instead of two. A new `app/conversation.py::condense_question` rewrites follow-up questions into standalone ones using client-supplied history, before the question ever reaches the graph. The browser is a single static page (`web/static/`) that POSTs to `/api/chat` and manually parses the SSE response body (POST bodies don't work with the browser's native `EventSource`).

**Tech Stack:** FastAPI, Starlette's `StreamingResponse` + `StaticFiles`, uvicorn, vanilla JS (no build step, no frontend framework) — all behind a new `web` dependency group so the default `uv sync` / CI stays unaffected.

**Spec:** [docs/superpowers/specs/2026-08-27-webscout-chat-ui-design.md](../specs/2026-08-27-webscout-chat-ui-design.md)

## Global Constraints

- **Fresh graph per request.** `/api/chat` calls `build_graph()` inside the request handler, never a cached/module-level graph — `build_graph()` already builds a fresh `UsageCollector` per call, and a shared one leaks usage between concurrent requests (same reason `evals/run_evals.py::_graph()` is explicitly not `@lru_cache`d).
- **`override_model` is process-global**, exactly like the CLI's `--model` already is. Two tabs picking different models concurrently will race — accepted, not fixed here.
- **No new config fields.** `condense_question` reuses the existing `verifier` role. `max_iterations` per-request override is a plain function argument, never touches `config.yaml`/`Settings`.
- **`run_pipeline`'s CLI-visible behavior does not change.** Printed lines and the returned dict shape stay byte-for-byte identical — every existing assertion in `tests/test_cli.py` on them must keep passing unmodified.
- **The `web` dependency group must not affect the default `uv sync`.** Any test importing `fastapi` starts with `pytest.importorskip("fastapi")` so CI (no `--group web`) stays green.
- **The offline suite (`-m "not integration"`) stays network-free.** `fastapi.testclient.TestClient` makes in-process ASGI calls, not real sockets. Tests that would otherwise reach the graph/model monkeypatch `build_graph`/`condense_question`/the injectable `model=` seam — never a real network call.
- **No markdown rendering, no `localStorage`, no debug/internals view, no multi-user auth** — all explicitly out of scope per the spec's non-goals. Do not add them.

---

### Task 1: `app/main.py` — `stream_pipeline` generator + `render_report_markdown` split

**Files:**
- Modify: `app/main.py:54-129` (the `run_pipeline` and `write_report` functions)
- Test: `tests/test_cli.py` (extend)

**Interfaces:**
- Consumes: `build_graph()` from `app/graph.py` (unchanged), `get_settings()` from `app/config.py` (unchanged) — both already imported in `app/main.py`.
- Produces:
  - `stream_pipeline(question: str, graph=None, max_iterations: int | None = None)` — a generator yielding `("status", node_name)` once per completed LangGraph node, in order, then exactly one `("result", out_dict)` as the last item. `out_dict` keys: `answer, sources, findings, search_calls, sufficient, iteration, total_tokens, total_cost` (same shape `run_pipeline` has always returned).
  - `render_report_markdown(question: str, out: dict) -> str` — the markdown report body as a string.
  - `run_pipeline(question: str, graph=None, max_iterations: int | None = None) -> dict` — same external behavior as before, plus the new optional `max_iterations` kwarg.
  - `write_report(question: str, out: dict, path: str) -> None` — now a two-line wrapper around `render_report_markdown`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cli.py` (after `test_run_pipeline_prints_progress_and_answer`, which already defines `FakeGraph`/`DELTAS` above it — reuse both):

```python
def test_stream_pipeline_yields_status_then_result():
    from app.main import stream_pipeline

    events = list(stream_pipeline("Q?", graph=FakeGraph(DELTAS)))
    kinds = [kind for kind, _ in events]
    assert kinds == ["status", "status", "status", "result"]
    assert [payload for kind, payload in events[:3]] == ["research", "verify", "answer"]
    result = events[-1][1]
    assert result["answer"] == "Final [1]."
    assert result["search_calls"] == 3
    assert result["sufficient"] is True


def test_render_report_markdown_matches_written_file(tmp_path, monkeypatch):
    from datetime import datetime as real_datetime

    import app.main as m

    class _FrozenDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return real_datetime(2026, 8, 27, 12, 0, 0)

    monkeypatch.setattr(m, "datetime", _FrozenDatetime)

    out = {
        "answer": "Body text [1].",
        "sources": [{"url": "https://a.dev", "title": "A"}],
        "findings": [{"claim": "c", "confidence": "high", "source_urls": ["https://a.dev"]}],
        "sufficient": True,
        "iteration": 2,
        "search_calls": 3,
        "total_tokens": 500,
        "total_cost": 0.01,
    }
    path = tmp_path / "report.md"
    m.write_report("Q?", out, str(path))
    assert m.render_report_markdown("Q?", out) == path.read_text(encoding="utf-8")


def test_run_pipeline_max_iterations_override():
    from app.main import run_pipeline

    class _CapturingGraph:
        def __init__(self, inner):
            self._inner = inner
            self.seen_state = None

        def stream(self, state, stream_mode="updates"):
            self.seen_state = state
            yield from self._inner.stream(state, stream_mode=stream_mode)

    capturing = _CapturingGraph(FakeGraph(DELTAS))
    run_pipeline("Q?", graph=capturing, max_iterations=1)
    assert capturing.seen_state["max_iterations"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -k "stream_pipeline or render_report_markdown or max_iterations_override" -v`
Expected: FAIL — `stream_pipeline`/`render_report_markdown` don't exist yet, and `run_pipeline` doesn't accept `max_iterations`.

- [ ] **Step 3: Implement `stream_pipeline`, rewrite `run_pipeline`, split `render_report_markdown`**

Replace `app/main.py:54-129` (from `def run_pipeline(question: str, graph=None) -> dict:` through the end of `write_report`) with:

```python
def stream_pipeline(question: str, graph=None, max_iterations: int | None = None):
    """Drive the LangGraph pipeline, yielding progress then the final result.

    Yields ("status", node_name) once per completed node, in order, then yields
    exactly one ("result", out_dict) as the last item.
    """
    if graph is None:
        require_openrouter_key()
    g = graph or build_graph()
    s = get_settings()
    mi = max_iterations if max_iterations is not None else s.max_iterations
    state = {"question": question, "iteration": 0, "max_iterations": mi}
    final = dict(state)
    for mode, chunk in g.stream(state, stream_mode=["updates", "values"]):
        if mode == "updates":
            for node in chunk:
                yield ("status", node)
        else:
            final = chunk
    yield (
        "result",
        {
            "answer": final.get("answer", ""),
            "sources": final.get("sources", []),
            "findings": final.get("findings", []),
            "search_calls": final.get("search_calls", 0),
            "sufficient": final.get("sufficient", False),
            "iteration": final.get("iteration", 0),
            "total_tokens": final.get("total_tokens", 0),
            "total_cost": round(final.get("total_cost", 0.0), 4),
        },
    )


def run_pipeline(question: str, graph=None, max_iterations: int | None = None) -> dict:
    out = None
    for kind, payload in stream_pipeline(question, graph=graph, max_iterations=max_iterations):
        if kind == "status":
            print(f"[{payload}] ...", flush=True)
        else:
            out = payload
    return out


def _print_result(out: dict) -> None:
    print("\n=== ANSWER ===\n")
    print(out["answer"])
    if out["sources"]:
        print("\n=== SOURCES ===")
        for i, src in enumerate(out["sources"], 1):
            print(f"[{i}] {src['title']} — {src['url']}")
    print(
        "\n=== METRICS ===\n"
        f"iterations: {out.get('iteration', 0)} | "
        f"searches: {out.get('search_calls', 0)} | "
        f"sources: {len(out.get('sources', []))} | "
        f"tokens: {out.get('total_tokens', 0)} | "
        f"est_cost: ${out.get('total_cost', 0.0):.4f}"
    )


def render_report_markdown(question: str, out: dict) -> str:
    lines = [
        "# WebScout Report",
        "",
        f"- **Question:** {question}",
        f"- **Generated:** {datetime.now().isoformat(timespec='seconds')}",
        f"- **Sufficient:** {out.get('sufficient', False)} | "
        f"Iterations: {out.get('iteration', 0)} | "
        f"Searches: {out.get('search_calls', 0)} | "
        f"Tokens: {out.get('total_tokens', 0)} | "
        f"Est. cost: ${out.get('total_cost', 0.0):.4f}",
        "",
        "## Answer",
        "",
        out.get("answer", ""),
        "",
    ]
    findings = out.get("findings") or []
    if findings:
        lines += ["## Findings", ""]
        for f in findings:
            refs = ", ".join(f.get("source_urls") or [])
            lines.append(
                f"- ({f.get('confidence', '?')}) {f.get('claim', '')}"
                + (f" — {refs}" if refs else "")
            )
        lines.append("")
    sources = out.get("sources") or []
    if sources:
        lines += ["## Sources", ""]
        for i, s in enumerate(sources, 1):
            lines.append(f"{i}. [{s.get('title', '')}]({s.get('url', '')})")
        lines.append("")
    return "\n".join(lines)


def write_report(question: str, out: dict, path: str) -> None:
    Path(path).write_text(render_report_markdown(question, out), encoding="utf-8")
```

Note: `_print_result` is unchanged from today's file — it's reproduced above only because it sits between the two functions being replaced; do not alter its body.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS — the 3 new tests, and every pre-existing test in this file (nothing about their assertions changed).

- [ ] **Step 5: Run the full offline suite and lint**

Run: `uv run pytest -m "not integration" -q && uv run ruff check app/main.py tests/test_cli.py`
Expected: all pass, `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add app/main.py tests/test_cli.py
git commit -m "feat: extract stream_pipeline generator and render_report_markdown

Single source of truth for driving the LangGraph pipeline, shared by
the CLI and the upcoming web UI. run_pipeline's printed output and
return shape are unchanged; write_report now wraps render_report_markdown."
```

---

### Task 2: `app/conversation.py` — `condense_question`

**Files:**
- Create: `app/conversation.py`
- Test: `tests/test_conversation.py`

**Interfaces:**
- Consumes: `get_model(role, settings)` from `app/models.py` and `get_settings()` from `app/config.py` (both unchanged).
- Produces: `condense_question(history: list[dict], question: str, settings=None, model=None) -> str`. `history` items are `{"question": str, "answer": str}` dicts, oldest first. The `model=` parameter is an injectable seam matching this repo's existing node-factory convention (e.g. `make_verify_node(s, model=...)`) — defaults to `get_model("verifier", settings)` when omitted.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_conversation.py`:

```python
from types import SimpleNamespace


def _fake_model(reply):
    def invoke(messages):
        return SimpleNamespace(content=reply)

    return SimpleNamespace(invoke=invoke)


class ExplodingModel:
    def invoke(self, messages):
        raise AssertionError("model must not be called when history is empty")


class RaisingModel:
    def invoke(self, messages):
        raise RuntimeError("rate limited")


class RecordingModel:
    def __init__(self, reply="Rewritten?"):
        self.reply = reply
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=self.reply)


def test_condense_question_skips_model_when_history_empty():
    from app.conversation import condense_question

    out = condense_question([], "What is LangGraph?", model=ExplodingModel())
    assert out == "What is LangGraph?"


def test_condense_question_rewrites_with_history():
    from app.conversation import condense_question

    history = [{"question": "What is LangGraph?", "answer": "A framework."}]
    out = condense_question(history, "and that?", model=_fake_model("What is LangChain?"))
    assert out == "What is LangChain?"


def test_condense_question_falls_back_on_model_error():
    from app.conversation import condense_question

    history = [{"question": "What is LangGraph?", "answer": "A framework."}]
    out = condense_question(history, "and that?", model=RaisingModel())
    assert out == "and that?"


def test_condense_question_blank_rewrite_falls_back_to_original():
    from app.conversation import condense_question

    history = [{"question": "What is LangGraph?", "answer": "A framework."}]
    out = condense_question(history, "and that?", model=_fake_model("   "))
    assert out == "and that?"


def test_condense_question_uses_last_three_turns_only():
    from app.conversation import condense_question

    history = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(5)]
    model = RecordingModel()
    condense_question(history, "and that?", model=model)
    sent = model.calls[0][1][1]  # messages == [("system", ...), ("human", text)]
    assert "q0" not in sent and "q1" not in sent
    assert "q2" in sent and "q3" in sent and "q4" in sent
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_conversation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.conversation'`

- [ ] **Step 3: Implement `app/conversation.py`**

```python
from app.config import Settings, get_settings
from app.models import get_model

_MAX_HISTORY_TURNS = 3

_CONDENSE_SYSTEM_PROMPT = (
    "You rewrite a follow-up question into a standalone question, using the "
    "conversation so far. Preserve the user's language and intent exactly. "
    "Output ONLY the rewritten question — no preamble, no quotes, no explanation. "
    "If the latest question is already standalone, return it unchanged."
)


def condense_question(
    history: list[dict],
    question: str,
    settings: Settings | None = None,
    model=None,
) -> str:
    """Rewrite `question` into a standalone question using prior turns.

    Returns `question` unchanged (no model call) when history is empty, and
    falls back to `question` unchanged if the rewrite call raises for any
    reason — a broken rewrite must never block the chat turn.
    """
    if not history:
        return question
    active_model = model or get_model("verifier", settings or get_settings())
    turns = history[-_MAX_HISTORY_TURNS:]
    transcript = "\n\n".join(
        f"Q: {t.get('question', '')}\nA: {t.get('answer', '')}" for t in turns
    )
    try:
        result = active_model.invoke(
            [
                ("system", _CONDENSE_SYSTEM_PROMPT),
                ("human", f"Conversation so far:\n{transcript}\n\nLatest question:\n{question}"),
            ]
        )
        rewritten = str(result.content).strip()
        return rewritten or question
    except Exception:
        return question
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_conversation.py -v`
Expected: PASS (5/5)

- [ ] **Step 5: Run the full offline suite and lint**

Run: `uv run pytest -m "not integration" -q && uv run ruff check app/conversation.py tests/test_conversation.py`
Expected: all pass, `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add app/conversation.py tests/test_conversation.py
git commit -m "feat: add condense_question for conversation-aware follow-ups

Rewrites a follow-up question into a standalone one using the last 3
prior turns, via the existing verifier role. Falls back to the
original question on empty history or any model failure."
```

---

### Task 3: `web/server.py` — FastAPI backend

**Files:**
- Create: `web/__init__.py` (empty)
- Create: `web/server.py`
- Create: `web/static/.gitkeep` (empty — makes the directory exist in git; `StaticFiles` requires the directory to exist at import time, and Task 4 fills it with real files)
- Modify: `pyproject.toml`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes:
  - `stream_pipeline(question, graph=None, max_iterations=None)` and `render_report_markdown(question, out) -> str` from `app/main.py` (Task 1).
  - `condense_question(history, question, settings=None, model=None) -> str` from `app/conversation.py` (Task 2).
  - `build_graph()` from `app/graph.py`, `MODEL_CHOICES`, `get_settings()`, `override_model(model)` from `app/config.py` (all unchanged, pre-existing).
- Produces: a FastAPI `app` object in `web/server.py` with routes `GET /`, `GET /static/*`, `GET /api/models`, `POST /api/chat` (SSE), `POST /api/report`; a `main()` console entry point; every name (`build_graph`, `condense_question`, `stream_pipeline`, `render_report_markdown`, `override_model`) imported as a **module-level bare name** in `web/server.py` (not accessed via `app.graph.build_graph()`), so tests can `monkeypatch.setattr(server, "build_graph", ...)` — matching this repo's existing pattern (`app/graph.py` looks up `build_research_agent` as a module attribute for the same reason).

- [ ] **Step 1: Add the `web` dependency group and script to `pyproject.toml`**

In `pyproject.toml`, change:

```toml
[project.scripts]
webscout = "app.main:main"
```
to:
```toml
[project.scripts]
webscout = "app.main:main"
webscout-web = "web.server:main"
```

Change:
```toml
[dependency-groups]
dev = ["pytest>=8.0", "ruff>=0.9"]
```
to:
```toml
[dependency-groups]
dev = ["pytest>=8.0", "ruff>=0.9"]
web = ["fastapi>=0.115", "uvicorn[standard]>=0.30"]
```

Change:
```toml
[tool.hatch.build.targets.wheel]
packages = ["app"]
```
to:
```toml
[tool.hatch.build.targets.wheel]
packages = ["app", "web"]
```

- [ ] **Step 2: Create the package skeleton**

```bash
mkdir -p web/static
touch web/__init__.py web/static/.gitkeep
```

(On PowerShell: `New-Item -ItemType Directory -Force web/static; New-Item -ItemType File web/__init__.py, web/static/.gitkeep -Force`)

- [ ] **Step 3: Install the `web` group**

Run: `uv sync --group web`
Expected: `fastapi`, `starlette`, `uvicorn` and friends install; `uv.lock` updates.

- [ ] **Step 4: Write the failing tests**

Create `tests/test_server.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `uv run pytest tests/test_server.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'web.server'`

- [ ] **Step 6: Implement `web/server.py`**

```python
import json
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import MODEL_CHOICES, get_settings, override_model
from app.conversation import condense_question
from app.graph import build_graph
from app.main import render_report_markdown, stream_pipeline

STATIC_DIR = Path(__file__).parent / "static"


class HistoryTurn(BaseModel):
    question: str
    answer: str


class ChatRequest(BaseModel):
    question: str
    history: list[HistoryTurn] = []
    model: str | None = None
    max_iterations: int | None = None


class ReportRequest(BaseModel):
    question: str
    out: dict


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


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/api/chat")
def chat(body: ChatRequest):
    if body.model:
        override_model(body.model)

    def gen():
        try:
            history = [t.model_dump() for t in body.history]
            question = condense_question(history, body.question)
            graph = build_graph()
            for kind, payload in stream_pipeline(
                question, graph=graph, max_iterations=body.max_iterations
            ):
                yield _sse(kind, {"node": payload} if kind == "status" else payload)
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

Run: `uv run pytest tests/test_server.py -v`
Expected: PASS (5/5)

- [ ] **Step 8: Run the full offline suite (with the `web` group installed) and lint**

Run: `uv run pytest -m "not integration" -q && uv run ruff check web/ tests/test_server.py pyproject.toml`
Expected: all pass (now 150+5=155-ish tests — exact count doesn't matter, zero failures), `All checks passed!`

- [ ] **Step 9: Manual smoke test — server actually starts**

Run: `uv run uvicorn web.server:app --port 8000` in one terminal, then in another:
```bash
curl -s http://127.0.0.1:8000/api/models
```
Expected: a JSON body with `"choices"` and `"key_configured"`. Stop the server (Ctrl+C) when confirmed — `GET /` will 404 or serve an empty file until Task 4 adds `index.html`; that's expected at this point.

- [ ] **Step 10: Commit**

```bash
git add web/__init__.py web/server.py web/static/.gitkeep pyproject.toml uv.lock tests/test_server.py
git commit -m "feat: add FastAPI chat backend with SSE streaming

/api/chat streams per-node status then the final result over
Server-Sent Events, building a fresh graph per request. /api/models
exposes the model shortlist and whether OPENROUTER_API_KEY is set.
/api/report renders the same markdown report the CLI --out produces.
New 'web' dependency group keeps fastapi/uvicorn out of the default
uv sync."
```

---

### Task 4: `web/static/` — browser chat frontend

**Files:**
- Create: `web/static/index.html`
- Create: `web/static/app.js`
- Create: `web/static/style.css`
- Modify: `tests/test_server.py` (extend)

**Interfaces:**
- Consumes: Task 3's exact contract — `GET /` , `GET /static/app.js`, `GET /static/style.css`, `GET /api/models` → `{choices, current, key_configured}`, `POST /api/chat` → SSE events `status {"node"}` / `result {...}` / `error {"message"}`, `POST /api/report` → `text/markdown`.
- Produces: a working browser chat UI. Nothing later in this plan depends on it (final task).

- [ ] **Step 1: Remove the placeholder and write the failing static-serving tests**

```bash
git rm web/static/.gitkeep
```

Append to `tests/test_server.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_server.py -k "index_page or static_assets" -v`
Expected: FAIL — `web/static/index.html` doesn't exist yet (404 or file-not-found error).

- [ ] **Step 3: Create `web/static/index.html`**

```html
<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <title>WebScout Chat</title>
  <link rel="stylesheet" href="/static/style.css" />
</head>
<body>
  <div id="app">
    <aside id="sidebar">
      <h1>WebScout</h1>
      <label for="model-select">Model</label>
      <select id="model-select"></select>
      <label for="max-iterations">Max iterations</label>
      <input id="max-iterations" type="number" min="1" max="10" value="3" />
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

- [ ] **Step 4: Create `web/static/style.css`**

```css
* {
  box-sizing: border-box;
}
body {
  margin: 0;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  background: #0f1115;
  color: #e6e6e6;
}
#app {
  display: flex;
  height: 100vh;
}
#sidebar {
  width: 240px;
  flex-shrink: 0;
  padding: 16px;
  background: #14161c;
  border-right: 1px solid #262a33;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
#sidebar h1 {
  font-size: 18px;
  margin: 0 0 8px;
}
#sidebar label {
  font-size: 12px;
  color: #9aa0ab;
}
#sidebar select,
#sidebar input {
  width: 100%;
  padding: 6px;
  background: #1c1f27;
  color: #e6e6e6;
  border: 1px solid #2c313c;
  border-radius: 6px;
}
.banner {
  margin-top: 12px;
  padding: 10px;
  background: #3a1f1f;
  border: 1px solid #6b2b2b;
  border-radius: 6px;
  font-size: 12px;
  color: #ffb4b4;
}
.hidden {
  display: none;
}
#chat {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
}
#log {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.bubble {
  max-width: 70%;
  padding: 10px 14px;
  border-radius: 10px;
  white-space: pre-wrap;
  line-height: 1.4;
}
.bubble.user {
  align-self: flex-end;
  background: #2d5be3;
  color: white;
}
.bubble.assistant {
  align-self: flex-start;
  background: #1c1f27;
  border: 1px solid #2c313c;
}
.bubble.assistant.error {
  border-color: #6b2b2b;
  color: #ffb4b4;
}
.bubble .sources,
.bubble .findings {
  margin-top: 10px;
  font-size: 13px;
}
.bubble .sources a {
  color: #7fb0ff;
}
.bubble .metrics {
  margin-top: 10px;
  font-size: 11px;
  color: #9aa0ab;
}
.bubble .download {
  margin-top: 8px;
  font-size: 12px;
  padding: 4px 10px;
  background: #262a33;
  color: #e6e6e6;
  border: 1px solid #363c48;
  border-radius: 6px;
  cursor: pointer;
}
#composer {
  display: flex;
  gap: 8px;
  padding: 16px;
  border-top: 1px solid #262a33;
}
#question {
  flex: 1;
  resize: none;
  padding: 10px;
  background: #1c1f27;
  color: #e6e6e6;
  border: 1px solid #2c313c;
  border-radius: 8px;
  font-family: inherit;
}
#send {
  padding: 0 18px;
  background: #2d5be3;
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
}
#send:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
```

- [ ] **Step 5: Create `web/static/app.js`**

```javascript
const logEl = document.getElementById("log");
const formEl = document.getElementById("composer");
const questionEl = document.getElementById("question");
const sendEl = document.getElementById("send");
const modelSelectEl = document.getElementById("model-select");
const maxIterEl = document.getElementById("max-iterations");
const bannerEl = document.getElementById("key-banner");

let turns = []; // {question, out}

const STATUS_LABELS = {
  research: "Đang research...",
  verify: "Đang verify...",
  answer: "Đang trả lời...",
};

function addBubble(role, text, extraClass) {
  const div = document.createElement("div");
  div.className = `bubble ${role}${extraClass ? " " + extraClass : ""}`;
  div.textContent = text;
  logEl.appendChild(div);
  logEl.scrollTop = logEl.scrollHeight;
  return div;
}

function renderResult(bubble, question, out) {
  bubble.textContent = out.answer || "";
  bubble.classList.remove("error");

  if (out.sources && out.sources.length) {
    const sources = document.createElement("div");
    sources.className = "sources";
    sources.innerHTML =
      "<strong>Sources</strong><br>" +
      out.sources
        .map(
          (s, i) =>
            `[${i + 1}] <a href="${s.url}" target="_blank" rel="noopener">${s.title || s.url}</a>`
        )
        .join("<br>");
    bubble.appendChild(sources);
  }

  if (out.findings && out.findings.length) {
    const findings = document.createElement("div");
    findings.className = "findings";
    findings.innerHTML =
      "<strong>Findings</strong><br>" +
      out.findings.map((f) => `(${f.confidence || "?"}) ${f.claim || ""}`).join("<br>");
    bubble.appendChild(findings);
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

async function sendQuestion(question) {
  addBubble("user", question);
  const thinking = addBubble("assistant", STATUS_LABELS.research);

  const history = turns.map((t) => ({ question: t.question, answer: t.out.answer }));
  const body = {
    question,
    history,
    model: modelSelectEl.value || null,
    max_iterations: Number(maxIterEl.value) || null,
  };

  let resp;
  try {
    resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (err) {
    thinking.textContent = `Lỗi kết nối: ${err.message}`;
    thinking.classList.add("error");
    return;
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop(); // keep the last, possibly incomplete frame
    for (const frame of frames) {
      const parsed = parseSseFrame(frame);
      if (!parsed) continue;
      const { event, data } = parsed;
      if (event === "status") {
        thinking.textContent = STATUS_LABELS[data.node] || `Đang ${data.node}...`;
      } else if (event === "result") {
        renderResult(thinking, question, data);
        turns.push({ question, out: data });
      } else if (event === "error") {
        thinking.textContent = `Lỗi: ${data.message}`;
        thinking.classList.add("error");
      }
    }
  }
}

formEl.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = questionEl.value.trim();
  if (!question) return;
  questionEl.value = "";
  sendQuestion(question);
});

questionEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    formEl.requestSubmit();
  }
});

async function loadModels() {
  const resp = await fetch("/api/models");
  const data = await resp.json();
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

loadModels();
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_server.py -v`
Expected: PASS (7/7 — the 5 from Task 3 plus the 2 new ones)

- [ ] **Step 7: Run the full offline suite and lint**

Run: `uv run pytest -m "not integration" -q && uv run ruff check web/ tests/test_server.py`
Expected: all pass, `All checks passed!` (ruff does not lint `.html`/`.js`/`.css` — nothing to check there)

- [ ] **Step 8: Manual end-to-end smoke test**

```bash
uv run uvicorn web.server:app --port 8000
```
In a browser, open `http://127.0.0.1:8000/`:
1. Confirm the model dropdown is populated and the page is not showing the API-key banner (assuming `.env` has `OPENROUTER_API_KEY` set).
2. Ask a question (e.g. "What is LangGraph?"). Confirm the assistant bubble cycles through "Đang research..." → "Đang verify..." → "Đang trả lời..." before showing the answer, sources, findings, and metrics line.
3. Ask a follow-up that only makes sense in context (e.g. "so sánh với cái trước"). Confirm the answer addresses the actual prior topic — this is `condense_question` working end to end, not just the offline mocks.
4. Click "Tải report.md" on a turn and confirm a `report.md` file downloads with the expected content.
5. Switch the model dropdown, ask another question, confirm it still works.
Stop the server (Ctrl+C) when done.

- [ ] **Step 9: Update `README.md`**

Add a short section (near the existing "Interactive" CLI usage section) documenting the new entry point:

```markdown
**Browser chat UI** — conversation-aware follow-ups, live per-node status, model picker:

```powershell
uv sync --group web
uv run uvicorn web.server:app --reload
```

Open `http://127.0.0.1:8000/`. This is a personal-testing tool: no login, no
persistence across page reloads — conversation history lives in the browser tab only.
```

- [ ] **Step 10: Commit**

```bash
git add web/static/index.html web/static/app.js web/static/style.css tests/test_server.py README.md
git commit -m "feat: add browser chat frontend

Single-page vanilla JS chat UI: SSE-driven live status per node,
model/max_iterations picker, per-turn markdown report download.
No build step, no framework, no persistence beyond the browser tab."
```

---

## Final verification (after all 4 tasks)

- [ ] Run: `uv sync --group web && uv run pytest -m "not integration" -q`
  Expected: all offline tests pass (Task 1-4 additions plus every pre-existing test, unmodified).
- [ ] Run: `uv run ruff check .`
  Expected: `All checks passed!`
- [ ] Run: `uv run pytest -m "not integration" -q` **without** `--group web` installed (fresh `uv sync` with no group) to confirm `pytest.importorskip("fastapi")` actually skips `tests/test_server.py` cleanly rather than erroring, proving CI (which does not install the `web` group) stays green.
- [ ] Confirm `uv run pytest -m integration` still passes (2 tests) — nothing in this plan touches the model layer or the graph itself.
