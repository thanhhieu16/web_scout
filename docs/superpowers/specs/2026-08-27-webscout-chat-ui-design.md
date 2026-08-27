# WebScout Chat UI — Design Spec

**Status:** approved for planning
**Author:** Claude (brainstorming session), reviewed by repo owner
**Date:** 2026-08-27

## 1. Problem

The only way to run WebScout today is the CLI (`webscout "question"` one-shot, or the
interactive REPL). Manual testing means retyping questions, scrolling terminal output, and
losing every prior answer the moment a new question runs. The owner wants a chat-style
window in the browser for manual testing: type a question, watch the agent work, read the
answer with its sources — and be able to ask a follow-up that understands the prior turn,
the way a normal chatbot does.

This is a new subsystem (HTTP server, wire protocol, browser frontend) layered on the
existing LangGraph pipeline. It does not change `app/graph.py`, `app/state.py`, the node
files, or the reducers — those are correct and out of scope.

## 2. Goals / Non-goals

**Goals**
- Browser chat UI: type a question, see live per-node status while the agent runs, read
  the final answer with sources/findings/metrics.
- Follow-up questions resolve pronouns/references against the immediately preceding turns
  ("what about the second one?" understood in context).
- Model picker (from the existing `MODEL_CHOICES` shortlist) and a `max_iterations`
  control, for fast manual comparison during testing.
- Download the current turn's answer as the same markdown report the CLI's `--out`
  produces.
- Reuse `run_pipeline`'s graph-driving logic for both CLI and web — one implementation,
  not two.

**Non-goals**
- No persistence across page reloads or server restarts. No database, no session cookies,
  no login. This is a single-user local testing tool.
- No concurrent-tab isolation beyond what already exists for the CLI's `--model` (a
  process-wide `Settings` override). Two browser tabs picking different models
  simultaneously is a known, accepted limitation — see §7.
- No exposure of intermediate tool calls, raw `FINDINGS` blocks, `weak_claims`, or
  `gaps` in the UI. That is a different, not-yet-requested "debug view" use case.
- No production deployment concerns (auth, HTTPS, rate limiting, multi-user). Local
  `uvicorn --reload` only.

## 3. Architecture

```
Browser (single HTML page + vanilla JS)
   │  fetch() POST, reads response.body as it streams
   ▼
FastAPI app (web/server.py)
   │  builds a fresh build_graph() + condenses the question
   ▼
app/main.py::stream_pipeline()  ◄── same generator the CLI now uses
   │  drives the existing LangGraph pipeline (research → verify → answer)
   ▼
app/graph.py::build_graph()      (unchanged)
```

The web layer adds no new business logic. It formats the same events the CLI already
prints, as Server-Sent Events, and hands the client-supplied conversation history to a new
`condense_question` step before the question ever reaches the graph.

## 4. `app/main.py` changes

### 4.1 `stream_pipeline` — new generator, single source of truth

```python
def stream_pipeline(question: str, graph=None, max_iterations: int | None = None):
    """Drive the LangGraph pipeline, yielding progress then the final result.

    Yields ("status", node_name) once per completed node, in order, then yields
    exactly one ("result", out_dict) as the last item. `out_dict` has the same
    shape run_pipeline has always returned.
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
```

### 4.2 `run_pipeline` — rewritten in terms of `stream_pipeline`, same signature plus one new optional kwarg

```python
def run_pipeline(question: str, graph=None, max_iterations: int | None = None) -> dict:
    out = None
    for kind, payload in stream_pipeline(question, graph=graph, max_iterations=max_iterations):
        if kind == "status":
            print(f"[{payload}] ...", flush=True)
        else:
            out = payload
    return out
```

Every existing caller (`main()`, the REPL loop) and every existing test in
`tests/test_cli.py` keeps working unchanged: the printed lines and returned dict are
identical to today's behavior. `max_iterations` is additive and optional.

### 4.3 `render_report_markdown` — split out of `write_report`

```python
def render_report_markdown(question: str, out: dict) -> str:
    """Build the markdown report body. write_report() below just persists this string."""
    lines = [...]  # move the entire `lines = [...]` build and every line that follows it,
                    # verbatim, from the current write_report body (app/main.py:97-129) —
                    # everything up to but not including the `Path(path).write_text(...)`
                    # call. Change only the final statement to `return "\n".join(lines)`.


def write_report(question: str, out: dict, path: str) -> None:
    Path(path).write_text(render_report_markdown(question, out), encoding="utf-8")
```

Pure refactor, no behavior change: `render_report_markdown` gets the exact body
`write_report` builds today at [app/main.py:96-131](../../../app/main.py#L96-L131)
(question, generated timestamp, sufficient/iterations/searches/tokens/cost line, answer,
findings, sources) — `write_report` shrinks to the two-line wrapper above.
`tests/test_cli.py::test_write_report_markdown` must keep passing unchanged; add one new
test asserting `render_report_markdown(q, out) == path.read_text()` after `write_report`
writes it.

## 5. `app/conversation.py` — new module

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
    history: list[dict], question: str, settings: Settings | None = None
) -> str:
    """Rewrite `question` into a standalone question using prior turns.

    `history` is a list of {"question": str, "answer": str} dicts in
    chronological order (oldest first) — the shape the browser client sends.
    Returns `question` unchanged (no model call) when history is empty, and
    falls back to `question` unchanged if the rewrite call raises for any
    reason (timeout, rate limit, malformed response) — a broken rewrite must
    never block the chat turn.
    """
    if not history:
        return question
    s = settings or get_settings()
    model = get_model("verifier", s)
    turns = history[-_MAX_HISTORY_TURNS:]
    transcript = "\n\n".join(
        f"Q: {t.get('question', '')}\nA: {t.get('answer', '')}" for t in turns
    )
    try:
        result = model.invoke(
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

Uses the `verifier` role (already temperature 0.0, already in `ROLE_NAMES` — no new config
field). Offline-testable with `GenericFakeChatModel` exactly like `tests/test_agent.py`
already does elsewhere in this repo.

## 6. `web/server.py` — FastAPI app

### 6.1 Routes

| Route | Method | Body | Response |
|---|---|---|---|
| `/` | GET | — | `web/static/index.html` |
| `/static/*` | GET | — | static files (`app.js`, `style.css`) |
| `/api/models` | GET | — | `{"choices": [...MODEL_CHOICES], "current": <default model>, "key_configured": bool}` |
| `/api/chat` | POST | `{"question": str, "history": [{"question","answer"}], "model": str \| null, "max_iterations": int \| null}` | `text/event-stream` (see §6.2) |
| `/api/report` | POST | `{"question": str, "out": <result dict from a `result` event>}` | `text/markdown` body, `Content-Disposition: attachment; filename="report.md"` |

`/api/report` calls `render_report_markdown(question, out)` directly — no duplicated
markdown-building logic in the web layer.

### 6.2 `/api/chat` SSE framing

Each event is `event: <name>\ndata: <json>\n\n`, UTF-8, flushed as produced.

```
event: status
data: {"node": "research"}

event: status
data: {"node": "verify"}

event: status
data: {"node": "answer"}

event: result
data: {"answer": "...", "sources": [...], "findings": [...], "search_calls": 2,
        "sufficient": true, "iteration": 1, "total_tokens": 4200, "total_cost": 0.0031}

```

On any exception while driving the pipeline (missing API key, model 404, network error,
rate limit exhausted after retries): emit exactly one `event: error` with
`data: {"message": "<str(exc)>"}` instead of `result`, then close the stream. The endpoint
wraps the whole per-node loop in `try/except Exception`.

### 6.3 Handler sketch

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
    out: dict  # the exact payload from the SSE "result" event for that turn


app = FastAPI()


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
            graph = build_graph()  # fresh graph + UsageCollector per request — see §7
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
        md, media_type="text/markdown",
        headers={"Content-Disposition": 'attachment; filename="report.md"'},
    )


def main() -> None:
    """Entry point for the `webscout-web` console script."""
    import uvicorn

    uvicorn.run("web.server:app", host="127.0.0.1", port=8000, reload=False)
```

`gen()` is a plain **sync** generator. Starlette's `StreamingResponse` runs a sync
generator in a threadpool automatically (`iterate_in_threadpool`) — this is what lets the
blocking `g.stream(...)` call (LangGraph's `.stream()` is synchronous) run inside an async
FastAPI app without an explicit `async def` / `run_in_threadpool` wrapper in application
code.

## 7. Global Constraints (binding on every task)

1. **Fresh graph per request.** `/api/chat` calls `build_graph()` inside the request
   handler, never a module-level or cached graph. `build_graph()` already constructs a new
   `UsageCollector` per call (`app/graph.py`) — this is the same reason
   `evals/run_evals.py::_graph()` is explicitly NOT `@lru_cache`d (see the comment there):
   a shared graph means a shared `UsageCollector`, and one in-flight request's `research`
   node can drain another request's usage. Do not add caching here.
2. **`override_model` is process-global**, exactly like the CLI's `--model` flag already
   is. Two browser tabs selecting different models concurrently will race — this is an
   accepted limitation of a single-user local tool, not a bug to fix in this plan. Do not
   add per-request settings isolation; that is out of scope.
3. **No new config fields.** `condense_question` reuses the existing `verifier` role.
   `max_iterations` per-request override does not touch `config.yaml` or `Settings` at
   all — it is a plain function argument threaded through `stream_pipeline`.
4. **`run_pipeline`'s CLI-visible behavior does not change.** Printed lines and the
   returned dict shape must be byte-for-byte identical to before this work — every
   existing test in `tests/test_cli.py` that asserts on them must keep passing unmodified.
5. **The `web` dependency group must not affect the default `uv sync`.** `fastapi` and
   `uvicorn` go in a new `[dependency-groups] web = [...]` entry. CI (`uv sync` with no
   `--group`) must stay green with the group absent — any test importing `fastapi` starts
   with `pytest.importorskip("fastapi")`.
6. **The offline test suite (`-m "not integration"`) stays network-free.** `TestClient`
   from `fastapi.testclient` (backed by `httpx`, already a base dependency) makes in-process
   ASGI calls, not real sockets — safe for the offline suite once `fastapi` is installed via
   the `web` group. Tests that would otherwise hit OpenRouter (a real `/api/chat` call)
   must monkeypatch `build_graph` or pass a fake graph the same way `tests/test_cli.py`
   already does with `FakeGraph`.

## 8. Frontend (`web/static/`)

Single page, vanilla JS, no build step, no framework.

- **`index.html`** — chat log container, `<textarea>` + submit button pinned to bottom
  (Enter to send, Shift+Enter for newline), sidebar with a `<select>` populated from
  `GET /api/models` and a `max_iterations` `<input type="number">`.
- **`app.js`**:
  - Two in-memory arrays, never persisted (`localStorage` explicitly out of scope — see §2
    non-goals):
    - `turns = []` — one entry per completed exchange, `{question, out}` where `out` is
      the exact `result` event payload (answer/sources/findings/metrics). This is what
      renders each assistant bubble and what `/api/report` needs.
    - the **request** history sent to `/api/chat` is derived from `turns` at submit time:
      `turns.map(t => ({question: t.question, answer: t.out.answer}))` — only
      question/answer text, matching `HistoryTurn` in §6.3. The server itself truncates to
      the last 3 turns (`_MAX_HISTORY_TURNS`), so the client sends the whole array.
  - On submit: push a user bubble, `fetch('/api/chat', {method:'POST', body: JSON.stringify({question, history, model, max_iterations})})`, read `response.body.getReader()`, decode chunks, split on `\n\n` to parse SSE frames (a `POST` body means the browser's built-in `EventSource` — GET-only — cannot be used here; this manual parse is the standard workaround).
  - `status` events update a single "thinking" bubble's text (`Đang research...` /
    `Đang verify...` / `Đang trả lời...` — Vietnamese labels matching the CLI's node
    names). `result` replaces that bubble with the final answer (rendered as plain text
    with line breaks — no markdown renderer dependency, keep it minimal), a numbered
    source list, a findings list (confidence + claim), and a metrics line. `error` replaces
    it with a red inline error message; on `error`, nothing is pushed to `turns`.
  - On successful `result`, push `{question, out: <result payload>}` onto `turns` and
    enable a "Download report" button for that turn, which POSTs `{question, out}` to
    `/api/report` and triggers a file save from the response blob.
  - On page load: `GET /api/models`; if `key_configured` is false, disable the input and
    show a banner instead of allowing a doomed first request.
- **`style.css`** — minimal chat bubble layout (user right-aligned, assistant
  left-aligned), no CSS framework dependency.

## 9. Testing plan (offline, `-m "not integration"`)

- `tests/test_conversation.py` (new):
  - empty history → `condense_question` returns the question unchanged, model never
    invoked (assert via a fake model whose `.invoke` raises if called).
  - non-empty history → fake model returns a rewritten string, `condense_question` returns
    it.
  - fake model raises → `condense_question` returns the original question unchanged.
- `tests/test_cli.py` (extend):
  - `stream_pipeline` yields `("status", "research")`, `("status", "verify")`,
    `("status", "answer")` then exactly one `("result", {...})` last, using the existing
    `FakeGraph`/`DELTAS` fixtures.
  - `run_pipeline` behavior tests already present (`test_run_pipeline_prints_progress_and_answer`,
    `test_run_pipeline_survives_two_research_iterations`,
    `test_run_pipeline_returns_usage_fields`) must pass with zero changes to their
    assertions — proves the refactor is behavior-preserving.
  - new: `render_report_markdown(q, out)` matches what `write_report` persists to disk.
  - new: `run_pipeline(q, graph=FakeGraph(...), max_iterations=1)` reaches the graph's
    `state["max_iterations"]` as `1` (fake graph captures the `state` dict it was invoked
    with).
- `tests/test_server.py` (new, `pytest.importorskip("fastapi")` at top):
  - `GET /api/models` returns the `MODEL_CHOICES` list and a `key_configured` boolean.
  - `POST /api/chat` with a monkeypatched `build_graph` (returns a `FakeGraph`) — parse the
    SSE body text, assert `status` events appear before the `result` event, assert the
    `result` payload matches the fake graph's expected output.
  - `POST /api/chat` where the monkeypatched graph raises — assert the SSE body contains
    an `error` event, not a 500.
  - `POST /api/report` returns markdown text equal to `render_report_markdown(...)` for a
    known input.

## 10. Dependencies & run commands

`pyproject.toml`:
```toml
[dependency-groups]
dev = ["pytest>=8.0", "ruff>=0.9"]
web = ["fastapi>=0.115", "uvicorn[standard]>=0.30"]

[project.scripts]
webscout = "app.main:main"
webscout-web = "web.server:main"   # thin wrapper: uvicorn.run(app, ...)
```

`[tool.hatch.build.targets.wheel] packages` gains `"web"` alongside `"app"` so the console
script resolves the same way in an installed environment as it does under `uv run`.

```powershell
uv sync --group web
uv run uvicorn web.server:app --reload
# or, once installed:
uv run webscout-web
```

## 11. Out of scope / explicitly deferred

- Markdown rendering of the answer in the browser (currently plain text) — cheap to add
  later, not required for a functional test tool.
- `localStorage` persistence of chat history across reloads.
- Any debug/internals view (raw tool calls, `weak_claims`, `gaps`) — a different, later
  request per the brainstorming session (see the four probe options originally offered:
  this spec implements only "chạy thử thủ công").
- Multi-user session isolation, auth, HTTPS.
