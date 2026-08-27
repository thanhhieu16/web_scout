# WebScout — ACP Integration (Agent Client Protocol)

**Status:** Approved for planning
**Builds on:** [2026-08-27-webscout-conversations-theme-design.md](2026-08-27-webscout-conversations-theme-design.md) (SQLite conversation store, already implemented). Realizes the "V4 ACP integration" item from the original `webscout_idea.md` (§27, §4.6, §33) — deferred out of scope by [2026-08-25-webscout-design.md](2026-08-25-webscout-design.md)'s Global Constraint 5, now in scope on its own.

## Goal

Expose WebScout as a third client surface — alongside the CLI (`app/main.py`) and the web chat UI (`web/server.py`) — that speaks [Agent Client Protocol](https://agentclientprotocol.com/) over stdio, so Zed editor (or any other ACP-compliant client) can drive a research turn and see live progress the same way the web UI's status trace does.

## Non-goals

- **Session resume.** Every `session/new` creates a brand-new conversation row. No `session/load`, no reconnecting to a conversation started in the web UI or a prior Zed session. Can be added later without breaking this design.
- **Per-session model / max-iterations override.** ACP prompts always run with `config.yaml`'s configured models and `max_iterations`, same as the plain CLI. No config surface is added for this.
- **Tool-level tracing.** Same node-level granularity (`research`/`verify`/`answer`) as the existing web UI trace — no per-`web_search`/`web_fetch`-call visibility. `stream_pipeline` doesn't emit that granularity today and instrumenting it is out of scope.
- **Any change to `app/graph.py`, the node contracts, or `web/server.py`'s existing routes/behavior** beyond the one refactor named below.
- **Permission prompts / tool-call approval flows** that ACP supports for editor-side tools (file edits, terminal) — WebScout's tools (`web_search`, `web_fetch`) are read-only and don't touch the client's filesystem, so none of ACP's permission machinery is exercised.

## Architecture

```
Zed editor
    │  stdio, JSON-RPC (ACP)
    ▼
app/acp_server.py  (new)
    │  acp.Agent subclass, run via run_agent()
    ▼
app/turn.py::run_chat_turn()  (new — extracted shared helper)
    │
    ├─ web/store.py         (existing — same data/webscout.db)
    ├─ app/conversation.py  (existing — condense_question)
    └─ app/main.py          (existing — build_graph, stream_pipeline)
```

`app/turn.py` is new: the turn logic currently inlined in `web/server.py`'s `chat()` handler (load conversation → condense follow-up question → stream the graph → persist the result) becomes a single generator both `web/server.py` and `app/acp_server.py` call. This is the same "one implementation of drive-the-graph-report-progress" principle the project already applies to `stream_pipeline` powering both the CLI and the web SSE endpoint (documented in `CLAUDE.md`) — ACP becomes a third caller of that same chain, not a second copy of the glue around it.

## `app/turn.py` (new)

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

    Raises KeyError if `conversation_id` doesn't exist in `db_path` — callers
    check existence themselves (via `store.get_conversation`) before calling
    this, so this is a programming-error guard, not an expected control path.
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

`web/server.py`'s `chat()` handler is rewritten to call this instead of inlining the same steps — its `gen()` body becomes:

```python
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
```

Note `run_chat_turn` itself never raises for pipeline failures (it converts them to a yielded `("error", ...)` tuple) — the outer `try/except` in `gen()` stays only as a backstop for the generator-construction/history-load step (e.g. `store.get_conversation` raising on a locked file), matching today's behavior. The existing `body.model` → `override_model()` call stays in `web/server.py` before `gen()` runs (ACP doesn't support per-session model override per the Non-goals above, so `run_chat_turn` itself takes no `model` parameter — model selection is process-global via `config.yaml`/`override_model`, unchanged from today).

## `app/acp_server.py` (new)

Subclasses `acp.Agent` from the `agent-client-protocol` PyPI package. Two methods:

- **`new_session`** — calls `store.create_conversation(db_path)`, returns the new row id (coerced to `str`, since ACP session ids are strings) as the session id. `db_path` comes from `get_settings().conversations_db_path` (same setting the web UI uses — same `data/webscout.db` file, so a conversation started in Zed is visible in the web UI's sidebar and vice versa).
- **`prompt`** — extracts the plain-text question from the incoming prompt content blocks, calls `run_chat_turn(db_path, conversation_id, question, max_iterations=None)` (module-level `max_iterations=None` — falls through to `Settings.max_iterations`'s configured default, same as the CLI), and for each yielded tuple:
  - `("status", {"node": name})` → emit a `session/update` notification of type `plan`, with three fixed entries (Research, Verify, Answer) whose `status` field is set to `pending` / `in_progress` / `completed` per which node just completed — mirrors the web UI's node-level trace chips exactly, reusing the same three-stage mental model.
  - `("result", out)` → emit one `session/update` notification of type `agent_message_chunk` carrying `out["answer"]` as its text content (Zed renders markdown client-side, same content the web UI now renders via `marked`+`DOMPurify`), then return `PromptResponse(stop_reason="end_turn")`.
  - `("error", {"message": msg})` → emit one `agent_message_chunk` containing the error message (prefixed, e.g. `f"⚠ Lỗi: {msg}"`, matching the web UI's error-bubble tone), then return `PromptResponse(stop_reason="end_turn")` — **not** a JSON-RPC-level error, so Zed shows a normal message bubble rather than tearing down the session (a broken research turn shouldn't kill the whole editor connection).

  **Implementation note for the implementer:** the exact Python call shape for building `plan` and `agent_message_chunk` `session/update` payloads (whether via `acp.helpers` builder functions or by constructing `acp.schema` Pydantic models directly) is not fully enumerated in the SDK's public docs as of this writing. Before writing this code, install `agent-client-protocol` and read `acp/helpers.py` and `acp/schema.py` from the installed package directly — do not guess function names from memory or from this spec. The `plan` variant's shape (confirmed via the protocol schema): a list of entries, each with `content` (a `ContentBlock`, e.g. a text block), `priority`, and `status` (`pending`/`in_progress`/`completed`).

- **Startup (`main()`, the `webscout-acp` console-script entry point):** calls `require_openrouter_key()` (existing, from `app/main.py`) before `run_agent()` — a missing key exits immediately with the existing clear stderr message, same as the CLI, rather than accepting a session and failing on first prompt.

## Dependencies

`pyproject.toml`:

```toml
[project.scripts]
webscout = "app.main:main"
webscout-web = "web.server:main"
webscout-acp = "app.acp_server:main"

[dependency-groups]
dev = ["pytest>=8.0", "ruff>=0.9"]
web = ["fastapi>=0.115", "uvicorn[standard]>=0.30"]
acp = ["agent-client-protocol"]
```

New group `acp`, separate from `dev` and `web` — same reasoning as the existing `web` group: never affects the default `uv sync` or CI. Install with `uv sync --group acp`.

## Zed configuration

Not part of this repo — documented in README as a usage note: Zed's `settings.json` needs an entry under its agent-servers config pointing at the `webscout-acp` executable (path depends on the user's venv), per Zed's own ACP-client docs. No code in this repo consumes or validates that config.

## Error handling

- `run_chat_turn` never lets a pipeline exception escape uncaught — converts to `("error", {"message": ...})`, matching the invariant `web/server.py` already has (failed turns aren't persisted) — this is now enforced in one place instead of two.
- `new_session` failures (e.g. DB file locked) propagate as a normal Python exception — the `acp` framework's stdio JSON-RPC layer converts an uncaught exception in a handler to a JSON-RPC error response; no special handling needed in `app/acp_server.py` beyond letting it propagate.
- Missing `OPENROUTER_API_KEY` — process exits at startup (before any session exists), same UX as running the plain CLI with no key configured.

## Testing

- **`tests/test_turn.py` (new):** exercises `run_chat_turn` directly against a `tmp_path`-backed sqlite db (`store.init_db`) and a fake graph (`GenericFakeChatModel`-backed, following the existing `monkeypatch.setattr(g, "build_research_agent", ...)` pattern used elsewhere) — asserts the yielded sequence for a normal turn, that a raised exception inside the pipeline becomes a yielded `("error", ...)` tuple with nothing persisted, and that a successful turn's `("result", ...)` is both yielded and present in `store.get_conversation(...)` afterward.
- **`tests/test_server.py` (extend):** re-run its existing `/api/chat` tests unchanged (behavior is identical after the refactor — same SSE events, same persistence) to confirm the `run_chat_turn` extraction didn't change observable behavior.
- **`tests/test_acp_server.py` (new):** starts with `pytest.importorskip("acp")` (mirrors `test_server.py`'s `pytest.importorskip("fastapi")`) so the default `uv sync`/CI, which doesn't install the `acp` group, skips this file cleanly. Tests call the `prompt`/`new_session` handler methods directly (not through a real stdio transport) with `run_chat_turn` monkeypatched to a fixed fake generator, and assert: the emitted `plan` updates transition `pending → in_progress → completed` in node order, the final `agent_message_chunk`'s text matches the fake result's `answer`, and an `("error", ...)` fake turn produces an `agent_message_chunk` (not a raised exception) and `stop_reason="end_turn"`.
- No integration test drives a real Zed↔stdio round-trip — manual verification against Zed is the acceptance check for the end-to-end wiring, same as how the web UI's SSE streaming was verified by running the server and using it, not by an automated browser test.
