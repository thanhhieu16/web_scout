# WebScout Chat UI — Conversations, Theme Toggle, Status Trace

**Status:** Approved for planning
**Builds on:** [2026-08-27-webscout-chat-ui-design.md](2026-08-27-webscout-chat-ui-design.md) (FastAPI + SSE + vanilla JS chat UI, already implemented) and the subsequent "Case File" visual redesign (dossier theme, findings ledger, citation tabs — implemented directly, no spec).

## Goal

Three additions to the existing single-page chat UI:

1. **Multiple conversations**, persisted server-side (survive a browser restart, a different browser, a server restart).
2. **Light/dark theme toggle** sharing one token system (same fonts, same accent, same "Case File" identity — only the time-of-day changes).
3. **Status trace** — the pipeline's node sequence (`research → verify → research → verify → answer`) stays visible per turn instead of being overwritten, so the user can see how many iterations a question took.

## Non-goals

- Multi-user auth, multi-tab live sync, multi-device sync beyond "same server, same DB file."
- Tool-level tracing (which `web_search`/`web_fetch` call ran inside a research node) — node-level only, per the approved design.
- A migration path from the old client-side `history` array — this is a pre-release personal tool; the old `ChatRequest.history` field is removed outright, not kept alongside the new contract.
- Any change to the LangGraph pipeline, `app/graph.py`, or the node contracts documented in the project's `CLAUDE.md`. This spec is UI + a new persistence layer only.

## Architecture

```
Browser (app.js)
  ├─ on load: GET /api/conversations → pick most-recent or POST create
  ├─ GET /api/conversations/{id} → replay stored messages into #log
  ├─ POST /api/chat {conversation_id, question, model, max_iterations}
  │     (SSE: status* → result | error)
  └─ sidebar: new / rename (PATCH) / delete (DELETE) conversations,
     theme toggle (localStorage only, no server round-trip)

FastAPI (web/server.py)
  └─ web/store.py (new) — sqlite3, stdlib only, no new dependency
        conversations(id, title, created_at, updated_at)
        messages(id, conversation_id, question, answer_json, created_at)
```

`web/store.py` owns all SQL. `web/server.py` never touches `sqlite3` directly — same separation-of-concerns the project already uses between `app/graph.py` (orchestration) and `app/tools/*` (I/O).

## Data model

New file: `data/webscout.db` (created on first run). Path comes from a new `Settings.conversations_db_path` field in [app/config.py](../../../app/config.py), default `str(REPO_ROOT / "data" / "webscout.db")` — same pattern `_yaml_path()` already uses for locating `config.yaml` relative to `REPO_ROOT`. Add `data/` to `.gitignore` alongside the existing `.env` entry.

```sql
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    question TEXT NOT NULL,
    answer_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

- Timestamps: `datetime.now(timezone.utc).isoformat()`, stored as TEXT (sqlite has no native datetime type; ISO-8601 strings sort correctly).
- `answer_json`: `json.dumps(out)` where `out` is exactly the dict `stream_pipeline`'s `"result"` event yields (`answer`, `sources`, `findings`, `search_calls`, `sufficient`, `iteration`, `total_tokens`, `total_cost`) — no separate schema to keep in sync; storing and re-rendering use the identical shape the SSE stream already produces.
- Every connection: `sqlite3.connect(path, timeout=5)` (short-lived, opened and closed per call — no shared connection, no threading concerns), immediately followed by `conn.execute("PRAGMA foreign_keys = ON")` so the `ON DELETE CASCADE` above actually cascades (sqlite disables FK enforcement by default).
- Schema creation: `init_db()` runs `CREATE TABLE IF NOT EXISTS` for both tables and is called once at `web/server.py` import time (module-level, same place `STATIC_DIR` is defined) plus creates the parent `data/` directory if missing.

### `web/store.py` functions

All take `db_path: str` as their first parameter (no module-level global connection or cache — mirrors the project's existing "no hidden shared state" pattern in `app/config.py`'s settings, and makes the module trivially testable with a tmp path).

- `init_db(db_path: str) -> None` — create parent dir + both tables if absent.
- `create_conversation(db_path: str, title: str = "Cuộc hội thoại mới") -> int` — insert, return new id. `created_at == updated_at` at insert time.
- `list_conversations(db_path: str) -> list[dict]` — `[{"id", "title", "updated_at"}, ...]` ordered by `updated_at DESC`.
- `get_conversation(db_path: str, conversation_id: int) -> dict | None` — `{"id", "title", "messages": [{"question", "out": {...}}, ...]}` ordered by `messages.id ASC` (chronological). Returns `None` if the id doesn't exist (caller maps this to a 404).
- `append_message(db_path: str, conversation_id: int, question: str, out: dict) -> None` — insert the message; also updates `conversations.updated_at` to now; if this is the conversation's first message **and** its title is still the default `"Cuộc hội thoại mới"`, sets the title to `question[:40] + ("…" if len(question) > 40 else "")`. Raises `KeyError` if `conversation_id` doesn't exist (caller maps this to a 404).
- `rename_conversation(db_path: str, conversation_id: int, title: str) -> bool` — `title` is trimmed by the caller (route) before this is called; returns `False` if the id doesn't exist (caller maps this to 404), `True` on success. Does not touch `updated_at` (renaming isn't "activity").
- `delete_conversation(db_path: str, conversation_id: int) -> bool` — `False` if the id didn't exist, `True` on success. Cascades to `messages` via the FK.

## API

All new routes live in `web/server.py` next to the existing ones, using the same bare-function style (no router class — the file is small enough that one more won't hurt readability; if it grows past ~150 lines a future pass can split it, but that's not this task).

- **`GET /api/conversations`** → `200 [{"id": 1, "title": "...", "updated_at": "..."}]`
- **`POST /api/conversations`** (no body) → `200 {"id": 1, "title": "Cuộc hội thoại mới"}`
- **`GET /api/conversations/{id}`** → `200 {"id": 1, "title": "...", "messages": [{"question": "...", "out": {...}}]}`, or `404 {"detail": "conversation not found"}`
- **`PATCH /api/conversations/{id}`** body `{"title": str}` → `200 {"id": 1, "title": "..."}`. Server trims the title; a title that's empty after trimming is rejected with `400 {"detail": "title cannot be empty"}`. Missing id → `404`.
- **`DELETE /api/conversations/{id}`** → `204` empty body, or `404` if the id didn't exist.
- **`POST /api/chat`** — `ChatRequest` changes shape:
  ```python
  class ChatRequest(BaseModel):
      conversation_id: int
      question: str
      model: str | None = None
      max_iterations: int | None = Field(default=None, ge=1, le=10)
  ```
  `HistoryTurn` and the `history` field are deleted — history is no longer client-supplied. Inside the handler, before streaming starts: `conversation = store.get_conversation(db_path, body.conversation_id)`; a missing conversation short-circuits with `404` (raised before the `StreamingResponse` is constructed, so it's a normal FastAPI JSON error response, not an SSE `error` event — the conversation has to exist before there's anywhere to stream into). `history` for `condense_question` is built from `conversation["messages"]` as `[{"question": m["question"], "answer": m["out"]["answer"]} for m in conversation["messages"]]` (condense_question already only looks at the last 3 turns, per its existing `_MAX_HISTORY_TURNS`). After the pipeline's `"result"` event is produced (inside the generator, same place the event is yielded), call `store.append_message(db_path, body.conversation_id, body.question, payload)` — this makes the persisted record and the SSE payload the same object, so there's no risk of them drifting apart.

  Db_path is read once via `get_settings().conversations_db_path` at the top of the route (not per-call `get_settings()` scattered around) and passed down into every `store.*` call.

## Frontend

### Sidebar restructure ([web/static/index.html](../../../web/static/index.html))

```
#sidebar (flex column)
 ├─ #sidebar-header (flex row: wordmark, theme-toggle button)
 ├─ #new-conversation button ("+ Cuộc hội thoại mới")
 ├─ #conversation-list (flex:1, overflow-y:auto — grows/scrolls, pushes settings down)
 │    each item: title span (click → select) + ✎ button (rename) + × button (delete)
 ├─ #settings (model select, max-iterations — unchanged fields, just relocated)
 └─ #key-banner (unchanged, stays last / pinned to bottom via margin-top:auto)
```

The active conversation's list item gets `.active` (accent left-border, matching the citation-tab amber already used elsewhere — no new color introduced).

### Page load sequence (`app.js`)

1. `loadModels()` (unchanged — populates `#model-select`, shows the key banner if unconfigured).
2. `loadConversations()`: `GET /api/conversations`. If the list is non-empty, select `list[0]` (already most-recent-first from the API). If empty, `POST /api/conversations` then select the new one.
3. `selectConversation(id)`: `GET /api/conversations/{id}`, clear `#log`, replay each stored message — a user bubble via the existing `addBubble("user", m.question)`, and an assistant dossier card via a fresh `div.bubble.assistant` fed straight into `renderResult(bubble, m.question, m.out)` (no SSE involved for replay — `renderResult` already renders a complete `out` object synchronously). Sets `activeConversationId` and re-renders the sidebar list's `.active` marker.

### Sending a question

`sendQuestion` drops `history` from the request body and adds `conversation_id: activeConversationId`. On the `"result"` SSE event, after `renderResult` runs, call `loadConversations()` again (cheap — one `SELECT`) to pick up the server-computed title (if this was the conversation's first message) and updated ordering, then re-mark `.active` on `activeConversationId` without changing the current selection or re-fetching messages.

### New / rename / delete

- **New**: `POST /api/conversations`, prepend to the in-memory list, `selectConversation(newId)`.
- **Rename**: `✎` calls `window.prompt("Đổi tên hội thoại:", currentTitle)`. `null` (cancel) or empty-after-trim → no-op. Otherwise `PATCH /api/conversations/{id}` then update the list item's text in place.
- **Delete**: `×` calls `window.confirm("Xóa hội thoại này?")`; on confirm, `DELETE /api/conversations/{id}`. If the deleted conversation was active: pick the new `list[0]` if any conversations remain, else create a fresh one (same as empty-list-on-load) and select it.

### Status trace

Today `sendQuestion` creates one placeholder bubble via `addBubble("assistant", STATUS_LABELS.research, "pending")` and overwrites its `textContent` on every `"status"` event; `renderResult` then wipes the bubble and rebuilds it from scratch (`bubble.textContent = ""`).

New structure: `sendQuestion` builds the pending bubble directly instead of going through `addBubble` for this one case —

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
```

- On each `"status"` event: the previous last `.chip.active` (if any) loses `.active`; a new `span.chip.active` is appended to `.trace` with the node name (`research`/`verify`/`answer`), and `status.textContent` updates as it does today.
- On `"result"`: the last chip loses `.active` (settles), and `renderResult` is changed to **not** clear the whole bubble — it removes every child except the existing `.trace` element, then appends the eyebrow/answer/findings/sources/metrics/download after it. Replayed conversations (from `selectConversation`) have no live trace to show — `renderResult` handles a bubble with no `.trace` child the same way (nothing to preserve, nothing breaks).
- On error paths: same "keep `.trace`, replace the rest" treatment, so a failed turn still shows how far the pipeline got before failing.

Visually: chips are small pill labels (reusing the `.citation-tab`-style mono/amber/bordered look already established, so no new visual language), settled chips slightly dimmer than the active/pulsing one, connected by a thin `→` separator (CSS `::after` on non-last chips, not real DOM nodes — one class, no JS string-building for the arrows).

### Theme toggle

- `html[data-theme="light"]` / `html[data-theme="dark"]` attribute, toggled by a sun/moon icon button (`#theme-toggle`, text glyph `"☀"`/`"☾"`, no icon font/SVG sprite needed).
- `style.css` restructures the existing `:root` block: **dark stays the default** (bare `:root`, matching the current look with zero visual change for anyone who never touches the toggle), a light palette is defined under `:root[data-theme="light"]`. Concretely:
  - dark (existing, unchanged): `--bg:#0c0d10 --panel:#15171c --panel-2:#1b1e25 --border:#242830 --text:#e9e6dc --text-dim:#9a9689`, accent `#c98a3f`
  - light (new): `--bg:#f3efe6 --panel:#ffffff --panel-2:#faf7f0 --border:#ddd5c4 --text:#1c1a16 --text-dim:#6b6559`, accent `#9c5f1f` (darkened from the dark-mode amber so it still passes contrast on a light background), confidence colors darkened the same way (`#3f8f6c` / `#a67a1f` / `#a83f38`) — everything else (fonts, spacing, the citation-tab / finding-row / dossier-card structure) is untouched; only the six palette tokens and the three confidence tones get a second value.
- On load: read `localStorage.getItem("webscout-theme")`. If present, apply it. If absent, apply `window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark"` and do **not** write to localStorage yet (so the user's OS setting keeps winning until they explicitly click the toggle once — matches the approved "system default until overridden" behavior). Clicking the toggle always writes the explicit choice to localStorage from then on.

## Error handling

- Any `store.py` function receiving an id that doesn't exist returns `None`/`False`/raises `KeyError` (per function, as specified above) rather than raising a generic `sqlite3` error up through the route — routes translate these into `404`, never a `500`.
- `POST /api/chat` with an invalid `conversation_id`: `404` before any LLM call is made (checked first, per the API section above) — no wasted OpenRouter spend on a request that can't be persisted anyway.
- Network/stream failures during `/api/chat` are unchanged from the existing implementation (already handled client-side per the current `sendQuestion`'s try/catch/finally) — this spec doesn't touch that path except for where the trace chips live inside the bubble.
- If `loadConversations()` itself fails on page load (server unreachable), surface it the same way the existing `key-banner` pattern does: a visible banner, composer disabled. (Reuses the existing `.banner` CSS class — no new error-banner styling needed.)

## Testing

- `tests/test_store.py` (new): every `web/store.py` function against a `tmp_path`-backed sqlite file — create/list/get/append/rename/delete, the cascade-delete behavior, the auto-title-from-first-message behavior, and the "second message doesn't overwrite a title" behavior.
- `tests/test_server.py` (extend, already `pytest.importorskip("fastapi")`-gated): the five new routes (happy path + 404s), and `POST /api/chat` using a real temp DB (via `monkeypatch` on `get_settings().conversations_db_path`, following the existing pattern in that test file for injecting fakes) to assert a message actually lands in the conversation after a chat call.
- No new JS test tooling is introduced (the project has none today) — the frontend changes are verified by running the app manually, consistent with how the original chat UI and the "Case File" redesign were both verified.
