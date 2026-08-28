# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```powershell
uv sync                                        # install (dev group = pytest, ruff)

uv run webscout "question"                     # one-shot run (full graph)
uv run webscout                                # interactive REPL
uv run webscout "q" --out report.md            # + markdown report
uv sync --group web && uv run uvicorn web.server:app --reload   # browser chat UI (see web/server.py)
uv sync --group acp                            # ACP agent for Zed (see app/acp_server.py) — combine with --group web if you want both

uv run pytest -m "not integration"             # offline suite (what CI runs)
uv run pytest -m integration                   # hits OpenRouter + live web
uv run pytest tests/test_node_verify.py::test_verify_returns_own_weak_claims_only   # single test

uv run python -m evals.run_evals --limit 2     # LangSmith eval (needs LANGSMITH_API_KEY)
uv run python -m evals.summarize               # A/B arm table from LangSmith projects
```

`uv sync` installs the `dev` dependency group (pytest, ruff) by default — no `--extra` flag. The `webscout` console script is installed by the hatchling build backend. On Windows consoles, set `$env:PYTHONIOENCODING="utf-8"` if printing the answer raises `cp65001` errors.

## Architecture

Two nested loops. **Deep Agents** owns the inner agent loop (LLM → tool → observation); **LangGraph** owns the outer product loop with exactly three nodes:

```
START → research → verify →(insufficient AND iteration < max)→ research
                     └────(sufficient OR budget exhausted)───→ answer → END
```

`verify` and `answer` are single bounded LLM calls — no tools, no internal loops. Routing lives in [route_after_verify](app/graph.py#L12).

### Model layer is where the load-bearing hacks are

[app/models.py](app/models.py) subclasses `ChatOpenAI` as `ResearchChatOpenAI` for two reasons, both discovered in the M0 spike ([notes](docs/superpowers/notes/2026-08-25-m0-spike-findings.md)) — do not "clean these up":

- `use_responses_api = False` is **required**. langchain-openai auto-routes to the Responses API when the payload carries a non-`function` tool type, which then crashes with `responses.create(messages=...)`.
- `_create_chat_result` lifts OpenRouter's `choices[0].message.annotations` into `additional_kwargs["annotations"]`, because langchain-openai drops that field. Every URL citation in the pipeline flows through this lift.

### One search path

`build_search_spec` produces the `openrouter:web_search` spec, but nothing wires it onto `ResearchChatOpenAI`'s bind — the server-tool injection path (`server_tools` field on the model + `attach_server_tools`) was deleted as dead code, since it was unwired after the `audit-remediation` branch and the v3 experiments already showed the researcher model tended to skip the raw server tool and discover pages via `web_fetch` alone ([v3 notes](docs/superpowers/notes/v3-experiments.md)). `build_search_spec` survives only because the **client-side `web_search` tool** ([app/tools/search_tool.py](app/tools/search_tool.py)) calls it directly — a LangChain tool that makes its own OpenRouter `/chat/completions` request carrying that spec, then flattens the annotations into `[SRC] url | title` / `EXCERPT:` text blocks. **This is what the agent actually gets** ([build_research_agent](app/agent.py)). Call count for this path is tracked via `UsageCollector`, not `usage.server_tool_use_details.web_search_requests` (that field is only populated by the now-removed server-tool path; docs claim `server_tool_use`; [count_web_searches](app/tools/search.py) reads both for when a server-tool path returns).

### Sources are reconstructed, never returned

The agent returns only prose. [build_sources](app/nodes/parsing.py) rebuilds the ordered source list by scanning the whole message history in this precedence: annotation citations → `[SRC]` blocks parsed out of `web_search` tool messages → successful `web_fetch` results (search-engine result pages filtered by `_SEARCH_ENGINE_MARKERS`). That order defines the `[Sn]` numbering.

### The FINDINGS contract

The research prompt ends every reply with a `## FINDINGS` block whose lines are parsed by strict regex in [parse_findings_block](app/nodes/parsing.py) (`- [Sn]... claim | confidence: high|medium|low`). A line may carry several refs (`- [S1][S2] claim | ...`); `parse_findings_block` returns a `FindingsParse` NamedTuple whose `dropped` and `block_found` fields surface contract violations as `weak_claims`. `map_refs_to_urls` then resolves each `Sn` to the nth URL from `build_sources`; unresolvable refs become `unresolved:Sn` and are surfaced as `weak_claims` by the research node. **`RESEARCH_SYSTEM_PROMPT`, `_LINE_RE`, and `map_refs_to_urls` must change together.**

### State accumulation runs through reducers

[ResearchState](app/state.py) is a `TypedDict` whose cumulative fields carry `Annotated[..., reducer]`: `findings` (`merge_findings`), `sources` (`merge_sources`), `weak_claims` (`merge_weak_claims`), and `iteration` / `search_calls` / `total_tokens` / `total_cost` (`operator.add`). **Nodes return only their own delta** — returning a running total would double-count it.

`merge_sources` dedupes by url with first-occurrence order, which is what keeps `[Sn]` numbering stable once a second research iteration runs. `merge_findings` dedupes by normalized claim text and takes the *lower* confidence on a merge: iteration 2 is instructed to research gaps only, so a repeated claim is not independent confirmation.

`gaps`, `sufficient`, `contradictory_claims`, and `answer` deliberately have no reducer — they are the current verdict, not a ledger.

### Config

pydantic-settings with source order **init kwargs > env > `.env` > `config.yaml`**. `.env` is auto-loaded at import time in [app/config.py](app/config.py). `get_settings()` is `lru_cache`d — tests construct `Settings(_env_file=None)` directly to avoid picking up the developer's real `.env`. Note `skills_enabled` defaults to `False` in code but `true` in `config.yaml`. Swapping models is a one-line `config.yaml` edit (`researcher` / `verifier` / `answer` / `judge` roles).

Every LLM call goes through [call_with_backoff](app/backoff.py). Its default is 5 attempts, linear 20s·n backoff, retrying only `OpenAIRateLimitError` — but `retry_on` accepts either a tuple of exception types or a predicate, and callers with a different failure profile pass their own: `web_search` ([app/tools/search_tool.py](app/tools/search_tool.py)) uses `attempts=3, base_delay=2.0` with an HTTP-status predicate, so a hung search fails fast instead of blocking on the LLM-call defaults.

### Web chat UI

`web/server.py` (FastAPI) streams `app/main.py`'s `stream_pipeline` generator as Server-Sent Events for `POST /api/chat`; the same generator also backs the CLI's `run_pipeline`, so there is one implementation of "drive the graph, report progress," not two. Conversation-aware follow-ups go through `app/conversation.py::condense_question` before reaching the graph. The `web` dependency group (`fastapi`, `uvicorn`) is separate from `dev` so it never affects the default `uv sync` or CI.

Conversations persist server-side in `data/webscout.db` via `web/store.py` (stdlib `sqlite3`, no new dependency) — `POST /api/chat` now takes a `conversation_id` instead of a client-supplied `history` array, loading prior turns from the DB and appending the new one after the pipeline finishes. The chat UI's light/dark theme is a single CSS token system (`:root` = dark, `:root[data-theme="light"]` overrides) toggled client-side via `localStorage["webscout-theme"]`; per-turn node status accumulates as visible trace chips instead of being overwritten.

## Testing conventions

The offline suite must stay network-free. Seams that make that possible:

- Node factories take an optional `model=` — tests pass `GenericFakeChatModel(messages=iter([AIMessage(...)]))`.
- `make_web_fetch(cfg, transport=...)` and `make_web_search(settings, transport=...)` take an `httpx.MockTransport`.
- `make_web_fetch(cfg, transport=..., resolve=...)` also takes a fake DNS resolver — this is what keeps the address-guard tests (private/loopback/link-local host rejection, redirect re-checks) network-free. Don't reach for monkeypatching `socket.getaddrinfo` instead; `resolve=` already exists for this.
- `make_web_search(settings, transport=..., usage=...)`, `build_research_agent(settings, usage=...)`, and `make_research_node(agent, settings, usage=...)` all take an injectable `UsageCollector` so tests can assert on drained tokens/cost/searches without a real HTTP round-trip.
- `build_graph` looks up `build_research_agent` as a module attribute, so tests `monkeypatch.setattr(g, "build_research_agent", ...)`.
- Anything needing a key or the live web must carry `@pytest.mark.integration`.
- `tests/test_server.py` starts with `pytest.importorskip("fastapi")` so it skips cleanly when the `web` group isn't installed — the default `uv sync`/CI never sees it.
- `web/server.py`'s `_db_path()` helper calls `store.init_db()` on every call (not once at import time), specifically so tests can redirect `CONVERSATIONS_DB_PATH` via `monkeypatch.setenv` + `get_settings.cache_clear()` before the first conversation route runs, instead of the real `data/webscout.db` being created as an import-time side effect.

## Docs

Design spec and plan live under [docs/superpowers/](docs/superpowers/); the notes files (spike findings, v3 experiment results) are written in Vietnamese and record why the model layer looks the way it does.
