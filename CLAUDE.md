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
uv run python -m evals.summarize               # A/B arm table from LangSmith projects (pass prefixes to pick arms)
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

### One search path, with a scrape-based fallback

`build_search_spec` produces the `openrouter:web_search` spec, but nothing wires it onto `ResearchChatOpenAI`'s bind — the server-tool injection path (`server_tools` field on the model + `attach_server_tools`) was deleted as dead code, since it was unwired after the `audit-remediation` branch and the v3 experiments already showed the researcher model tended to skip the raw server tool and discover pages via `web_fetch` alone ([v3 notes](docs/superpowers/notes/v3-experiments.md)). `build_search_spec` survives only because the **client-side `web_search` tool** ([app/tools/search_tool.py](app/tools/search_tool.py)) calls it directly — a LangChain tool that makes its own OpenRouter `/chat/completions` request carrying that spec, then flattens the annotations into `[Sn] url | title` / `EXCERPT:` text blocks (IDs from the shared `SourceRegistry`, see below). **This is what the agent actually gets** ([build_research_agent](app/agent.py)). Call count for this path is tracked via `UsageCollector`, not `usage.server_tool_use_details.web_search_requests` (that field is only populated by the now-removed server-tool path; docs claim `server_tool_use`; [count_web_searches](app/tools/search.py) reads both for when a server-tool path returns).

When the OpenRouter call hard-fails (exception after `call_with_backoff` retries exhaust, or a 200 with zero usable annotations), `web_search` falls back to scraping `https://html.duckduckgo.com/html/` — the no-JS DuckDuckGo endpoint — via regex (`_duckduckgo_fallback`), not an HTML parser, since this path only fires on primary-search failure and doesn't justify a new dependency. It decodes DuckDuckGo's `uddg` redirect param back to the real URL and emits the same `[Sn] url | title` / `EXCERPT:` shape, so downstream parsing in `app/nodes/parsing.py` doesn't know which engine answered. A successful fallback counts as one search against `UsageCollector`. Tests inject `fallback_transport=` (an `httpx.MockTransport`) the same way `transport=` covers the primary call — any test exercising a primary failure path must also pass a `fallback_transport` that fails, or it silently reaches the real network.

### Search/fetch cache

[app/tools/cache.py](app/tools/cache.py)'s `TTLCache` (30 min TTL, 500-entry LRU cap, in-memory) sits behind two process-lifetime singletons, `search_cache` and `fetch_cache`, wired into `web_search`/`web_fetch` only by `build_research_agent` — the factories themselves default `cache=None` (off), so nothing outside `app/agent.py` opts in implicitly. Because the singletons live at module scope rather than inside `build_graph`/`build_research_agent` (both rebuilt per turn), a hit survives across research iterations *and* across turns in the same process — this is what makes it useful for a long-running `uvicorn` process, not just a single CLI run. Only successful results are cached (search header + unnumbered `(url, title, excerpt)` entries / extracted page text — never the `[Sn]` labels, since the cache outlives a turn's registry); `SEARCH_ERROR`/`FETCH_ERROR` responses are never cached, so a transient failure doesn't get stuck for the TTL window. A search cache hit does not call `usage.add` — no OpenRouter request happened, so it must not appear in the `searches`/cost METRICS.

### Sources are reconstructed, never returned

The agent returns only prose. [build_sources](app/nodes/parsing.py) rebuilds the ordered source list by scanning the whole message history in this precedence: annotation citations → `[Sn]` blocks parsed out of `web_search` tool messages → successful `web_fetch` results (search-engine result pages filtered by `_SEARCH_ENGINE_MARKERS`).

`[Sn]` IDs are **assigned by the tools, not counted by the model**. `build_research_agent` hands one [SourceRegistry](app/tools/sources.py) to both `web_search` and `web_fetch`; each prints `[Sn]` next to every URL it returns (`web_fetch` as a `[Sn] url` header line, parsed back by `_FETCH_HEADER_RE`), and a URL keeps its ID for the whole turn, across research iterations. `build_sources` returns `(sources, ref_map)` where `ref_map` (`{"S3": url}`) comes from those printed IDs, so it is independent of the source list order. Only a transcript where no tool printed an ID (annotation-only, or the legacy `[SRC]` label) falls back to positional numbering. This replaced an instruction to number sources "in the order you first used them", which never matched `build_sources`' order and silently mis-attributed citations.

### The FINDINGS contract

The research prompt ends every reply with a `## FINDINGS` block whose lines are parsed by strict regex in [parse_findings_block](app/nodes/parsing.py) (`- [Sn]... claim | confidence: high|medium|low | quote: "..."`; the quote part is optional to the regex). A line may carry several refs (`- [S1][S2] claim | ...`); `parse_findings_block` returns a `FindingsParse` NamedTuple whose `dropped` and `block_found` fields surface contract violations as `weak_claims`. `map_refs_to_urls` then resolves each `Sn` through `build_sources`' `ref_map`; unresolvable refs become `unresolved:Sn` and are surfaced as `weak_claims` by the research node.

[ground_findings](app/nodes/parsing.py) then checks each quote (`FindingsParse.quotes`) against the full text the agent saw for the finding's cited URLs (`collect_source_texts`: fetched bodies + search excerpts — not `build_sources`' 500-char excerpts), after word-level normalization (`_normalize_for_match`; `...` splits the quote into fragments that must all appear). Found → `grounded=True`. No quote or fewer than `_MIN_QUOTE_WORDS` words → `high` capped at `medium`. Quote given but in no cited source → `low` plus a weak claim. The research node then swaps each grounded source's excerpt for its quote (`_excerpts_from_quotes`), because the excerpt is all the answer writer and the `citation_support` judge see of a source. `render_findings` in the answer node prints refs as `[n]` source numbers, drops findings with no resolvable source, and lists `low` findings under a separate "Uncertain findings" heading. **`RESEARCH_SYSTEM_PROMPT`, `_LINE_RE` (incl. its `quote` group), `map_refs_to_urls`, and the `[Sn]` output shapes of `web_search`/`web_fetch` (with `_SEARCH_BLOCK_RE`/`_FETCH_HEADER_RE`) must change together.**

### State accumulation runs through reducers

[ResearchState](app/state.py) is a `TypedDict` whose cumulative fields carry `Annotated[..., reducer]`: `findings` (`merge_findings`), `sources` (`merge_sources`), `weak_claims` (`merge_weak_claims`), and `iteration` / `search_calls` / `total_tokens` / `total_cost` (`operator.add`). **Nodes return only their own delta** — returning a running total would double-count it.

`merge_sources` dedupes by url with first-occurrence order, which is what keeps `[Sn]` numbering stable once a second research iteration runs. `merge_findings` dedupes by normalized claim text and takes the *lower* confidence on a merge: iteration 2 is instructed to research gaps only, so a repeated claim is not independent confirmation. It does adopt a later grounded quote when the existing finding had none.

`gaps`, `sufficient`, `contradictory_claims`, and `answer` deliberately have no reducer — they are the current verdict, not a ledger.

### Config

pydantic-settings with source order **init kwargs > env > `.env` > `config.yaml`**. `.env` is auto-loaded at import time in [app/config.py](app/config.py). `get_settings()` is `lru_cache`d — tests construct `Settings(_env_file=None)` directly to avoid picking up the developer's real `.env`. Note `skills_enabled` defaults to `False` in code but `true` in `config.yaml`. When on, the agent's deepagents file tools see a `CompositeBackend` ([_skills_backend](app/agent.py)): `/skills/` routes to a `FilesystemBackend` rooted at `skills/` with writes denied by a `FilesystemPermission`, everything else goes to an in-memory `StateBackend`. Never root the backend at the repo — the agent reads untrusted web pages and would then be able to read `.env` or rewrite code. Swapping models is a one-line `config.yaml` edit (`researcher` / `verifier` / `answer` / `judge` roles).

Every LLM call goes through [call_with_backoff](app/backoff.py). The researcher is the exception in *where*: its calls happen inside the deep agent loop, so `build_research_agent` sets `ResearchChatOpenAI.rate_limit_attempts`, which wraps each `_generate` call — the research node must **not** wrap `agent.invoke` in it, because that would replay every search and fetch of the run on a single 429. Its default is 5 attempts, linear 20s·n backoff, retrying only `OpenAIRateLimitError` — but `retry_on` accepts either a tuple of exception types or a predicate, and callers with a different failure profile pass their own: `web_search` ([app/tools/search_tool.py](app/tools/search_tool.py)) uses `attempts=3, base_delay=2.0` with an HTTP-status predicate, so a hung search fails fast instead of blocking on the LLM-call defaults.

### Web chat UI

`web/server.py` (FastAPI) streams `app/main.py`'s `stream_pipeline` generator as Server-Sent Events for `POST /api/chat`; the same generator also backs the CLI's `run_pipeline`, so there is one implementation of "drive the graph, report progress," not two. Conversation-aware follow-ups go through `app/conversation.py::condense_question` before reaching the graph. The `web` dependency group (`fastapi`, `uvicorn`) is separate from `dev` so it never affects the default `uv sync` or CI. A request's `model` field becomes a per-request copy of the settings (`app.config.with_model`) threaded through `run_chat_turn(..., settings=)` into `condense_question` and `build_graph` — never `override_model`, which mutates the cached Settings every concurrent request reads (that stays CLI-only).

Conversations persist server-side in `data/webscout.db` via `web/store.py` (stdlib `sqlite3`, no new dependency) — `POST /api/chat` now takes a `conversation_id` instead of a client-supplied `history` array, loading prior turns from the DB and appending the new one after the pipeline finishes. The chat UI's light/dark theme is a single CSS token system (`:root` = dark, `:root[data-theme="light"]` overrides) toggled client-side via `localStorage["webscout-theme"]`; per-turn node status accumulates as visible trace chips instead of being overwritten.

## Testing conventions

The offline suite must stay network-free. Seams that make that possible:

- Node factories take an optional `model=` — tests pass `GenericFakeChatModel(messages=iter([AIMessage(...)]))`.
- `make_web_fetch(cfg, transport=...)` and `make_web_search(settings, transport=..., fallback_transport=...)` take an `httpx.MockTransport` — `fallback_transport` covers `web_search`'s DuckDuckGo scrape path, exercised only when `transport`'s response is a hard failure.
- `make_web_fetch(cfg, transport=..., resolve=...)` also takes a fake DNS resolver — this is what keeps the address-guard tests (private/loopback/link-local host rejection, redirect re-checks) network-free. Don't reach for monkeypatching `socket.getaddrinfo` instead; `resolve=` already exists for this.
- `make_web_search(settings, transport=..., usage=...)`, `build_research_agent(settings, usage=...)`, and `make_research_node(agent, settings, usage=...)` all take an injectable `UsageCollector` so tests can assert on drained tokens/cost/searches without a real HTTP round-trip.
- `build_graph` looks up `build_research_agent` as a module attribute, so tests `monkeypatch.setattr(g, "build_research_agent", ...)`.
- `make_web_search(..., registry=...)` and `make_web_fetch(..., registry=...)` take an `app.tools.sources.SourceRegistry`; each defaults to its own, so pass one shared instance when a test needs search and fetch IDs to agree (see `tests/test_source_ids.py`). `build_research_agent(settings, registry=...)` accepts one too.
- Tests that monkeypatch `turn.build_graph` must accept a `settings` argument (`lambda settings=None: FakeGraph()`), since `run_chat_turn` passes it through.
- `make_web_search(..., cache=...)` and `make_web_fetch(..., cache=...)` take an `app.tools.cache.TTLCache` and default to `None` (caching off) — tests that need to assert on cache behavior pass a fresh `TTLCache()` instance; everything else is unaffected since the default keeps the old no-cache behavior.
- Anything needing a key or the live web must carry `@pytest.mark.integration`.
- `tests/test_server.py` starts with `pytest.importorskip("fastapi")` so it skips cleanly when the `web` group isn't installed — the default `uv sync`/CI never sees it.
- `web/server.py`'s `_db_path()` helper calls `store.init_db()` on every call (not once at import time), specifically so tests can redirect `CONVERSATIONS_DB_PATH` via `monkeypatch.setenv` + `get_settings.cache_clear()` before the first conversation route runs, instead of the real `data/webscout.db` being created as an import-time side effect.

## Docs

Design spec and plan live under [docs/superpowers/](docs/superpowers/); the notes files (spike findings, v3 experiment results) are written in Vietnamese and record why the model layer looks the way it does.
