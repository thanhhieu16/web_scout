# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```powershell
uv sync                                        # install (dev group = pytest, ruff)

uv run webscout "question"                     # one-shot run (full graph)
uv run webscout                                # interactive REPL
uv run webscout "q" --out report.md            # + markdown report

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
- `_get_request_payload` appends `server_tools` to the wire payload (used by `attach_server_tools`), instead of `extra_body={"tools": ...}` which would overwrite `bind_tools` output.

### Two independent search paths

1. **Server tool** — `build_search_spec` produces the `openrouter:web_search` spec, injected via `server_tools`. Call count lives at `usage.server_tool_use_details.web_search_requests` (docs claim `server_tool_use`; [count_web_searches](app/tools/search.py) reads both).
2. **Client-side `web_search` tool** ([app/tools/search_tool.py](app/tools/search_tool.py)) — a LangChain tool that makes its own OpenRouter `/chat/completions` request carrying the server-tool spec, then flattens the annotations into `[SRC] url | title` / `EXCERPT:` text blocks. **This is what the agent actually gets** ([build_research_agent](app/agent.py)); the researcher model tended to skip the raw server tool and discover pages via `web_fetch` alone ([v3 notes](docs/superpowers/notes/v3-experiments.md)).

### Sources are reconstructed, never returned

The agent returns only prose. [build_sources](app/nodes/parsing.py) rebuilds the ordered source list by scanning the whole message history in this precedence: annotation citations → `[SRC]` blocks parsed out of `web_search` tool messages → successful `web_fetch` results (search-engine result pages filtered by `_SEARCH_ENGINE_MARKERS`). That order defines the `[Sn]` numbering.

### The FINDINGS contract

The research prompt ends every reply with a `## FINDINGS` block whose lines are parsed by strict regex in [parse_findings_block](app/nodes/parsing.py) (`- [Sn] claim | confidence: high|medium|low`). `map_refs_to_urls` then resolves `Sn` to the nth URL from `build_sources`; unresolvable refs become `unresolved:Sn` and are surfaced as `weak_claims` by the research node. **`RESEARCH_SYSTEM_PROMPT`, `_LINE_RE`, and `map_refs_to_urls` must change together.**

### State accumulation runs through reducers

[ResearchState](app/state.py) is a `TypedDict` whose cumulative fields carry `Annotated[..., reducer]`: `findings` (`merge_findings`), `sources` (`merge_sources`), `weak_claims` (`merge_weak_claims`), and `iteration` / `search_calls` / `total_tokens` / `total_cost` (`operator.add`). **Nodes return only their own delta** — returning a running total would double-count it.

`merge_sources` dedupes by url with first-occurrence order, which is what keeps `[Sn]` numbering stable once a second research iteration runs. `merge_findings` dedupes by normalized claim text and takes the *lower* confidence on a merge: iteration 2 is instructed to research gaps only, so a repeated claim is not independent confirmation.

`gaps`, `sufficient`, `contradictory_claims`, and `answer` deliberately have no reducer — they are the current verdict, not a ledger.

### Config

pydantic-settings with source order **init kwargs > env > `.env` > `config.yaml`**. `.env` is auto-loaded at import time in [app/config.py](app/config.py). `get_settings()` is `lru_cache`d — tests construct `Settings(_env_file=None)` directly to avoid picking up the developer's real `.env`. Note `skills_enabled` defaults to `False` in code but `true` in `config.yaml`. Swapping models is a one-line `config.yaml` edit (`researcher` / `verifier` / `answer` roles).

Every LLM call goes through [call_with_backoff](app/backoff.py) — 5 attempts, linear 20s·n backoff on `OpenAIRateLimitError` only.

## Testing conventions

The offline suite must stay network-free. Seams that make that possible:

- Node factories take an optional `model=` — tests pass `GenericFakeChatModel(messages=iter([AIMessage(...)]))`.
- `make_web_fetch(cfg, transport=...)` and `make_web_search(settings, transport=...)` take an `httpx.MockTransport`.
- `build_graph` looks up `build_research_agent` as a module attribute, so tests `monkeypatch.setattr(g, "build_research_agent", ...)`.
- Anything needing a key or the live web must carry `@pytest.mark.integration`.

## Docs

Design spec and plan live under [docs/superpowers/](docs/superpowers/); the notes files (spike findings, v3 experiment results) are written in Vietnamese and record why the model layer looks the way it does.
