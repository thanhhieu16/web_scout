# WebScout Audit Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all 16 findings from the 2026-08-26 codebase audit — evidence loss across research iterations, false cost/token metrics, non-portable model and path assumptions, an unguarded fetch tool, and repo hygiene.

**Architecture:** `ResearchState` gains LangGraph reducers so nodes return only their own delta and accumulation stops being hand-written. The `web_search` tool's out-of-band OpenRouter call reports its spend through an injected `UsageCollector` instead of a parsed text header. `web_fetch` gains an SSRF guard with an injectable resolver so the offline suite stays network-free.

**Tech Stack:** Python 3.12+, uv, LangGraph, Deep Agents, langchain-openai 1.6, httpx, trafilatura, pydantic-settings, pytest, ruff, LangSmith.

**Spec:** [docs/superpowers/specs/2026-08-26-webscout-audit-remediation-design.md](../specs/2026-08-26-webscout-audit-remediation-design.md)

## Global Constraints

- Python `>=3.12`. Venv is 3.13 locally; CI must cover 3.12 and 3.13 on ubuntu-latest and windows-latest.
- The offline suite (`-m "not integration"`) must never touch the network. Anything needing a key or the live web carries `@pytest.mark.integration`.
- Every LLM call goes through `call_with_backoff`.
- Shell is PowerShell on Windows. Set `$env:PYTHONIOENCODING="utf-8"` if printing answers raises `cp65001`.
- `ruff check .` must pass with `select = ["E", "F", "I", "UP", "B"]`, `line-length = 100`.
- Breaking changes to `RESEARCH_SYSTEM_PROMPT`, `_LINE_RE`, `map_refs_to_urls`, and `ResearchState` are explicitly approved. The old eval baseline is discarded.
- Work happens on branch `audit-remediation` (already created, spec committed at `0d4110e`).
- Commit after every task.

## File Structure

**Created:**
- `app/usage.py` — `UsageCollector`: thread-safe accumulator for usage from HTTP calls outside LangChain's message stream.
- `tests/test_usage.py` — collector unit tests.
- `tests/test_state_reducers.py` — reducer unit tests.
- `tests/test_graph_iterations.py` — graph-level regression guard for evidence accumulation.
- `tests/test_fetch_guard.py` — SSRF guard tests.

**Modified:**
- `pyproject.toml` — build-system, dependency-groups, ruff config.
- `.github/workflows/test.yml` — OS/Python matrix, ruff step.
- `app/state.py` — reducers.
- `app/nodes/parsing.py` — `FindingsParse`, multi-ref regex, `find_unknown_refs`, drop the text-header search regex.
- `app/nodes/research.py`, `app/nodes/verify.py`, `app/nodes/answer.py` — return deltas only.
- `app/agent.py` — repo-root paths, `usage=` passthrough, prompt update.
- `app/models.py` — no empty `tools` key, `usage: {include: true}`, `judge` role.
- `app/config.py` — repo-root `config.yaml` fallback, fetch guard settings, `judge` role.
- `app/graph.py` — build and thread the `UsageCollector`.
- `app/tools/fetch.py` — SSRF guard, manual redirect loop.
- `app/tools/search_tool.py` — usage reporting, retry.
- `app/tools/search.py` — drop `attach_server_tools`.
- `app/backoff.py` — generalized `retry_on`.
- `app/main.py` — new parse contract, own collector, single rounding point.
- `evals/evaluators.py`, `config.yaml` — judge role, full citation coverage.
- `README.md`, `CLAUDE.md` — command and contract updates.

---

### Task 1: Packaging, dev group, ruff, CI matrix

Spec §8.1–§8.4. No behavior change. Goes first because `uv run pytest` as currently documented fails when dev extras are not synced — it silently falls back to the system pytest and produces 17 `ImportError`s.

**Files:**
- Modify: `pyproject.toml`
- Modify: `.github/workflows/test.yml`
- Modify: `app/backoff.py:11`, `app/main.py:93`, `app/main.py:130`, `app/nodes/answer.py:1`, `app/nodes/parsing.py:46`, `app/nodes/research.py:2`, `app/nodes/verify.py:4`, `app/nodes/verify.py:28`, `evals/summarize.py:5`, `evals/summarize.py:9`
- Modify: `tests/test_backoff.py:10`, `tests/test_backoff.py:52`, `tests/test_cli.py:124`, `tests/test_node_research.py:31`, `tests/test_node_research.py:38`, `tests/test_node_verify.py:37`, `tests/test_search_tool.py:75`, `tests/test_search_tool.py:87`, `tests/test_skill.py:19`
- Modify: `README.md`, `CLAUDE.md`

**Interfaces:**
- Consumes: nothing.
- Produces: `uv sync` installs pytest and ruff without extras flags; `uv run pytest`, `uv run ruff check .`, and `uv run webscout --help` all work.

- [ ] **Step 1: Rewrite `pyproject.toml`**

```toml
[project]
name = "webscout"
version = "0.1.0"
description = "Mini deep-research agent"
requires-python = ">=3.12"
dependencies = [
    "deepagents>=0.6",
    "langchain-openai>=1.0",
    "langgraph>=1.0",
    "trafilatura>=2.0",
    "httpx>=0.28",
    "pydantic-settings[yaml]>=2.6",
    "langsmith>=0.3",
    "python-dotenv>=1.0",
]

[project.scripts]
webscout = "app.main:main"

[dependency-groups]
dev = ["pytest>=8.0", "ruff>=0.9"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["app"]

[tool.pytest.ini_options]
markers = [
    "integration: requires OPENROUTER_API_KEY and network",
]

[tool.ruff]
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
```

`[project.optional-dependencies]` is deleted on purpose — two sources of truth for dev deps is what caused the original confusion.

- [ ] **Step 2: Sync and confirm the toolchain resolves**

Run:
```powershell
uv sync
uv run pytest -m "not integration" -q
uv run webscout --help
```
Expected: sync installs pytest and ruff; 88 tests pass; `webscout --help` prints the argparse usage line including `--out`.

- [ ] **Step 3: Auto-fix the mechanical ruff violations**

Run: `uv run ruff check --fix .`
Expected: fixes `UP035` in `app/nodes/answer.py`, `app/nodes/research.py`, `app/nodes/verify.py` (`from collections.abc import Callable`) and `F541` in `app/main.py:93`. Reports 14 remaining.

- [ ] **Step 4: Fix the 14 remaining violations by hand**

`app/backoff.py:11` — drop the unused binding:

```python
        except OpenAIRateLimitError:
```

`app/nodes/parsing.py:46` — the two lists are built together in `parse_findings_block` and are always the same length, so assert it:

```python
    for finding, finding_refs in zip(findings, refs, strict=True):
```

`app/main.py:130` — wrap the argparse line:

```python
    parser.add_argument(
        "--out",
        default=None,
        help="write a markdown report to this path (one-shot mode)",
    )
```

`app/nodes/verify.py:28` — split the JSON shape line in the prompt:

```python
Reply with JSON only, exactly this shape:
{"sufficient": bool, "missing_information": [str], "weak_claims": [str],
 "contradictory_claims": [str]}"""
```

`evals/summarize.py:5` and `:9` — wrap:

```python
KEYS = (
    "correctness", "citation_support", "latency_s",
    "total_tokens", "search_calls", "num_sources",
)
ARMS = ["skill-on-iters3", "skill-off-iters3", "skill-on-iters1", "skill-off-iters1"]
print(
    f"{'arm':22s} {'n':>2} {'corr':>5} {'cite':>5} "
    f"{'lat_s':>7} {'tok':>7} {'srch':>5} {'src':>4}"
)
```

`tests/test_backoff.py:10` — wrap the SimpleNamespace:

```python
    request = SimpleNamespace(
        method="POST",
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={},
    )
```

`tests/test_backoff.py:52` — delete the unused `fn = SimpleNamespace()` line entirely; the test only needs the `call_with_backoff` assertion below it.

`tests/test_cli.py:124` — reformat `DELTAS` across lines:

```python
DELTAS = [
    {
        "research": {
            "findings": [{"claim": "c", "source_urls": [], "confidence": "high"}],
            "sources": [
                {
                    "url": "https://s",
                    "title": "S",
                    "source_type": "secondary",
                    "excerpt": "",
                }
            ],
            "iteration": 1,
            "search_calls": 3,
            "weak_claims": [],
        }
    },
    {"verify": {"sufficient": True, "gaps": [], "weak_claims": [], "contradictory_claims": []}},
    {"answer": {"answer": "Final [1]."}},
]
```

`tests/test_node_research.py:31` and `:38` — delete both unused `s = Settings(_env_file=None)` lines; those two tests only exercise `build_research_input`, which takes no settings.

`tests/test_node_verify.py:37` — wrap the payload:

```python
    payload = (
        '{"sufficient": true, "missing_information": [], '
        '"weak_claims": [], "contradictory_claims": []}'
    )
```

`tests/test_search_tool.py:75` and `:87` — wrap the tool-call dicts:

```python
        AIMessage(
            content="",
            tool_calls=[{"name": "web_search", "args": {"query": "q"}, "id": "s1"}],
        ),
```
```python
        AIMessage(
            content="",
            tool_calls=[{"name": "web_fetch", "args": {"url": "https://z"}, "id": "f1"}],
        ),
```

`tests/test_skill.py:19` — wrap the heading tuple:

```python
    for heading in (
        "Source priority",
        "Verify important claims",
        "Handle conflicts",
        "Citation integrity",
    ):
```

- [ ] **Step 5: Verify lint and tests are both clean**

Run:
```powershell
uv run ruff check .
uv run pytest -m "not integration" -q
```
Expected: `All checks passed!` and 88 passed.

- [ ] **Step 6: Rewrite the CI workflow**

`.github/workflows/test.yml`:

```yaml
name: tests

on:
  push:
    branches: [master]
  pull_request:

jobs:
  pytest:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest]
        python-version: ["3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Sync dependencies
        run: uv sync

      - name: Lint
        run: uv run ruff check .

      - name: Run offline test suite
        run: uv run pytest -m "not integration" -q
```

- [ ] **Step 7: Update the docs that carried the workaround**

In `README.md`: delete the parenthetical in the Usage section that reads "recommended entry point — the repo declares no build-system yet, so the `webscout` console script in `pyproject.toml` is never installed", and replace the three usage commands' `uv run python -m app.main` with `uv run webscout`. Delete the `webscout: command not found` row from the Troubleshooting table.

In `CLAUDE.md`: replace the paragraph beginning "`pyproject.toml` declares a `webscout` console script but there is **no `[build-system]`**" with:

```markdown
`uv sync` installs the `dev` dependency group (pytest, ruff) by default — no `--extra` flag. The `webscout` console script is installed by the hatchling build backend. On Windows consoles, set `$env:PYTHONIOENCODING="utf-8"` if printing the answer raises `cp65001` errors.
```

And in the Commands block replace `uv sync --extra dev` with `uv sync`, and the three `uv run python -m app.main` lines with `uv run webscout`.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .github/workflows/test.yml app tests evals README.md CLAUDE.md
git commit -m "chore: hatchling build, PEP 735 dev group, ruff, CI matrix"
```

---

### Task 2: Portability — empty tools array and CWD-relative paths

Spec §6. Two independent one-liners plus the `config.yaml` lookup they expose.

**Files:**
- Modify: `app/models.py:13-18`
- Modify: `app/agent.py:1,42-56`
- Modify: `app/config.py:37-42`
- Test: `tests/test_payload_merge.py`, `tests/test_agent.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `app.agent._REPO_ROOT: Path` — the repo root resolved from `__file__`, reused by Task 10's docs and by `app/config.py`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_payload_merge.py`:

```python
def test_payload_omits_tools_when_none_present():
    m = ResearchChatOpenAI(
        model="stealth/ox-alpha",
        api_key="not-set",
        base_url="https://openrouter.ai/api/v1",
    )
    payload = m._get_request_payload([HumanMessage("hi")])
    assert "tools" not in payload
```

Append to `tests/test_agent.py`:

```python
def test_skills_resolve_from_repo_root_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    s.skills_enabled = True
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return object()

    import app.agent as a

    monkeypatch.setattr(a, "_create_deep_agent", fake_create)
    a.build_research_agent(s)
    assert captured["skills"] == ["skills/"]
    assert (Path(captured["backend"].root_dir) / "skills").is_dir()
```

Append to `tests/test_config.py`:

```python
def test_config_yaml_falls_back_to_repo_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    # repo config.yaml sets skills_enabled: true; the code default is False
    assert s.skills_enabled is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_payload_merge.py tests/test_agent.py tests/test_config.py -q`
Expected: FAIL — `assert "tools" not in payload` (payload has `tools: []`), `AttributeError: module 'app.agent' has no attribute '_create_deep_agent'`, and `assert False is True` for the config fallback.

- [ ] **Step 3: Drop the empty tools key**

Replace `app/models.py:13-18` with:

```python
    def _get_request_payload(self, *args, **kwargs):
        payload = super()._get_request_payload(*args, **kwargs)
        tools = list(payload.get("tools") or []) + list(self.server_tools)
        if tools:
            payload["tools"] = tools
        else:
            payload.pop("tools", None)
        return payload
```

- [ ] **Step 4: Resolve skills from the repo root**

Rewrite `app/agent.py`'s imports and `build_research_agent`. Hoist the deepagents import to module level under a private alias so tests can patch it:

```python
from pathlib import Path

from deepagents import create_deep_agent as _create_deep_agent

from app.config import Settings, get_settings
from app.models import get_model
from app.tools.fetch import make_web_fetch
from app.tools.search_tool import make_web_search

_REPO_ROOT = Path(__file__).resolve().parent.parent
```

```python
def build_research_agent(settings: Settings | None = None):
    s = settings or get_settings()
    model = get_model("researcher", s)
    kwargs = dict(
        model=model,
        tools=[make_web_search(s), make_web_fetch(s.fetch)],
        system_prompt=RESEARCH_SYSTEM_PROMPT,
    )
    if s.skills_enabled and (_REPO_ROOT / "skills").is_dir():
        from deepagents.backends.filesystem import FilesystemBackend

        kwargs["backend"] = FilesystemBackend(root_dir=str(_REPO_ROOT))
        kwargs["skills"] = ["skills/"]
    return _create_deep_agent(**kwargs)
```

- [ ] **Step 5: Fall back to the repo-root `config.yaml`**

Replace the `model_config` block in `app/config.py:37-42` with:

```python
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _yaml_path() -> str:
    """Prefer ./config.yaml so a local override wins; fall back to the repo's own."""
    local = Path("config.yaml")
    return str(local if local.is_file() else _REPO_ROOT / "config.yaml")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        yaml_file=_yaml_path(),
        extra="ignore",
    )
```

Add `from pathlib import Path` to the imports at the top of `app/config.py`.

`_yaml_path()` is evaluated at class-definition time, so a process that `chdir`s after import keeps the path it resolved at import. That is correct for the CLI (one process, one CWD) and for `test_config.py`, which runs `Settings(...)` fresh under `monkeypatch.chdir`. If a future test needs re-evaluation it must pass `_yaml_file=` explicitly.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest -m "not integration" -q`
Expected: 91 passed.

Note: `test_config.py::test_defaults_when_no_files` asserts code defaults under `chdir(tmp_path)`. It only asserts `researcher.model` and `search`, both of which match the repo `config.yaml`, so the fallback does not break it.

- [ ] **Step 7: Commit**

```bash
git add app/models.py app/agent.py app/config.py tests/
git commit -m "fix: omit empty tools array, resolve skills and config.yaml from repo root"
```

---

### Task 3: Reducers on `ResearchState`

Spec §4.2–§4.3. The core fix for finding #1.

**Files:**
- Modify: `app/state.py` (full rewrite)
- Modify: `app/nodes/research.py:68-79`
- Modify: `app/nodes/verify.py:73-95`
- Modify: `app/nodes/answer.py:51-56`
- Modify: `app/main.py:60-70`
- Test: `tests/test_state_reducers.py` (create), `tests/test_node_verify.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `app.state.merge_findings(prior: list, new: list) -> list`
  - `app.state.merge_sources(prior: list, new: list) -> list`
  - `app.state.merge_weak_claims(prior: list, new: list) -> list`
  - Node callables now return **deltas only**: `research` returns `iteration=1` per run, `search_calls`/`total_tokens`/`total_cost` for that run alone, and `findings`/`sources`/`weak_claims` from that run alone.

- [ ] **Step 1: Write the failing reducer tests**

Create `tests/test_state_reducers.py`:

```python
from app.state import merge_findings, merge_sources, merge_weak_claims


def _src(url, title="T"):
    return {"url": url, "title": title, "source_type": "secondary", "excerpt": ""}


def _find(claim, urls, conf):
    return {"claim": claim, "source_urls": list(urls), "confidence": conf}


def test_merge_sources_dedupes_by_url_and_keeps_first_order():
    prior = [_src("https://a.dev", "A"), _src("https://b.dev", "B")]
    new = [_src("https://b.dev", "B again"), _src("https://c.dev", "C")]
    merged = merge_sources(prior, new)
    assert [s["url"] for s in merged] == [
        "https://a.dev",
        "https://b.dev",
        "https://c.dev",
    ]
    assert merged[1]["title"] == "B"


def test_merge_findings_unions_urls_and_keeps_lower_confidence():
    prior = [_find("Claim one", ["https://a.dev"], "high")]
    new = [_find("  claim ONE  ", ["https://b.dev"], "low")]
    merged = merge_findings(prior, new)
    assert len(merged) == 1
    assert merged[0]["claim"] == "Claim one"
    assert merged[0]["source_urls"] == ["https://a.dev", "https://b.dev"]
    assert merged[0]["confidence"] == "low"


def test_merge_findings_appends_distinct_claims():
    merged = merge_findings(
        [_find("one", ["https://a"], "high")],
        [_find("two", ["https://b"], "medium")],
    )
    assert [f["claim"] for f in merged] == ["one", "two"]


def test_merge_weak_claims_dedupes_preserving_order():
    assert merge_weak_claims(["a", "b"], ["b", "c"]) == ["a", "b", "c"]


def test_reducers_do_not_mutate_prior():
    prior = [_src("https://a.dev")]
    merge_sources(prior, [_src("https://b.dev")])
    assert len(prior) == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_state_reducers.py -q`
Expected: FAIL with `ImportError: cannot import name 'merge_findings' from 'app.state'`.

- [ ] **Step 3: Rewrite `app/state.py`**

```python
import operator
from typing import Annotated, TypedDict

_CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


def merge_sources(prior: list, new: list) -> list:
    """Dedupe by url, first occurrence wins — this is what keeps [Sn] numbering
    stable once a second research iteration runs."""
    out = list(prior)
    seen = {s.get("url") for s in out}
    for source in new:
        url = source.get("url")
        if url and url not in seen:
            seen.add(url)
            out.append(source)
    return out


def _claim_key(finding: dict) -> str:
    return " ".join(str(finding.get("claim", "")).split()).casefold()


def merge_findings(prior: list, new: list) -> list:
    """Dedupe by normalized claim text. A repeat is not independent confirmation —
    iteration 2 is instructed to research gaps only — so confidence never rises on
    a merge; it takes the more pessimistic of the two."""
    out = [dict(f) for f in prior]
    index = {_claim_key(f): i for i, f in enumerate(out)}
    for finding in new:
        key = _claim_key(finding)
        if key not in index:
            index[key] = len(out)
            out.append(dict(finding))
            continue
        existing = out[index[key]]
        urls = list(existing.get("source_urls") or [])
        for url in finding.get("source_urls") or []:
            if url not in urls:
                urls.append(url)
        existing["source_urls"] = urls
        old = _CONFIDENCE_ORDER.get(existing.get("confidence", "low"), 0)
        incoming = _CONFIDENCE_ORDER.get(finding.get("confidence", "low"), 0)
        if incoming < old:
            existing["confidence"] = finding["confidence"]
    return out


def merge_weak_claims(prior: list, new: list) -> list:
    return list(dict.fromkeys(list(prior) + list(new)))


class ResearchState(TypedDict, total=False):
    question: str
    max_iterations: int
    answer_language: str

    findings: Annotated[list, merge_findings]
    sources: Annotated[list, merge_sources]
    weak_claims: Annotated[list, merge_weak_claims]

    iteration: Annotated[int, operator.add]
    search_calls: Annotated[int, operator.add]
    total_tokens: Annotated[int, operator.add]
    total_cost: Annotated[float, operator.add]

    # Current verdict, not a ledger — last write wins on purpose.
    gaps: list
    contradictory_claims: list
    sufficient: bool
    answer: str
```

- [ ] **Step 4: Run to verify the reducer tests pass**

Run: `uv run pytest tests/test_state_reducers.py -q`
Expected: 5 passed.

- [ ] **Step 5: Strip hand-accumulation from the research node**

Replace `app/nodes/research.py:68-79` with:

```python
        tokens, cost = sum_usage(messages)
        return {
            "findings": findings,
            "sources": sources,
            "weak_claims": weak,
            "iteration": 1,
            "search_calls": count_total_searches(messages),
            "total_tokens": tokens,
            "total_cost": cost,
        }
```

Also delete the now-unused `prior_weak = state.get("weak_claims") or []` line above it.

- [ ] **Step 6: Strip hand-accumulation from the verify node**

Replace `app/nodes/verify.py:73-95` with:

```python
        tokens, cost = sum_usage([reply])
        usage_delta = {"total_tokens": tokens, "total_cost": cost}
        if result is None:
            return {
                "sufficient": False,
                "gaps": ["verifier parse error"],
                "contradictory_claims": [],
                **usage_delta,
            }
        return {
            "sufficient": result.sufficient,
            "gaps": result.missing_information,
            "weak_claims": result.weak_claims,
            "contradictory_claims": result.contradictory_claims,
            **usage_delta,
        }
```

The parse-error branch no longer echoes `weak_claims` at all — echoing a reducer-managed field would be a no-op at best.

- [ ] **Step 7: Strip hand-accumulation from the answer node**

Replace `app/nodes/answer.py:51-56` with:

```python
        tokens, cost = sum_usage([reply])
        return {
            "answer": str(reply.content).strip(),
            "total_tokens": tokens,
            "total_cost": cost,
        }
```

- [ ] **Step 8: Round the cost once, at the edge**

In `app/main.py`, `run_pipeline` already does `round(final.get("total_cost", 0.0), 4)` — leave it. In `write_report` and `_print_result` the `:.4f` formats already round. No change needed here; the per-node `round(..., 6)` calls were removed in steps 5–7, which is the whole point.

- [ ] **Step 9: Update the verify tests to the delta contract**

In `tests/test_node_verify.py`, replace `test_verify_merges_result` and `test_verify_degrades_after_failed_retry` with:

```python
def test_verify_returns_own_weak_claims_only():
    payload = (
        '{"sufficient": false, "missing_information": ["replay"], '
        '"weak_claims": ["new weak"], "contradictory_claims": []}'
    )
    node = make_verify_node(Settings(_env_file=None), model=_fake_model(payload))  # type: ignore[call-arg]
    delta = node(dict(STATE))
    assert delta["sufficient"] is False
    assert delta["gaps"] == ["replay"]
    assert delta["weak_claims"] == ["new weak"]
    assert "old weak" not in delta["weak_claims"]


def test_verify_degrades_after_failed_retry(capsys):
    model = GenericFakeChatModel(
        messages=iter([AIMessage(content="junk"), AIMessage(content="still junk")])
    )
    node = make_verify_node(Settings(_env_file=None), model=model)  # type: ignore[call-arg]
    delta = node(dict(STATE))
    assert delta == {
        "sufficient": False,
        "gaps": ["verifier parse error"],
        "contradictory_claims": [],
        "total_tokens": 0,
        "total_cost": 0.0,
    }
    assert "[warn]" in capsys.readouterr().err
```

- [ ] **Step 10: Run the full offline suite**

Run: `uv run pytest -m "not integration" -q`
Expected: 96 passed.

- [ ] **Step 11: Rewrite the state section of `CLAUDE.md`**

Replace the whole "### State accumulation is manual" section with:

```markdown
### State accumulation runs through reducers

[ResearchState](app/state.py) is a `TypedDict` whose cumulative fields carry `Annotated[..., reducer]`: `findings` (`merge_findings`), `sources` (`merge_sources`), `weak_claims` (`merge_weak_claims`), and `iteration` / `search_calls` / `total_tokens` / `total_cost` (`operator.add`). **Nodes return only their own delta** — returning a running total would double-count it.

`merge_sources` dedupes by url with first-occurrence order, which is what keeps `[Sn]` numbering stable once a second research iteration runs. `merge_findings` dedupes by normalized claim text and takes the *lower* confidence on a merge: iteration 2 is instructed to research gaps only, so a repeated claim is not independent confirmation.

`gaps`, `sufficient`, `contradictory_claims`, and `answer` deliberately have no reducer — they are the current verdict, not a ledger.
```

- [ ] **Step 12: Commit**

```bash
git add app/state.py app/nodes tests/test_state_reducers.py tests/test_node_verify.py CLAUDE.md
git commit -m "fix: accumulate research state with LangGraph reducers"
```

---

### Task 4: Parse contract — surface dropped lines and a missing block

Spec §4.4. Findings #6.

**Files:**
- Modify: `app/nodes/parsing.py:15-37`
- Modify: `app/nodes/research.py:51`
- Modify: `app/main.py:36`
- Test: `tests/test_parsing.py`, `tests/test_node_research.py`

**Interfaces:**
- Consumes: `merge_weak_claims` from Task 3 (dedupes what the research node emits).
- Produces: `app.nodes.parsing.FindingsParse(findings, refs, narrative, dropped, block_found)` — a `NamedTuple`. `parse_findings_block(text) -> FindingsParse`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_parsing.py`:

```python
def test_parse_reports_dropped_lines():
    text = (
        "Body.\n\n## FINDINGS\n"
        "- [S1] good claim | confidence: high\n"
        "- [S2] bad claim | confidence: very-high\n"
        "just some prose\n"
    )
    parsed = parse_findings_block(text)
    assert len(parsed.findings) == 1
    assert parsed.dropped == [
        "- [S2] bad claim | confidence: very-high",
        "just some prose",
    ]
    assert parsed.block_found is True


def test_parse_reports_missing_block():
    parsed = parse_findings_block("No contract here at all.")
    assert parsed.block_found is False
    assert parsed.findings == []
    assert parsed.dropped == []
    assert parsed.narrative == "No contract here at all."
```

Append to `tests/test_node_research.py`:

```python
def test_missing_findings_block_becomes_weak_claim(capsys):
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    agent = FakeAgent(_msg("Prose only, no contract.", []))
    delta = make_research_node(agent, s)({"question": "Q?", "iteration": 0})
    assert any("no ## FINDINGS block" in w for w in delta["weak_claims"])
    assert "[warn]" in capsys.readouterr().err


def test_unparseable_findings_line_becomes_weak_claim(capsys):
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    bad = "Body.\n\n## FINDINGS\n- [S1] claim | confidence: certainly\n"
    agent = FakeAgent(_msg(bad, [{"url": "https://y.dev", "title": "Y", "content": ""}]))
    delta = make_research_node(agent, s)({"question": "Q?", "iteration": 0})
    assert any("unparseable FINDINGS line" in w for w in delta["weak_claims"])
    assert "[warn]" in capsys.readouterr().err
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_parsing.py tests/test_node_research.py -q`
Expected: FAIL with `AttributeError: 'tuple' object has no attribute 'dropped'` and `KeyError`/assertion failures on the two research-node tests.

- [ ] **Step 3: Return a `FindingsParse`**

In `app/nodes/parsing.py`, add `NamedTuple` to the typing import and replace lines 15-37:

```python
class FindingsParse(NamedTuple):
    findings: list[Finding]
    refs: list[list[str]]
    narrative: str
    dropped: list[str]
    block_found: bool


def parse_findings_block(text: str) -> FindingsParse:
    match = _BLOCK_RE.search(text)
    if not match:
        return FindingsParse([], [], text, [], False)
    narrative = text[: match.start()].rstrip()
    findings: list[Finding] = []
    refs: list[list[str]] = []
    dropped: list[str] = []
    for raw in text[match.end() :].splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _LINE_RE.match(line)
        if not m:
            dropped.append(line)
            continue
        findings.append(
            Finding(
                claim=m.group("claim").strip(),
                source_urls=[],
                confidence=m.group("conf").lower(),
            )
        )
        refs.append([m.group("ref")])
    return FindingsParse(findings, refs, narrative, dropped, True)
```

- [ ] **Step 4: Consume the new signals in the research node**

In `app/nodes/research.py`, replace line 51 and extend the weak-claim assembly. The `research` function body from the parse onward becomes:

```python
        messages = result["messages"]
        message = messages[-1]
        parsed = parse_findings_block(str(message.content))
        sources, ref_order = build_sources(messages)
        findings, _ = map_refs_to_urls(parsed.findings, parsed.refs, ref_order)
        _, unknown = reconcile_sources(findings, ref_order)
        unknown_set = set(unknown)
        seen_weak: set[str] = set()
        weak: list[str] = []

        def _note(item: str) -> None:
            if item not in seen_weak:
                seen_weak.add(item)
                weak.append(item)

        if not parsed.block_found:
            _note("research reply had no ## FINDINGS block")
            print(
                "[warn] research reply had no ## FINDINGS block",
                file=sys.stderr,
                flush=True,
            )
        for line in parsed.dropped:
            _note(f"unparseable FINDINGS line: {line}")
            print(f"[warn] unparseable FINDINGS line: {line}", file=sys.stderr, flush=True)
        for finding in findings:
            for u in finding.get("source_urls", []):
                if u in unknown_set and u.startswith("unresolved:"):
                    _note(
                        f"claim '{finding.get('claim', '')}' references "
                        f"unknown citation: {u.replace('unresolved:', '[ref] ')}"
                    )
```

Add `import sys` to the top of `app/nodes/research.py`.

- [ ] **Step 5: Update the CLI call site**

In `app/main.py`, replace line 36 (`findings, refs, narrative = parse_findings_block(...)`) with:

```python
    parsed = parse_findings_block(str(message.content))
```

and update the three following lines that used the unpacked names:

```python
    sources, ref_order = build_sources(messages)
    findings, _ = map_refs_to_urls(parsed.findings, parsed.refs, ref_order)
    _, unknown = reconcile_sources(findings, ref_order)
```

and the `"answer"` entry of the returned dict:

```python
        "answer": parsed.narrative.strip(),
```

- [ ] **Step 6: Update the old 3-tuple unpacks in `tests/test_parsing.py`**

Lines 21, 31, 38, 95, and 107 unpack three values. Rewrite them as attribute access:

```python
def test_parse_findings_block_extracts_and_strips():
    parsed = parse_findings_block(SAMPLE)
    assert len(parsed.findings) == 2
    assert parsed.findings[0]["claim"].startswith("LangGraph is")
    assert parsed.refs[0] == ["S1"]
    assert parsed.findings[0]["confidence"] == "high"
    assert "FINDINGS" not in parsed.narrative
    assert parsed.narrative.strip().startswith("Nghiên cứu")


def test_parse_findings_block_absent():
    parsed = parse_findings_block("Chỉ là văn bản thường.")
    assert parsed.findings == []
    assert parsed.narrative == "Chỉ là văn bản thường."


def test_parse_findings_bad_confidence_skipped():
    text = "## FINDINGS\n- [S1] claim one | confidence: very-high\n"
    assert parse_findings_block(text).findings == []
```

```python
def test_parse_maps_source_refs_to_urls():
    parsed = parse_findings_block(SAMPLE)
    citations = [
        {"url": "https://langchain.ai", "title": "LG docs", "content": ""},
        {"url": "https://temporal.io", "title": "Temporal docs", "content": ""},
    ]
    mapped, unknown = map_refs_to_urls(parsed.findings, parsed.refs, citations)
    assert mapped[0]["source_urls"] == ["https://langchain.ai"]
    assert mapped[1]["source_urls"] == ["https://temporal.io"]
    assert unknown == []


def test_map_refs_unresolved_marker():
    parsed = parse_findings_block("## FINDINGS\n- [S9] Ghost | confidence: low\n")
    mapped, _ = map_refs_to_urls(
        parsed.findings, parsed.refs, [{"url": "https://y.dev", "title": "Y", "content": ""}]
    )
    assert mapped[0]["source_urls"] == ["unresolved:S9"]
```

- [ ] **Step 7: Run the full offline suite**

Run: `uv run pytest -m "not integration" -q`
Expected: 100 passed.

- [ ] **Step 8: Commit**

```bash
git add app/nodes/parsing.py app/nodes/research.py app/main.py tests/
git commit -m "fix: surface dropped FINDINGS lines and missing contract block"
```

---

### Task 5: Multi-source citations per finding

Spec §4.5. Finding #7. `RESEARCH_SYSTEM_PROMPT`, `_LINE_RE`, and `map_refs_to_urls` are one coupled unit — this task touches all three in a single commit.

**Files:**
- Modify: `app/nodes/parsing.py:7-11,15-37`
- Modify: `app/agent.py:32-39`
- Test: `tests/test_parsing.py`, `tests/test_agent.py`

**Interfaces:**
- Consumes: `FindingsParse` from Task 4.
- Produces: `parse_findings_block` populates `refs[i]` with one entry per `[Sn]` on the line. `map_refs_to_urls` is unchanged — it already iterates the per-finding ref list.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_parsing.py`:

```python
def test_parse_accepts_multiple_refs_per_claim():
    text = "## FINDINGS\n- [S1][S2] Cross-checked claim | confidence: high\n"
    parsed = parse_findings_block(text)
    assert parsed.refs == [["S1", "S2"]]
    assert parsed.findings[0]["claim"] == "Cross-checked claim"


def test_parse_accepts_comma_separated_refs():
    text = "## FINDINGS\n- [S1], [S3] Another claim | confidence: medium\n"
    assert parse_findings_block(text).refs == [["S1", "S3"]]


def test_multi_ref_maps_to_multiple_urls():
    parsed = parse_findings_block(
        "## FINDINGS\n- [S1][S2] Cross-checked | confidence: high\n"
    )
    citations = [{"url": "https://a.dev"}, {"url": "https://b.dev"}]
    mapped, _ = map_refs_to_urls(parsed.findings, parsed.refs, citations)
    assert mapped[0]["source_urls"] == ["https://a.dev", "https://b.dev"]
```

Append to `tests/test_agent.py`:

```python
def test_system_prompt_documents_multi_ref():
    assert "[S1][S2]" in RESEARCH_SYSTEM_PROMPT
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_parsing.py tests/test_agent.py -q`
Expected: FAIL — `assert [] == [['S1', 'S2']]` (the current regex does not match a line with two bracketed refs) and the prompt assertion.

- [ ] **Step 3: Widen the line regex**

Replace `app/nodes/parsing.py:7-11` with:

```python
_LINE_RE = re.compile(
    r"^-\s*(?P<refs>(?:\[S\d+\][\s,]*)+)\s*(?P<claim>.+?)\s*\|\s*confidence:\s*"
    r"(?P<conf>high|medium|low)\s*$",
    re.IGNORECASE,
)
_REF_RE = re.compile(r"S\d+", re.IGNORECASE)
```

- [ ] **Step 4: Collect every ref on the line**

In `parse_findings_block`, replace `refs.append([m.group("ref")])` with:

```python
        refs.append([r.upper() for r in _REF_RE.findall(m.group("refs"))])
```

- [ ] **Step 5: Update the research prompt**

Replace the output-contract section at the end of `RESEARCH_SYSTEM_PROMPT` in `app/agent.py` with:

```
Output contract — end EVERY final reply with exactly this block:

## FINDINGS
- [S1] <one factual claim> | confidence: high|medium|low
- [S1][S2] <a claim confirmed by two sources> | confidence: high|medium|low

Where [Sn] refers to the nth URL in the sources you used, counted in the
order you first used them. Cite EVERY source that supports the claim, not
just the first. One line per claim. No prose inside the block."""
```

- [ ] **Step 6: Run the full offline suite**

Run: `uv run pytest -m "not integration" -q`
Expected: 104 passed.

- [ ] **Step 7: Update the FINDINGS contract section in `CLAUDE.md`**

Replace the regex shown in the "The FINDINGS contract" section with `- [Sn]... claim | confidence: high|medium|low` and add: "A line may carry several refs (`- [S1][S2] claim | ...`); `parse_findings_block` returns a `FindingsParse` NamedTuple whose `dropped` and `block_found` fields surface contract violations as `weak_claims`."

- [ ] **Step 8: Commit**

```bash
git add app/nodes/parsing.py app/agent.py tests/ CLAUDE.md
git commit -m "feat: allow multiple source refs per finding"
```

---

### Task 6: Graph-level regression guard for evidence accumulation

Spec §4.6. This is the test that would have caught finding #1. It must fail against the pre-Task-3 code and pass now.

**Files:**
- Create: `tests/test_graph_iterations.py`

**Interfaces:**
- Consumes: reducers (Task 3), `build_graph` from `app/graph.py`.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Write the test**

Create `tests/test_graph_iterations.py`:

```python
from langchain_core.messages import AIMessage

import app.graph as g
from app.config import Settings


def _run(url, title, claim):
    return [
        AIMessage(
            content=f"Body.\n\n## FINDINGS\n- [S1] {claim} | confidence: high\n",
            additional_kwargs={
                "annotations": [
                    {"url_citation": {"url": url, "title": title, "content": "x"}}
                ]
            },
        )
    ]


class ScriptedAgent:
    """Returns a different message list on each invoke, like a real second pass."""

    def __init__(self, runs):
        self._runs = list(runs)
        self.calls = 0

    def invoke(self, payload, config=None):
        messages = self._runs[min(self.calls, len(self._runs) - 1)]
        self.calls += 1
        return {"messages": messages}


def test_evidence_accumulates_across_iterations(monkeypatch):
    seen = {}
    verdicts = iter([False, True])

    monkeypatch.setattr(
        g,
        "build_research_agent",
        lambda *a, **k: ScriptedAgent(
            [
                _run("https://one.dev", "One", "Claim one"),
                _run("https://two.dev", "Two", "Claim two"),
            ]
        ),
    )
    monkeypatch.setattr(
        g,
        "make_verify_node",
        lambda *a, **k: (
            lambda state: {
                "sufficient": next(verdicts),
                "gaps": ["dig deeper"],
                "weak_claims": [],
                "contradictory_claims": [],
            }
        ),
    )

    def fake_answer_factory(*a, **k):
        def node(state):
            seen["findings"] = list(state.get("findings") or [])
            seen["sources"] = list(state.get("sources") or [])
            seen["iteration"] = state.get("iteration")
            return {"answer": "done"}

        return node

    monkeypatch.setattr(g, "make_answer_node", fake_answer_factory)

    graph = g.build_graph(Settings(_env_file=None))  # type: ignore[call-arg]
    graph.invoke({"question": "Q?", "iteration": 0, "max_iterations": 3})

    assert seen["iteration"] == 2, "two research passes should have run"
    assert [f["claim"] for f in seen["findings"]] == ["Claim one", "Claim two"]
    assert [s["url"] for s in seen["sources"]] == ["https://one.dev", "https://two.dev"]
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_graph_iterations.py -q`
Expected: 1 passed. If it fails with only `Claim two` present, Task 3's reducers were not wired into `ResearchState`.

- [ ] **Step 3: Prove it is a real guard**

Run: `git stash push app/state.py && uv run pytest tests/test_graph_iterations.py -q; git stash pop`
Expected: FAIL with `assert ['Claim two'] == ['Claim one', 'Claim two']` while stashed, then PASS again after the pop. This confirms the test detects the original bug rather than passing vacuously.

- [ ] **Step 4: Commit**

```bash
git add tests/test_graph_iterations.py
git commit -m "test: guard evidence accumulation across research iterations"
```

---

### Task 7: `UsageCollector` for out-of-band search spend

Spec §5.2–§5.4. Finding #3.

**Files:**
- Create: `app/usage.py`
- Create: `tests/test_usage.py`
- Modify: `app/tools/search_tool.py`
- Modify: `app/agent.py:42-56`
- Modify: `app/nodes/research.py:41-49,68-79`
- Modify: `app/graph.py:23-38`
- Modify: `app/main.py:26-46`
- Modify: `app/nodes/parsing.py:99-111`
- Test: `tests/test_search_tool.py`

**Interfaces:**
- Consumes: the delta-only node contract from Task 3.
- Produces:
  - `app.usage.UsageCollector` with `add(tokens: int = 0, cost: float = 0.0, searches: int = 0) -> None` and `drain() -> tuple[int, float, int]`.
  - `make_web_search(settings, transport=None, usage=None)`
  - `build_research_agent(settings=None, usage=None)`
  - `make_research_node(agent, settings, usage=None)`

- [ ] **Step 1: Write the failing collector tests**

Create `tests/test_usage.py`:

```python
from app.usage import UsageCollector


def test_add_accumulates():
    u = UsageCollector()
    u.add(tokens=100, cost=0.01, searches=1)
    u.add(tokens=50, cost=0.005, searches=2)
    assert u.drain() == (150, 0.015, 3)


def test_drain_resets():
    u = UsageCollector()
    u.add(tokens=10, cost=0.1, searches=1)
    u.drain()
    assert u.drain() == (0, 0.0, 0)


def test_empty_collector_drains_to_zeros():
    assert UsageCollector().drain() == (0, 0.0, 0)
```

- [ ] **Step 2: Write the failing search-tool usage test**

Append to `tests/test_search_tool.py`:

```python
def test_web_search_records_usage():
    from app.usage import UsageCollector

    body = _ok_response_body()
    body["usage"]["total_tokens"] = 900
    body["usage"]["cost"] = 0.0042

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    collector = UsageCollector()
    tool = make_web_search(
        Settings(_env_file=None),  # type: ignore[call-arg]
        transport=httpx.MockTransport(handler),
        usage=collector,
    )
    tool.invoke({"query": "q"})
    tokens, cost, searches = collector.drain()
    assert tokens == 900
    assert abs(cost - 0.0042) < 1e-9
    assert searches == 1
```

- [ ] **Step 3: Run to verify both fail**

Run: `uv run pytest tests/test_usage.py tests/test_search_tool.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.usage'`.

- [ ] **Step 4: Create `app/usage.py`**

```python
import threading


class UsageCollector:
    """Collects usage from HTTP calls made outside LangChain's message stream.

    The `web_search` tool issues its own OpenRouter request, so its tokens and
    cost never reach `sum_usage`. Nodes call `drain()` once per iteration; the
    reset is what stops iteration 2 from re-counting iteration 1.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tokens = 0
        self._cost = 0.0
        self._searches = 0

    def add(self, tokens: int = 0, cost: float = 0.0, searches: int = 0) -> None:
        with self._lock:
            self._tokens += int(tokens or 0)
            self._cost += float(cost or 0.0)
            self._searches += int(searches or 0)

    def drain(self) -> tuple[int, float, int]:
        """Return the accumulated totals and reset to zero."""
        with self._lock:
            totals = (self._tokens, round(self._cost, 6), self._searches)
            self._tokens = 0
            self._cost = 0.0
            self._searches = 0
        return totals
```

- [ ] **Step 5: Report usage from the search tool**

In `app/tools/search_tool.py`, change the factory signature and record usage before returning. Replace line 8 with:

```python
def make_web_search(settings: Settings, transport=None, usage=None):
```

and replace lines 55-61 with:

```python
        totals = data.get("usage") or {}
        details = totals.get("server_tool_use_details") or totals.get("server_tool_use") or {}
        searches = details.get("web_search_requests", 0) if isinstance(details, dict) else 0
        if usage is not None:
            usage.add(
                tokens=totals.get("total_tokens", 0) or 0,
                cost=totals.get("cost", 0.0) or 0.0,
                searches=searches,
            )
        header = f"SEARCH_RESULTS ({len(lines)} results, {searches} search executed):"
        return header + "\n\n" + "\n\n".join(lines)
```

- [ ] **Step 6: Thread the collector through the agent factory**

In `app/agent.py`, change the signature and the tool construction:

```python
def build_research_agent(settings: Settings | None = None, usage=None):
    s = settings or get_settings()
    model = get_model("researcher", s)
    kwargs = dict(
        model=model,
        tools=[make_web_search(s, usage=usage), make_web_fetch(s.fetch)],
        system_prompt=RESEARCH_SYSTEM_PROMPT,
    )
```

The rest of the function is unchanged from Task 2.

- [ ] **Step 7: Drain the collector in the research node**

In `app/nodes/research.py`, change the factory signature:

```python
def make_research_node(agent, settings: Settings, usage=None) -> Callable[[ResearchState], dict]:
```

and replace the return block written in Task 3 with:

```python
        tokens, cost = sum_usage(messages)
        tool_tokens, tool_cost, tool_searches = (
            usage.drain() if usage is not None else (0, 0.0, 0)
        )
        return {
            "findings": findings,
            "sources": sources,
            "weak_claims": weak,
            "iteration": 1,
            "search_calls": count_total_searches(messages) + tool_searches,
            "total_tokens": tokens + tool_tokens,
            "total_cost": cost + tool_cost,
        }
```

- [ ] **Step 8: Build and wire the collector in the graph**

Replace `app/graph.py:23-29` with:

```python
def build_graph(settings=None):
    s = settings or get_settings()
    usage = UsageCollector()
    agent = build_research_agent(s, usage=usage)
    graph = StateGraph(ResearchState)
    graph.add_node("research", make_research_node(agent, s, usage=usage))
    graph.add_node("verify", make_verify_node(s))
    graph.add_node("answer", make_answer_node(s))
```

and add `from app.usage import UsageCollector` to the imports.

- [ ] **Step 9: Give the agent-only CLI path its own collector**

In `app/main.py`, `run_question` currently builds its agent without one. Replace lines 26-30 with:

```python
def run_question(question: str, agent=None, settings=None) -> dict:
    s = settings or get_settings()
    usage = UsageCollector()
    deep = agent or build_research_agent(s, usage=usage)
```

and replace `searches = count_total_searches(messages)` with:

```python
    _, _, tool_searches = usage.drain()
    searches = count_total_searches(messages) + tool_searches
```

Add `from app.usage import UsageCollector` to the imports.

- [ ] **Step 10: Remove the duplicate text-header search count**

The collector is now the single source for client-tool search counts. Leaving the regex in place would double-count. Replace `app/nodes/parsing.py:99-111` with:

```python
def count_total_searches(messages) -> int:
    """Counts only server-tool usage reported in message metadata. Searches made
    by the client-side web_search tool arrive through UsageCollector instead."""
    total = 0
    for message in messages:
        try:
            total += count_web_searches(message)
        except (TypeError, ValueError):
            continue
    return total
```

`re` is still used elsewhere in the module (`_LINE_RE`, `_BLOCK_RE`, `collect_search_tool_sources`) — do not remove the import.

- [ ] **Step 11: Run the full offline suite**

Run: `uv run pytest -m "not integration" -q`
Expected: 109 passed. `tests/test_search_tool.py::test_web_search_formats_annotations` still asserts the `SEARCH_RESULTS (1 results, 1 search executed)` header — the header text is deliberately kept for the model to read even though nothing parses it any more.

- [ ] **Step 12: Commit**

```bash
git add app/usage.py app/tools/search_tool.py app/agent.py app/nodes app/graph.py app/main.py tests/
git commit -m "fix: account for web_search tool spend via UsageCollector"
```

---

### Task 8: Turn on OpenRouter usage accounting

Spec §5.5. Finding #2 — `est_cost` currently reports `$0.0000` on every run because nothing requests cost in the payload.

**Files:**
- Modify: `app/models.py:42-56`
- Modify: `app/tools/search_tool.py:12-24`
- Test: `tests/test_models.py`, `tests/test_search_tool.py`

**Interfaces:**
- Consumes: `UsageCollector` (Task 7) — it reads `usage.cost`, which only appears once this flag is set.
- Produces: every OpenRouter request carries `usage: {"include": true}`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_models.py`:

```python
def test_payload_requests_usage_accounting():
    from langchain_core.messages import HumanMessage

    payload = get_model("verifier")._get_request_payload([HumanMessage("hi")])
    assert payload["usage"] == {"include": True}
```

Append to `tests/test_search_tool.py`:

```python
def test_web_search_body_requests_usage_accounting():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_ok_response_body())

    tool = make_web_search(
        Settings(_env_file=None),  # type: ignore[call-arg]
        transport=httpx.MockTransport(handler),
    )
    tool.invoke({"query": "q"})
    assert captured["usage"] == {"include": True}
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_models.py tests/test_search_tool.py -q`
Expected: FAIL with `KeyError: 'usage'` in both.

- [ ] **Step 3: Set the flag on every model**

In `app/models.py`, add the kwarg to the `ResearchChatOpenAI` construction in `get_model`:

```python
    return ResearchChatOpenAI(
        model=cfg.model,
        temperature=cfg.temperature,
        api_key=s.openrouter_api_key or "not-set",
        base_url=s.openrouter_base_url,
        max_retries=4,
        timeout=180,
        model_kwargs={"usage": {"include": True}},
    )
```

- [ ] **Step 4: Set the flag on the search tool's own request**

In `app/tools/search_tool.py`, add the key to `body`:

```python
        body = {
            "model": settings.researcher.model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Call web_search now for this query: {query}. Then stop."
                    ),
                }
            ],
            "tools": [build_search_spec(settings.search)],
            "max_tokens": 800,
            "usage": {"include": True},
        }
```

- [ ] **Step 5: Add the integration test that settles whether cost actually arrives**

Append to `tests/test_models.py`:

```python
@pytest.mark.integration
def test_cost_present_in_response_metadata():
    import os

    import pytest as _pytest

    if not os.environ.get("OPENROUTER_API_KEY"):
        _pytest.skip("needs OPENROUTER_API_KEY")
    reply = get_model("verifier").invoke([("human", "Reply with the single word: ok")])
    token_usage = reply.response_metadata.get("token_usage") or {}
    assert "cost" in token_usage, f"no cost in {token_usage}"
    assert float(token_usage["cost"]) > 0
```

Add `import pytest` to the top of `tests/test_models.py`.

- [ ] **Step 6: Run the offline suite**

Run: `uv run pytest -m "not integration" -q`
Expected: 111 passed, 2 deselected.

- [ ] **Step 7: Run the integration check if a key is available**

Run: `uv run pytest -m integration -q`
Expected: PASS. If `cost` is absent, OpenRouter's usage-accounting contract has changed — stop and report rather than deleting the assertion.

- [ ] **Step 8: Commit**

```bash
git add app/models.py app/tools/search_tool.py tests/
git commit -m "fix: request OpenRouter usage accounting so est_cost is real"
```

---

### Task 9: Generalize `call_with_backoff`

Spec §7.2. Prerequisite for Task 10's search retry. Default behavior is unchanged so `tests/test_backoff.py` keeps passing untouched.

**Files:**
- Modify: `app/backoff.py`
- Test: `tests/test_backoff.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `call_with_backoff(fn, *args, attempts=5, base_delay=20.0, retry_on=(OpenAIRateLimitError,), **kwargs)` where `retry_on` is a tuple of exception types **or** a callable `(exc) -> bool`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_backoff.py`:

```python
def test_retry_on_predicate(monkeypatch):
    monkeypatch.setattr("app.backoff.time.sleep", lambda s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ValueError("429 too many requests")
        return "ok"

    result = call_with_backoff(
        flaky,
        attempts=3,
        base_delay=0.0,
        retry_on=lambda exc: "429" in str(exc),
    )
    assert result == "ok"
    assert calls["n"] == 2


def test_retry_on_predicate_declines(monkeypatch):
    monkeypatch.setattr("app.backoff.time.sleep", lambda s: None)

    def broken():
        raise ValueError("404 not found")

    with pytest.raises(ValueError):
        call_with_backoff(
            broken, attempts=3, base_delay=0.0, retry_on=lambda exc: "429" in str(exc)
        )


def test_retry_on_tuple_of_types(monkeypatch):
    monkeypatch.setattr("app.backoff.time.sleep", lambda s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise KeyError("transient")
        return "ok"

    assert call_with_backoff(flaky, attempts=3, base_delay=0.0, retry_on=(KeyError,)) == "ok"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_backoff.py -q`
Expected: FAIL with `TypeError: call_with_backoff() got an unexpected keyword argument 'retry_on'`.

- [ ] **Step 3: Rewrite `app/backoff.py`**

```python
import sys
import time

from langchain_openai.chat_models.base import OpenAIRateLimitError


def _should_retry(exc: BaseException, retry_on) -> bool:
    if callable(retry_on) and not isinstance(retry_on, tuple):
        return bool(retry_on(exc))
    return isinstance(exc, retry_on)


def call_with_backoff(
    fn,
    *args,
    attempts: int = 5,
    base_delay: float = 20.0,
    retry_on=(OpenAIRateLimitError,),
    **kwargs,
):
    """Linear backoff. `retry_on` is a tuple of exception types or a predicate.

    The default keeps the original LLM behavior: retry provider rate limits only,
    20s * attempt. Callers with a different failure profile (the web_search HTTP
    call) pass their own predicate and a much shorter base_delay.
    """
    for attempt in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if not _should_retry(exc, retry_on) or attempt == attempts - 1:
                raise
            wait = base_delay * (attempt + 1)
            print(
                f"[warn] retrying after {type(exc).__name__}, "
                f"attempt {attempt + 1}/{attempts - 1} in {wait:.0f}s...",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait)
```

- [ ] **Step 4: Run the backoff tests**

Run: `uv run pytest tests/test_backoff.py -q`
Expected: 7 passed. `test_non_rate_limit_errors_raise_immediately` still passes because `RuntimeError` is not an `OpenAIRateLimitError` and the default `retry_on` declines it.

- [ ] **Step 5: Run the full offline suite**

Run: `uv run pytest -m "not integration" -q`
Expected: 114 passed.

- [ ] **Step 6: Commit**

```bash
git add app/backoff.py tests/test_backoff.py
git commit -m "refactor: let call_with_backoff take a retry predicate"
```

---

### Task 10: SSRF guard for `web_fetch`, retry for `web_search`

Spec §7.1–§7.2. Findings #5 and #8.

**Files:**
- Modify: `app/config.py` (`FetchConfig`)
- Modify: `app/tools/fetch.py` (full rewrite of `make_web_fetch`)
- Modify: `app/tools/search_tool.py`
- Create: `tests/test_fetch_guard.py`
- Test: `tests/test_fetch.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `call_with_backoff(..., retry_on=...)` from Task 9.
- Produces:
  - `app.tools.fetch.default_resolve(host: str) -> list[str]`
  - `app.tools.fetch.check_url(url: str, cfg: FetchConfig, resolve) -> str | None` — returns a `FETCH_ERROR: ...` message, or `None` when the URL is allowed.
  - `make_web_fetch(cfg, transport=None, resolve=default_resolve)`
  - `FetchConfig.max_redirects: int = 5`, `FetchConfig.allow_private_hosts: bool = False`

- [ ] **Step 1: Write the failing guard tests**

Create `tests/test_fetch_guard.py`:

```python
import httpx

from app.config import FetchConfig
from app.tools.fetch import check_url, make_web_fetch

PUBLIC = ["93.184.216.34"]


def _public_resolver(host):
    return PUBLIC


def _private_resolver(host):
    return ["127.0.0.1"]


def test_check_url_rejects_non_http_scheme():
    assert "FETCH_ERROR" in (check_url("ftp://a.dev/x", FetchConfig(), _public_resolver) or "")


def test_check_url_rejects_loopback():
    msg = check_url("http://internal.example/x", FetchConfig(), _private_resolver)
    assert msg is not None
    assert "private" in msg


def test_check_url_rejects_link_local_metadata():
    msg = check_url(
        "http://metadata.example/latest",
        FetchConfig(),
        lambda host: ["169.254.169.254"],
    )
    assert msg is not None


def test_check_url_allows_public_host():
    assert check_url("https://ok.example/page", FetchConfig(), _public_resolver) is None


def test_check_url_allows_private_when_opted_in():
    cfg = FetchConfig(allow_private_hosts=True)
    assert check_url("http://localhost/x", cfg, _private_resolver) is None


def test_tool_blocks_private_host():
    def handler(request):
        raise AssertionError("request must not be sent to a private host")

    tool = make_web_fetch(
        FetchConfig(),
        transport=httpx.MockTransport(handler),
        resolve=_private_resolver,
    )
    out = tool.invoke({"url": "http://internal.example/x"})
    assert out.startswith("FETCH_ERROR")
    assert "private" in out


def test_tool_blocks_redirect_into_private_network():
    def handler(request):
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "http://internal.example/x"})
        raise AssertionError("must not follow the redirect")

    def resolve(host):
        return ["127.0.0.1"] if host == "internal.example" else PUBLIC

    tool = make_web_fetch(
        FetchConfig(), transport=httpx.MockTransport(handler), resolve=resolve
    )
    out = tool.invoke({"url": "https://ok.example/start"})
    assert out.startswith("FETCH_ERROR")
    assert "private" in out


def test_tool_follows_public_redirect():
    def handler(request):
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "https://ok.example/final"})
        return httpx.Response(
            200,
            text="<html><body><p>Final page body with enough text.</p></body></html>",
            headers={"content-type": "text/html"},
        )

    tool = make_web_fetch(
        FetchConfig(), transport=httpx.MockTransport(handler), resolve=_public_resolver
    )
    out = tool.invoke({"url": "https://ok.example/start"})
    assert "Final page body" in out


def test_tool_stops_at_redirect_limit():
    def handler(request):
        return httpx.Response(302, headers={"location": "https://ok.example/again"})

    tool = make_web_fetch(
        FetchConfig(max_redirects=2),
        transport=httpx.MockTransport(handler),
        resolve=_public_resolver,
    )
    out = tool.invoke({"url": "https://ok.example/start"})
    assert out.startswith("FETCH_ERROR")
    assert "redirect" in out
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_fetch_guard.py -q`
Expected: FAIL with `ImportError: cannot import name 'check_url' from 'app.tools.fetch'`.

- [ ] **Step 3: Add the two fetch settings**

In `app/config.py`, extend `FetchConfig`:

```python
class FetchConfig(BaseModel):
    timeout_seconds: float = 15.0
    max_chars: int = 20000
    user_agent: str = "WebScout/0.1 (research agent)"
    max_download_bytes: int = 2_000_000
    max_redirects: int = 5
    allow_private_hosts: bool = False
```

- [ ] **Step 4: Rewrite `app/tools/fetch.py`**

Keep `clean_html` exactly as it is. Replace everything from the imports through `make_web_fetch`:

```python
import ipaddress
import re
import socket
from urllib.parse import urljoin, urlsplit

import httpx
import trafilatura
from langchain_core.tools import tool

from app.config import FetchConfig

_ALLOWED_SCHEMES = {"http", "https"}


def default_resolve(host: str) -> list[str]:
    """Resolve a hostname to every address it points at."""
    infos = socket.getaddrinfo(host, None)
    return [info[4][0] for info in infos]


def _is_blocked_ip(raw: str) -> bool:
    try:
        ip = ipaddress.ip_address(raw)
    except ValueError:
        return True
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def check_url(url: str, cfg: FetchConfig, resolve=default_resolve) -> str | None:
    """Return a FETCH_ERROR message if the URL must not be fetched, else None.

    Called once per redirect hop, not just on the initial URL — a public host
    that 302s to 169.254.169.254 is the whole reason this exists.
    """
    parts = urlsplit(url)
    if parts.scheme.lower() not in _ALLOWED_SCHEMES:
        return f"FETCH_ERROR: unsupported scheme {parts.scheme!r}"
    host = parts.hostname
    if not host:
        return f"FETCH_ERROR: no host in url {url!r}"
    if cfg.allow_private_hosts:
        return None
    try:
        addresses = resolve(host)
    except Exception as exc:
        return f"FETCH_ERROR: cannot resolve {host}: {type(exc).__name__}: {exc}"
    if not addresses:
        return f"FETCH_ERROR: cannot resolve {host}"
    for address in addresses:
        if _is_blocked_ip(address):
            return f"FETCH_ERROR: refusing to fetch private address {address} for host {host}"
    return None


def make_web_fetch(
    cfg: FetchConfig,
    transport: httpx.BaseTransport | None = None,
    resolve=default_resolve,
):
    @tool
    def web_fetch(url: str) -> str:
        """Fetch a web page and return its readable main text."""
        too_large = (
            f"FETCH_ERROR: response exceeds max_download_bytes "
            f"({cfg.max_download_bytes})"
        )
        current = url
        try:
            with httpx.Client(
                timeout=cfg.timeout_seconds,
                headers={"User-Agent": cfg.user_agent},
                follow_redirects=False,
                transport=transport,
            ) as client:
                for _ in range(cfg.max_redirects + 1):
                    blocked = check_url(current, cfg, resolve)
                    if blocked:
                        return blocked
                    with client.stream("GET", current) as resp:
                        if resp.is_redirect:
                            location = resp.headers.get("location")
                            if not location:
                                return "FETCH_ERROR: redirect without a location header"
                            current = urljoin(current, location)
                            continue
                        resp.raise_for_status()
                        declared = resp.headers.get("content-length")
                        if declared and int(declared) > cfg.max_download_bytes:
                            return too_large
                        buf = bytearray()
                        for chunk in resp.iter_bytes():
                            room = cfg.max_download_bytes + 1 - len(buf)
                            buf.extend(chunk[:room])
                            if len(buf) > cfg.max_download_bytes:
                                return too_large
                        ctype = resp.headers.get("content-type", "")
                        if "html" not in ctype and "text" not in ctype:
                            return f"FETCH_ERROR: unsupported content-type {ctype}"
                        text = bytes(buf).decode(resp.encoding or "utf-8", errors="replace")
                        return clean_html(text, cfg.max_chars)
                return f"FETCH_ERROR: exceeded max_redirects ({cfg.max_redirects})"
        except Exception as exc:
            return f"FETCH_ERROR: {type(exc).__name__}: {exc}"

    return web_fetch
```

The content-type check moved *inside* the `with` block; previously it ran after the response context closed, which happened to work but read on borrowed state.

- [ ] **Step 5: Give the existing fetch tests a resolver**

`tests/test_fetch.py` uses hostnames (`hostile.example`, `ok.example`) that do not resolve. Without a resolver the guard would attempt real DNS and break the offline guarantee. Add at the top of the file:

```python
def _public_resolver(host):
    return ["93.184.216.34"]
```

and pass `resolve=_public_resolver` to every `make_web_fetch(...)` call in `test_tool_rejects_content_length_over_cap`, `test_tool_rejects_streamed_body_over_cap`, and `test_tool_extracts_small_page_via_mock_transport`. Also add `headers={"content-type": "text/html"}` to the response in `test_tool_extracts_small_page_via_mock_transport`, since the content-type check now runs on that path.

`test_tool_returns_error_for_bad_scheme` and `test_tool_returns_error_for_unreachable_host` keep the default resolver: the first is now rejected by `check_url` before any DNS, and the second targets `127.0.0.1:9`, which the guard rejects as loopback without a socket call — that also removes the 2-second timeout from the suite.

- [ ] **Step 6: Retry transient search failures**

In `app/tools/search_tool.py`, add the predicate above the factory:

```python
_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


def _search_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRY_STATUS
    return isinstance(exc, httpx.TransportError)
```

and wrap the request. Replace the `try` block that performs the POST with:

```python
        def _post() -> httpx.Response:
            with httpx.Client(
                transport=transport, timeout=settings.fetch.timeout_seconds * 8
            ) as client:
                response = client.post(
                    settings.openrouter_base_url + "/chat/completions",
                    headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
                    json=body,
                )
                response.raise_for_status()
                return response

        try:
            resp = call_with_backoff(
                _post, attempts=3, base_delay=2.0, retry_on=_search_retryable
            )
        except Exception as exc:
            return f"SEARCH_ERROR: {type(exc).__name__}: {exc}"
```

Add `from app.backoff import call_with_backoff` to the imports.

A 20s×n backoff is for provider rate limits on LLM calls; applied to search it would stall the agent for a minute on one flaky request, hence `attempts=3, base_delay=2.0`.

- [ ] **Step 7: Add a search retry test**

Append to `tests/test_search_tool.py`:

```python
def test_web_search_retries_transient_500(monkeypatch):
    monkeypatch.setattr("app.backoff.time.sleep", lambda s: None)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, json=_ok_response_body())

    tool = make_web_search(
        Settings(_env_file=None),  # type: ignore[call-arg]
        transport=httpx.MockTransport(handler),
    )
    out = tool.invoke({"query": "q"})
    assert out.startswith("SEARCH_RESULTS")
    assert calls["n"] == 2
```

`test_web_search_http_error_returns_search_error` uses a 500, which is retryable — it will now make 3 attempts before returning `SEARCH_ERROR`. The assertion still holds; only the call count changes.

- [ ] **Step 8: Run the full offline suite**

Run: `uv run pytest -m "not integration" -q`
Expected: 124 passed, and noticeably faster — the 2-second unreachable-host test is now short-circuited by the guard.

- [ ] **Step 9: Document the fetch boundary in `README.md`**

Add to the feature bullet list, after the "Real reading" bullet:

```markdown
- **Guarded fetching** — `web_fetch` refuses non-HTTP schemes and any host resolving to a private, loopback, or link-local address, re-checking on every redirect hop.
```

And add two rows to the Configuration table:

```markdown
| `fetch.max_redirects` | `5` | Redirect hops followed, each re-checked against the address guard |
| `fetch.allow_private_hosts` | `false` | Set true only to fetch from localhost or a private network during development |
```

- [ ] **Step 10: Commit**

```bash
git add app/config.py app/tools tests/ README.md
git commit -m "fix: block private-address fetches and retry transient search failures"
```

---

### Task 11: Remove dead code

Spec §8.5. Finding #15. Left until after Tasks 3–10 so it does not conflict with them.

**Files:**
- Modify: `app/tools/search.py:38-39`
- Modify: `app/nodes/parsing.py:40-58,235-259`
- Modify: `app/nodes/research.py`, `app/main.py`
- Test: `tests/test_parsing.py`

**Interfaces:**
- Consumes: everything from Tasks 3–10.
- Produces:
  - `map_refs_to_urls(findings, refs, citations) -> list[Finding]` — single return value.
  - `find_unknown_refs(findings, citations) -> list[str]` — replaces `reconcile_sources`.
  - `attach_server_tools` no longer exists.

- [ ] **Step 1: Update the tests to the new signatures**

In `tests/test_parsing.py`, replace the `reconcile_sources` import with `find_unknown_refs` and rewrite the three affected tests:

```python
def test_find_unknown_refs_flags_missing_citation():
    findings = [
        {"claim": "x", "source_urls": ["https://a.dev"], "confidence": "high"},
        {"claim": "y", "source_urls": ["https://ghost.dev"], "confidence": "low"},
    ]
    citations = [
        {"url": "https://a.dev", "title": "A", "content": "long" * 300},
        {"url": "https://a.dev", "title": "A dup", "content": ""},
    ]
    assert find_unknown_refs(findings, citations) == ["https://ghost.dev"]


def test_find_unknown_refs_flags_unresolved_prefix():
    findings = [{"claim": "g", "source_urls": ["unresolved:S9"], "confidence": "low"}]
    assert find_unknown_refs(findings, []) == ["unresolved:S9"]
```

and drop the `, unknown` / `, _` from the two `map_refs_to_urls` call sites:

```python
    mapped = map_refs_to_urls(parsed.findings, parsed.refs, citations)
```
```python
    mapped = map_refs_to_urls(
        parsed.findings, parsed.refs, [{"url": "https://y.dev", "title": "Y", "content": ""}]
    )
```
```python
    mapped = map_refs_to_urls(parsed.findings, parsed.refs, citations)
    assert mapped[0]["source_urls"] == ["https://a.dev", "https://b.dev"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_parsing.py -q`
Expected: FAIL with `ImportError: cannot import name 'find_unknown_refs'`.

- [ ] **Step 3: Simplify `map_refs_to_urls`**

Replace `app/nodes/parsing.py:40-58` with:

```python
def map_refs_to_urls(
    findings: list[Finding], refs: list[list[str]], citations: list[dict]
) -> list[Finding]:
    ordered_urls = [c.get("url") for c in citations if c.get("url")]
    ref_to_url = {f"S{i + 1}": u for i, u in enumerate(ordered_urls)}
    mapped: list[Finding] = []
    for finding, finding_refs in zip(findings, refs, strict=True):
        urls = [ref_to_url.get(r, f"unresolved:{r}") for r in finding_refs]
        mapped.append(
            Finding(
                claim=finding["claim"],
                source_urls=list(dict.fromkeys(urls)),
                confidence=finding["confidence"],
            )
        )
    return mapped
```

- [ ] **Step 4: Replace `reconcile_sources` with `find_unknown_refs`**

Replace `app/nodes/parsing.py:235-259` with:

```python
def find_unknown_refs(findings: list[Finding], citations: list[dict]) -> list[str]:
    """URLs a finding cites that are not in the reconstructed citation list."""
    known = {c["url"] for c in citations if c.get("url")}
    unknown: list[str] = []
    for finding in findings:
        for url in finding.get("source_urls", []):
            resolvable = url.startswith("http") or url.startswith("unresolved:")
            if resolvable and url not in known and url not in unknown:
                unknown.append(url)
    return unknown
```

- [ ] **Step 5: Update both call sites**

In `app/nodes/research.py`, change the import of `reconcile_sources` to `find_unknown_refs` and replace the two lines:

```python
        findings = map_refs_to_urls(parsed.findings, parsed.refs, ref_order)
        unknown = find_unknown_refs(findings, ref_order)
```

In `app/main.py`, same change:

```python
    findings = map_refs_to_urls(parsed.findings, parsed.refs, ref_order)
    unknown = find_unknown_refs(findings, ref_order)
```

- [ ] **Step 6: Delete `attach_server_tools`**

Remove lines 38-39 of `app/tools/search.py`. `build_search_spec` and `count_web_searches` stay — both are still used.

- [ ] **Step 7: Run the full suite and the linter**

Run:
```powershell
uv run pytest -m "not integration" -q
uv run ruff check .
```
Expected: 124 passed, `All checks passed!`.

- [ ] **Step 8: Commit**

```bash
git add app tests
git commit -m "refactor: drop dead returns, reconcile_sources, and attach_server_tools"
```

---

### Task 12: Eval credibility

Spec §9. Finding #16.

**Files:**
- Modify: `config.yaml`
- Modify: `app/config.py` (`Settings.judge`)
- Modify: `app/models.py:6` (`ROLES`)
- Modify: `evals/evaluators.py`
- Test: `tests/test_evaluators.py`, `tests/test_models.py`

**Interfaces:**
- Consumes: `get_model` (Task 8's `model_kwargs`).
- Produces: `get_model("judge")` is valid; `citation_support_evaluator` scores the fraction of cited refs whose excerpt is judged supporting.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_models.py`:

```python
def test_judge_role_is_valid():
    m = get_model("judge", _settings())
    assert m.model_name
```

Append to `tests/test_evaluators.py`:

```python
def test_citation_support_scores_fraction_of_supported_refs():
    from types import SimpleNamespace

    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    from evals.evaluators import citation_support_evaluator

    judge = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(content='{"supported": true}'),
                AIMessage(content='{"supported": false}'),
            ]
        )
    )
    run = SimpleNamespace(
        inputs={"question": "q"},
        outputs={
            "answer": "Claim A [1] and claim B [2].",
            "sources": [
                {"url": "https://a", "excerpt": "supports A"},
                {"url": "https://b", "excerpt": "unrelated text"},
            ],
        },
    )
    result = citation_support_evaluator(run, None, judge=judge)
    assert abs(result.score - 0.5) < 1e-9
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_models.py tests/test_evaluators.py -q`
Expected: FAIL with `ValueError: unknown role: judge` and a score of `1.0` instead of `0.5`.

- [ ] **Step 3: Add the judge role**

`app/models.py` line 6:

```python
ROLES = ("researcher", "verifier", "answer", "judge")
```

`app/config.py`, alongside the other roles:

```python
    judge: RoleConfig = RoleConfig(temperature=0.0)
```

`config.yaml`, after the `answer` block:

```yaml
judge:
  model: stealth/ox-alpha
  temperature: 0.0
```

The default model is the same today, but the role now exists as a one-line swap so the grader can be moved off the model under test.

- [ ] **Step 4: Grade every cited ref**

In `evals/evaluators.py`, change `correctness_evaluator` line 15 from `judge = get_model("verifier")` to:

```python
    judge = get_model("judge")
```

and replace the body of `citation_support_evaluator` from line 52 (`cited = sorted(refs)`) to the end with:

```python
    cited = sorted(refs)
    checkable = [
        (n, str((sources[n - 1] or {}).get("excerpt") or "").strip())
        for n in cited
        if str((sources[n - 1] or {}).get("excerpt") or "").strip()
    ]
    if not checkable:
        return EvaluationResult(
            key="citation_support",
            score=1.0,
            comment=f"resolved refs, no excerpts to check: {cited}",
        )
    active_judge = judge or get_model("judge")
    supported = 0
    notes: list[str] = []
    for ref, excerpt in checkable:
        verdict = active_judge.invoke(
            [
                (
                    "system",
                    'You verify whether an excerpt supports an answer claim. '
                    'Reply JSON {"supported": bool}.',
                ),
                ("human", f"Answer claim:\n{answer}\n\nExcerpt:\n{excerpt}"),
            ]
        )
        try:
            parsed = json.loads(re.search(r"\{[\s\S]*\}", str(verdict.content)).group(0))
            if bool(parsed["supported"]):
                supported += 1
            else:
                notes.append(f"[{ref}] unsupported")
        except Exception as exc:
            notes.append(f"[{ref}] judge parse failure: {exc}")
    return EvaluationResult(
        key="citation_support",
        score=supported / len(checkable),
        comment=f"{supported}/{len(checkable)} refs supported"
        + (f"; {', '.join(notes)}" if notes else ""),
    )
```

- [ ] **Step 5: Run the full suite and linter**

Run:
```powershell
uv run pytest -m "not integration" -q
uv run ruff check .
```
Expected: 126 passed, `All checks passed!`.

- [ ] **Step 6: Update the config table in `README.md`**

Add after the `answer.temperature` row:

```markdown
| `judge.model` | `stealth/ox-alpha` | Model grading eval runs — swap it off the model under test to avoid self-grading |
| `judge.temperature` | `0.0` | Judge temperature (deterministic) |
```

- [ ] **Step 7: Commit**

```bash
git add config.yaml app/config.py app/models.py evals/evaluators.py tests/ README.md
git commit -m "feat: independent judge role and full citation coverage in evals"
```

---

### Task 13: Re-run the eval baseline

Spec §9, §14. The prompt changed in Task 5, so `evals/runs/skill-ab-v2.md` is no longer comparable.

**Files:**
- Create: `evals/runs/2026-08-26-post-remediation.md`

**Interfaces:**
- Consumes: everything above.
- Produces: a recorded baseline for future comparison.

- [ ] **Step 1: Confirm the pipeline runs end to end**

Run: `uv run webscout "What changed in the EU AI Act in 2026?" --out $env:TEMP\smoke.md`
Expected: prints `[research]`/`[verify]`/`[answer]` progress, an answer, a numbered source list, and a METRICS line where **`est_cost` is not `$0.0000`** and `searches` is at least 1. Both were broken before this plan.

- [ ] **Step 2: Run the eval**

Run: `uv run python -m evals.run_evals --limit 5 --experiment-prefix post-remediation`
Expected: an experiment appears in the LangSmith project with `correctness`, `citation_support`, and the four metric keys.

- [ ] **Step 3: Record the numbers**

Create `evals/runs/2026-08-26-post-remediation.md` containing the experiment name, the date, the `config.yaml` role models used, and the per-key averages that `uv run python -m evals.summarize` prints (or the LangSmith experiment table if the arm names do not match `evals/summarize.py`'s `ARMS` list). Note explicitly that this baseline supersedes `skill-ab-v2.md` because `RESEARCH_SYSTEM_PROMPT` changed in Task 5.

- [ ] **Step 4: Commit**

```bash
git add evals/runs/2026-08-26-post-remediation.md
git commit -m "docs: post-remediation eval baseline"
```

---

## Definition of Done

Verified against spec §14:

1. `uv run pytest -m "not integration" -q` passes on ubuntu + windows × 3.12 + 3.13, with no network access.
2. `tests/test_graph_iterations.py` proves evidence accumulates across iterations (and Task 6 Step 3 proves the test is not vacuous).
3. A real run prints a non-zero `est_cost` that includes `web_search` spend.
4. `uv run webscout "question"` works after a plain `uv sync`, from any directory.
5. `web_fetch` refuses private addresses, including via redirect.
6. `uv run ruff check .` is clean.
7. A new eval baseline exists in `evals/runs/`.
