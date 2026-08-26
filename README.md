# WebScout

WebScout is a mini deep-research agent: it takes a question, searches the web, reads sources, checks whether the evidence is sufficient, and answers with citations. Deep Agents acts as the research harness inside a LangGraph control plane, with all models served through OpenRouter and observability/evals on LangSmith.

## How it works

Deep Agents owns the agent loop (LLM → tool → observation); LangGraph owns the product loop below:

```text
START
  |
  v
research <-------------------------------------+
  |  (Deep Agent: openrouter:web_search,       |
  |   web_fetch -> trafilatura extraction)     |
  v                                            |
verify                                         |
  |  (bounded call: is evidence sufficient?)   |
  |-- insufficient AND iteration < max -------+
  |
  v  (sufficient OR iteration cap reached)
answer
  |
  v
 END
```

Only three nodes exist (`research`, `verify`, `answer`); `verify` and `answer` are single bounded structured calls with no tools and no internal loops. The final answer keeps inline `[n]` citations reconciled against the URLs actually returned by search.

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)
- An [OpenRouter](https://openrouter.ai/) API key
- Optional: a LangSmith API key for tracing and evals

## Setup

```powershell
# 1. Install dependencies (including dev extras)
uv sync --extra dev

# 2. Configure environment
Copy-Item .env.example .env
#    then edit .env and set:
#      OPENROUTER_API_KEY=sk-or-...   (required)
#      LANGSMITH_TRACING=true         (optional)
#      LANGSMITH_API_KEY=lsv2_...     (optional, enables LangSmith tracing)
#      LANGSMITH_PROJECT=webscout     (optional)
```

Precedence for settings: init kwargs > environment variables > `.env` > `config.yaml`.

## Usage

One-shot question (recommended entry point — the repo has no build-system yet, so the `webscout` console script declared in `pyproject.toml` is not installed):

```powershell
uv run python -m app.main "What changed in the EU AI Act in 2026?"
```

Interactive mode (omit the question; type `exit`, `quit`, or press Ctrl+C/Ctrl+D to leave):

```powershell
uv run python -m app.main
webscout> your question here
```

Output prints progress per node (`[research] ...`, `[verify] ...`, `[answer] ...`), then the answer followed by numbered sources. Answer language follows the language of the question.

## Testing

Unit tests (no network, no key required):

```powershell
uv run pytest -m "not integration"
```

Integration tests hit OpenRouter and the live web, so they need `OPENROUTER_API_KEY` and network access:

```powershell
$env:OPENROUTER_API_KEY = "sk-or-..."
uv run pytest -m integration
```

## Evals

Evals run through LangSmith (`LANGSMITH_API_KEY` required): they upload the dataset from `evals/dataset.json`, execute the full graph per example, and score correctness, citation support, and metrics (sources count, search calls).

```powershell
uv run python -m evals.run_evals                       # full dataset
uv run python -m evals.run_evals --limit 2             # first N examples
uv run python -m evals.run_evals --experiment-prefix demo
```

Results appear as an experiment in your LangSmith project.

## Configuration (`config.yaml`)

| Key | Default | Description |
|---|---|---|
| `researcher.model` | `stealth/ox-alpha` | Model for the research agent (via OpenRouter) |
| `researcher.temperature` | `0.2` | Sampling temperature for research |
| `verifier.model` | `stealth/ox-alpha` | Model for the sufficiency verdict |
| `verifier.temperature` | `0.0` | Verifier temperature (deterministic) |
| `answer.model` | `stealth/ox-alpha` | Model writing the final narrative |
| `answer.temperature` | `0.3` | Answer temperature |
| `max_iterations` | `3` | Max research→verify loops before forcing an answer |
| `skills_enabled` | `true` | Attach the `web-research` methodology skill to the agent |
| `search.max_results` | `5` | Results requested per `openrouter:web_search` call |
| `search.max_uses` | `4` | Cap on web_search calls per research turn |
| `search.max_characters` | `4000` | Max characters kept per search result snippet |
| `fetch.timeout_seconds` | `15.0` | HTTP timeout for `web_fetch` |
| `fetch.max_chars` | `20000` | Max characters extracted per fetched page |
| `fetch.user_agent` | `WebScout/0.1 (research agent)` | User-Agent header sent by `web_fetch` |

Environment variable equivalents override YAML where defined (e.g. `OPENROUTER_API_KEY`, also accepts `WEBCOUT_OPENROUTER_API_KEY`); `openrouter_base_url` defaults to `https://openrouter.ai/api/v1`.

## Project layout

```text
app/
  main.py        CLI: run_pipeline (graph) + run_question (agent only)
  graph.py       LangGraph product loop (research -> verify -> answer)
  agent.py       Deep Agents research harness
  models.py      ResearchChatOpenAI (OpenRouter-backed chat model)
  config.py      pydantic-settings (env + config.yaml)
  schemas.py     Findings / verdict / answer contracts
  nodes/         parsing, research, verify, answer nodes
  tools/         web_search adapter, web_fetch (httpx + trafilatura)
skills/
  web-research/  research methodology skill (toggled by skills_enabled)
evals/           dataset.json, evaluators, run_evals runner
tests/           unit suite (+ integration marker)
```

## Docs

- Design spec: [`docs/superpowers/specs/2026-08-25-webscout-design.md`](docs/superpowers/specs/2026-08-25-webscout-design.md)
- Implementation plan: [`docs/superpowers/plans/2026-08-25-webscout-v0-v3.md`](docs/superpowers/plans/2026-08-25-webscout-v0-v3.md)

## Troubleshooting

- **Unicode errors on Windows consoles** (`cp65001` print failures): set `$env:PYTHONIOENCODING="utf-8"` in your PowerShell session before running.
- **`webscout` command not found**: expected — no build-system is declared yet; use `uv run python -m app.main ...` instead.
