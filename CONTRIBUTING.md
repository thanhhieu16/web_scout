# Contributing

## Setup

```powershell
uv sync                                        # installs the dev group (pytest, ruff)
```

Copy `.env.example` to `.env` and fill in `OPENROUTER_API_KEY` before running the pipeline.

## Running

```powershell
uv run webscout "question"                     # one-shot CLI run
uv run webscout                                # interactive REPL
uv sync --group web && uv run uvicorn web.server:app --reload   # browser chat UI
```

## Tests

```powershell
uv run pytest -m "not integration"             # offline suite — this is what CI runs
uv run pytest -m integration                    # hits OpenRouter + live web; needs a real key
uv run ruff check .
```

The offline suite must stay network-free. If your change needs an HTTP call, thread an injectable
`transport=` (an `httpx.MockTransport`) through the factory function instead of hitting the network —
see the "Testing conventions" section in [CLAUDE.md](CLAUDE.md) for the existing seams
(`make_web_fetch`, `make_web_search`, model/agent `usage=` injection, etc.) and follow the same
pattern for new ones. Anything that genuinely needs a live key or the real web gets
`@pytest.mark.integration`.

## Before opening a PR

- `uv run pytest -m "not integration"` and `uv run ruff check .` both pass.
- New behavior has a test; bug fixes have a regression test that fails without the fix.
- Keep the diff scoped to the change described in the PR — no unrelated refactors.

## Where things live

[CLAUDE.md](CLAUDE.md) documents the architecture and the load-bearing hacks in the model layer —
read it before touching `app/models.py`, the search paths, or the LangGraph node wiring. Design docs
and experiment notes live under [docs/superpowers/](docs/superpowers/).
