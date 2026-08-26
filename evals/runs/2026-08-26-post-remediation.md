# Post-remediation baseline — 2026-08-26

Supersedes `evals/runs/skill-ab-v2.md`: `RESEARCH_SYSTEM_PROMPT` changed in Task 5 of the
`audit-remediation` plan (findings may now carry several `[Sn]` refs per line), so the old
baseline's citation shape is no longer comparable. This file is the first live, end-to-end
exercise of all twelve prior remediation tasks together.

## Config in force (`config.yaml`)

```
researcher: stealth/ox-alpha (temperature 0.2)
verifier:   stealth/ox-alpha (temperature 0.0)
answer:     stealth/ox-alpha (temperature 0.3)
judge:      stealth/ox-alpha (temperature 0.0)
max_iterations: 3
skills_enabled: true
search: max_results 5, max_uses 4, max_characters 4000
fetch: timeout_seconds 15.0, max_chars 20000
```

## Step 1 — smoke run

Command:
```
uv run webscout "What are the main obligations under the EU AI Act that took effect in 2025?" --out $env:TEMP\smoke.md
```

Full METRICS line (verbatim):
```
iterations: 1 | searches: 2 | sources: 10 | tokens: 67774 | est_cost: $0.0140
```

Assessment: `tokens` > 0, `searches` >= 1, 10 sources listed, answer carries `[1]`–`[10]`
inline citations that match the SOURCES list — all satisfied. `est_cost` is **non-zero**
($0.0140): `stealth/ox-alpha` is priced 0/0 for prompt+completion tokens on OpenRouter, but
OpenRouter bills `web_search` **requests** separately from token pricing (2 searches this
run), so a run that searches costs real money even on a free model. **Spec §14 DoD item 3 —
"a real run prints a non-zero `est_cost` that includes `web_search` spend" — is genuinely
satisfied by this run.** Full verbatim CLI output, including the answer text, is in the task
report.

## Step 2 — eval run

Command:
```
uv run python -m evals.run_evals --limit 5 --experiment-prefix post-remediation
```

`LANGSMITH_API_KEY` was present (in `.env`, auto-loaded by `app.config`), so the eval did
run against real LangSmith infrastructure — but two problems surfaced:

1. **10-minute process timeout.** 4 of 5 dataset examples completed; the 5th
   ("Ưu nhược điểm của uv so với pip+venv cho dự án Python vừa?") was still `pending` in
   LangSmith (no `end_time`) when the run was killed. This looks like a slow example, not a
   pipeline bug — the other 4 finished within the same window.
2. **`metrics_evaluator` was broken against the installed SDK at the time of this run.**
   `evals/evaluators.py::metrics_evaluator` returned a bare `list[EvaluationResult]`;
   `langsmith==0.11.1`'s `_format_evaluator_result` requires a list of dicts or an
   `EvaluationResults` mapping, not a raw list of `EvaluationResult` objects. Every root run
   raised `ValueError: Expected a list of dicts or EvaluationResults. Received [...]` for this
   evaluator, so the `latency_s` / `total_tokens` / `search_calls` / `num_sources` feedback
   keys were **never recorded** to LangSmith for this experiment. `correctness` and
   `citation_support` (which each return one bare `EvaluationResult`, not a list) were
   unaffected. **This has since been fixed** (`metrics_evaluator` now returns
   `{"results": [...]}`, pinned by `tests/test_evaluators.py::test_metrics_evaluator_returns_wrapped_results`)
   — but that fix landed after this run, so the numbers below for `total_tokens`,
   `search_calls`, and `num_sources` are still hand-computed from `run.outputs` /
   `run.total_tokens`, not from LangSmith feedback. This record documents what actually
   happened in this run; it is not being re-run or backfilled now that the evaluator works.
   A future eval run should produce these four keys natively in LangSmith.

Numbers below are read directly from the LangSmith experiment view
(project `post-remediation-eb864dbc`), not from `uv run python -m evals.summarize` —
`summarize.py`'s hardcoded `ARMS` list (`skill-on-iters3`, `skill-off-iters3`,
`skill-on-iters1`, `skill-off-iters1`) does not include `post-remediation`, so it prints
`MISSING` for all four arms, exactly as anticipated. (Separately, at the time of this run
`evals.summarize` also 401ed when invoked completely standalone — it never imported
`app.config`, so `.env` was never auto-loaded, unlike `run_evals.py`. **This has since been
fixed** by adding that import to `evals/summarize.py`.)

| metric | n | avg | source |
|---|---|---|---|
| correctness | 4 | 0.95 | LangSmith feedback, project `post-remediation-eb864dbc` |
| citation_support | 3 | 0.778 | same (2 runs never reached this evaluator: 1 killed by timeout before it ran, 1 never finished) |
| total_tokens | 4 (manual) | 32,547 | `run.total_tokens` read directly per completed root run (20866, 26418, 36033, 46871) — `metrics_evaluator` feedback is unusable, see above |
| search_calls | 4 (manual) | 2.0 | `run.outputs["search_calls"]` per completed root run (3, 1, 1, 3) |
| num_sources | 4 (manual) | 7.0 | `run.outputs["sources"]` length per completed root run (5, 5, 5, 13) |

Per-run correctness scores: 1.0, 1.0, 1.0, 0.8. Per-run citation_support scores (3 of 4
completed runs): 0.333, 1.0, 1.0.

This baseline is **not directly comparable in scale** to `skill-ab-v2.md`/`v3-experiments.md`
(n=3 per arm there vs. n=4–5 partial here), and is incomplete (1 example never finished, no
recorded latency/token/search/source feedback in LangSmith itself due to the SDK bug above).
Treat it as evidence the post-remediation pipeline runs live and produces graded answers with
multi-ref citations, not as a tuned quality number.
