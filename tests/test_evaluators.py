from types import SimpleNamespace

from evals.evaluators import (
    citation_support_evaluator,
    metrics_evaluator,
)


class FakeRun(SimpleNamespace):
    pass


def _run(answer, sources, searches=2, tokens=1234, seconds=9.0):
    _dt = __import__("datetime")
    start = _dt.datetime(2026, 1, 1, 0, 0, 0)
    return FakeRun(
        outputs={"answer": answer, "sources": sources, "search_calls": searches},
        total_tokens=tokens,
        start_time=start,
        end_time=start + _dt.timedelta(seconds=seconds),
        error=None,
    )


EXAMPLE = SimpleNamespace(outputs={"reference_notes": "note"})


def test_metrics_evaluator_values():
    results = metrics_evaluator(
        _run("a [1]", [{"url": "https://x"}], searches=4, tokens=999, seconds=12.0),
        EXAMPLE,
    )
    kv = {r.key: r.value for r in results}
    assert kv["latency_s"] == 12.0
    assert kv["total_tokens"] == 999
    assert kv["search_calls"] == 4
    assert kv["num_sources"] == 1


def test_citation_support_flags_unresolved_refs():
    result = citation_support_evaluator(
        _run("claim [3]", [{"url": "https://x"}]), EXAMPLE
    )
    assert result.score <= 0.5


def test_citation_support_passes_resolved():
    result = citation_support_evaluator(
        _run("claim [1] done", [{"url": "https://x", "title": "X", "excerpt": "claim"}]),
        EXAMPLE,
    )
    assert result.score >= 0.5
