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
    kv = {r.key: r.value for r in results["results"]}
    assert kv["latency_s"] == 12.0
    assert kv["total_tokens"] == 999
    assert kv["search_calls"] == 4
    assert kv["num_sources"] == 1


def test_metrics_evaluator_returns_wrapped_results():
    from types import SimpleNamespace

    from evals.evaluators import metrics_evaluator

    run = SimpleNamespace(
        inputs={"question": "q"},
        outputs={"answer": "a", "sources": [{"url": "u"}], "search_calls": 3},
        start_time=None,
        end_time=None,
        total_tokens=1234,
    )
    out = metrics_evaluator(run, None)
    assert isinstance(out, dict), "langsmith requires an EvaluationResults mapping"
    keys = {r.key for r in out["results"]}
    assert keys == {"latency_s", "total_tokens", "total_cost", "search_calls", "num_sources"}


def test_metrics_evaluator_prefers_outputs_total_tokens_over_run_total_tokens():
    from types import SimpleNamespace

    from evals.evaluators import metrics_evaluator

    run = SimpleNamespace(
        inputs={"question": "q"},
        outputs={
            "answer": "a",
            "sources": [{"url": "u"}],
            "search_calls": 3,
            "total_tokens": 555,
            "total_cost": 0.0099,
        },
        start_time=None,
        end_time=None,
        # run.total_tokens deliberately differs and must lose to outputs["total_tokens"]:
        # it never sees the web_search tool's own OpenRouter call.
        total_tokens=99999,
    )
    out = metrics_evaluator(run, None)
    kv = {r.key: r.value for r in out["results"]}
    assert kv["total_tokens"] == 555
    assert abs(kv["total_cost"] - 0.0099) < 1e-9


def test_metrics_evaluator_falls_back_to_run_total_tokens_when_outputs_lack_it():
    from types import SimpleNamespace

    from evals.evaluators import metrics_evaluator

    run = SimpleNamespace(
        inputs={"question": "q"},
        outputs={"answer": "a", "sources": [{"url": "u"}], "search_calls": 3},
        start_time=None,
        end_time=None,
        total_tokens=777,
    )
    out = metrics_evaluator(run, None)
    kv = {r.key: r.value for r in out["results"]}
    assert kv["total_tokens"] == 777
    assert kv["total_cost"] == 0.0


def test_citation_support_flags_unresolved_refs():
    result = citation_support_evaluator(
        _run("claim [3]", [{"url": "https://x"}]), EXAMPLE
    )
    assert result.score == 0.0


def _fake_judge(reply):
    def invoke(messages):
        return SimpleNamespace(content=reply)

    return SimpleNamespace(invoke=invoke)


class ExplodingJudge:
    def invoke(self, messages):
        raise AssertionError("judge must not be called")


def test_citation_support_judge_supported_scores_full():
    result = citation_support_evaluator(
        _run("claim [1] done", [{"url": "https://x", "title": "X", "excerpt": "claim"}]),
        EXAMPLE,
        judge=_fake_judge('{"supported": true}'),
    )
    assert result.score == 1.0


def test_citation_support_judge_unsupported_scores_zero():
    result = citation_support_evaluator(
        _run("claim [1] done", [{"url": "https://x", "title": "X", "excerpt": "claim"}]),
        EXAMPLE,
        judge=_fake_judge('{"supported": false}'),
    )
    assert result.score == 0.0


def test_citation_support_judge_garbage_scores_zero():
    result = citation_support_evaluator(
        _run("claim [1] done", [{"url": "https://x", "title": "X", "excerpt": "claim"}]),
        EXAMPLE,
        judge=_fake_judge("total garbage"),
    )
    assert result.score == 0.0


def test_citation_support_no_excerpts_skips_judge():
    result = citation_support_evaluator(
        _run("claim [1] done", [{"url": "https://x", "title": "X", "excerpt": ""}]),
        EXAMPLE,
        judge=ExplodingJudge(),
    )
    assert result.score == 1.0


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
