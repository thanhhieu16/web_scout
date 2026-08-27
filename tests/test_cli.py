import pytest

import app.main
from app.config import Settings
from app.main import main, require_openrouter_key, run_question


def _cli_message():
    from langchain_core.messages import AIMessage

    return AIMessage(
        content=(
            "Answer body citing [S1].\n\n## FINDINGS\n"
            "- [S1] LG runs on LangGraph | confidence: high\n"
        ),
        additional_kwargs={
            "annotations": [
                {
                    "url_citation": {
                        "url": "https://docs.langchain.com/lg",
                        "title": "LG Docs",
                        "content": "lg docs text",
                    }
                }
            ]
        },
    )


class FakeAgent:
    def __init__(self, messages=None):
        self._messages = list(messages) if messages is not None else [_cli_message()]

    def invoke(self, payload, config=None):
        return {"messages": self._messages}


def test_run_question_assembles_result():
    out = run_question("So sánh?", agent=FakeAgent())
    assert "[S1]" in out["answer"]
    assert out["sources"][0]["url"] == "https://docs.langchain.com/lg"
    assert out["sources"][0]["source_type"] == "secondary"
    assert out["findings"][0]["claim"].startswith("LG runs")


def test_run_question_collects_citations_across_messages():
    from types import SimpleNamespace

    early = SimpleNamespace(
        content="searching...",
        annotations=[
            {
                "url_citation": {
                    "url": "https://early.dev",
                    "title": "Early",
                    "content": "early text",
                }
            }
        ],
        additional_kwargs={},
        response_metadata={"usage": {"server_tool_use": {"web_search_requests": 2}}},
    )
    out = run_question("Q?", agent=FakeAgent([early, _cli_message()]))
    urls = [src["url"] for src in out["sources"]]
    assert urls == ["https://early.dev", "https://docs.langchain.com/lg"]
    assert out["search_calls"] == 2


def test_run_question_uses_injected_usage_collector():
    """When an agent is injected, run_question must not build its own orphaned
    UsageCollector wired to nothing (drain() always zeros) — a caller that also
    built the agent should be able to pass the same collector the agent uses."""
    from app.usage import UsageCollector

    usage = UsageCollector()
    usage.add(tokens=42, cost=0.01, searches=2)
    out = run_question("So sánh?", agent=FakeAgent(), usage=usage)
    assert out["search_calls"] == 2


@pytest.mark.integration
def test_cli_real_roundtrip():
    import os

    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("needs OPENROUTER_API_KEY")
    out = run_question("Deep agents la gi? Tra loii ngan.")
    assert out["sources"], "expected at least one citation"
    assert "[S" in out["answer"] or out["answer"].strip()


def test_one_shot_never_reads_stdin(monkeypatch, capsys):
    def forbidden_input(prompt=""):
        raise AssertionError("input must not be called in one-shot mode")

    def fake_pipeline(question, graph=None):
        return {"answer": "canned answer", "sources": [], "findings": [], "search_calls": 0}

    monkeypatch.setattr("builtins.input", forbidden_input)
    monkeypatch.setattr(app.main, "run_pipeline", fake_pipeline)
    monkeypatch.setattr(
        app.main,
        "get_settings",
        lambda: Settings(_env_file=None, openrouter_api_key="test-key"),
    )
    main(["why?"])
    captured = capsys.readouterr()
    assert "=== ANSWER ===" in captured.out
    assert "canned answer" in captured.out


def test_interactive_eof_exits_cleanly(monkeypatch, capsys):
    def raise_eof(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)
    monkeypatch.setattr(
        app.main,
        "get_settings",
        lambda: Settings(_env_file=None, openrouter_api_key="test-key"),
    )
    main([])
    captured = capsys.readouterr()
    assert captured.err == ""


def _state_reducers() -> dict:
    """The reducer for each Annotated field on ResearchState, keyed by field name."""
    from typing import get_type_hints

    from app.state import ResearchState

    hints = get_type_hints(ResearchState, include_extras=True)
    return {
        name: hint.__metadata__[0]
        for name, hint in hints.items()
        if hasattr(hint, "__metadata__")
    }


class FakeGraph:
    """Stands in for LangGraph's real .stream(): a bare string stream_mode (e.g.
    "updates") yields each node's raw delta dict directly, exactly like the real
    single-mode contract; a list stream_mode (e.g. ["updates", "values"]) yields
    (mode, chunk) tuples, where "values" chunks carry the fully-reduced running
    state — computed here by applying ResearchState's own reducers, the same way
    LangGraph does internally."""

    def __init__(self, node_deltas):
        self._node_deltas = node_deltas  # list of {node_name: delta_dict}

    def stream(self, state, stream_mode="updates"):
        if isinstance(stream_mode, (list, tuple)) and "values" in stream_mode:
            yield from self._stream_with_values(state)
        else:
            yield from self._node_deltas

    def _stream_with_values(self, state):
        reducers = _state_reducers()
        values = dict(state)
        for update in self._node_deltas:
            for node, delta in update.items():
                yield ("updates", {node: delta})
                for key, val in delta.items():
                    reducer = reducers.get(key)
                    if reducer is not None and key in values:
                        values[key] = reducer(values[key], val)
                    else:
                        values[key] = val
                yield ("values", dict(values))


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


def test_run_pipeline_prints_progress_and_answer(capsys):
    from app.main import run_pipeline

    out = run_pipeline("Q?", graph=FakeGraph(DELTAS))
    captured = capsys.readouterr().out
    assert "research" in captured
    assert "verify" in captured
    assert "answer" in captured
    assert out["answer"] == "Final [1]."
    assert out["search_calls"] == 3
    assert out["sufficient"] is True


def test_stream_pipeline_yields_status_then_result():
    from app.main import stream_pipeline

    events = list(stream_pipeline("Q?", graph=FakeGraph(DELTAS)))
    kinds = [kind for kind, _ in events]
    assert kinds == ["status", "status", "status", "result"]
    assert [payload for kind, payload in events[:3]] == ["research", "verify", "answer"]
    result = events[-1][1]
    assert result["answer"] == "Final [1]."
    assert result["search_calls"] == 3
    assert result["sufficient"] is True


def test_render_report_markdown_matches_written_file(tmp_path, monkeypatch):
    from datetime import datetime as real_datetime

    import app.main as m

    class _FrozenDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return real_datetime(2026, 8, 27, 12, 0, 0)

    monkeypatch.setattr(m, "datetime", _FrozenDatetime)

    out = {
        "answer": "Body text [1].",
        "sources": [{"url": "https://a.dev", "title": "A"}],
        "findings": [{"claim": "c", "confidence": "high", "source_urls": ["https://a.dev"]}],
        "sufficient": True,
        "iteration": 2,
        "search_calls": 3,
        "total_tokens": 500,
        "total_cost": 0.01,
    }
    path = tmp_path / "report.md"
    m.write_report("Q?", out, str(path))
    assert m.render_report_markdown("Q?", out) == path.read_text(encoding="utf-8")


def test_run_pipeline_max_iterations_override():
    from app.main import run_pipeline

    class _CapturingGraph:
        def __init__(self, inner):
            self._inner = inner
            self.seen_state = None

        def stream(self, state, stream_mode="updates"):
            self.seen_state = state
            yield from self._inner.stream(state, stream_mode=stream_mode)

    capturing = _CapturingGraph(FakeGraph(DELTAS))
    run_pipeline("Q?", graph=capturing, max_iterations=1)
    assert capturing.seen_state["max_iterations"] == 1


def test_main_fast_fails_without_api_key(monkeypatch):
    def forbidden_input(prompt=""):
        raise AssertionError("input must not be called without an API key")

    def forbidden_pipeline(question, graph=None):
        raise AssertionError("run_pipeline must not run without an API key")

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("WEBSCOUT_OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(app.main, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr("builtins.input", forbidden_input)
    monkeypatch.setattr(app.main, "run_pipeline", forbidden_pipeline)
    with pytest.raises(SystemExit):
        main(["q"])


TWO_ITERATION_DELTAS = [
    {
        "research": {
            "findings": [{"claim": "c1", "source_urls": ["https://s1"], "confidence": "high"}],
            "sources": [
                {"url": "https://s1", "title": "S1", "source_type": "secondary", "excerpt": ""}
            ],
            "weak_claims": [],
            "iteration": 1,
            "search_calls": 2,
            "total_tokens": 100,
            "total_cost": 0.01,
        }
    },
    {
        "verify": {
            "sufficient": False,
            "gaps": ["need more"],
            "weak_claims": [],
            "contradictory_claims": [],
            "total_tokens": 10,
            "total_cost": 0.001,
        }
    },
    {
        "research": {
            "findings": [{"claim": "c2", "source_urls": ["https://s2"], "confidence": "medium"}],
            "sources": [
                {"url": "https://s2", "title": "S2", "source_type": "secondary", "excerpt": ""}
            ],
            "weak_claims": [],
            "iteration": 1,
            "search_calls": 3,
            "total_tokens": 200,
            "total_cost": 0.02,
        }
    },
    {
        "verify": {
            "sufficient": True,
            "gaps": [],
            "weak_claims": [],
            "contradictory_claims": [],
            "total_tokens": 10,
            "total_cost": 0.001,
        }
    },
    {
        "answer": {
            "answer": "Final answer citing both [1][2].",
            "total_tokens": 50,
            "total_cost": 0.005,
        }
    },
]


def test_run_pipeline_survives_two_research_iterations():
    """Regression for the Critical finding: streaming raw per-node deltas through
    dict.update() silently overwrote the first research pass's findings/sources
    and reset the iteration count once nodes stopped returning running totals."""
    from app.main import run_pipeline

    out = run_pipeline("Q?", graph=FakeGraph(TWO_ITERATION_DELTAS))
    assert out["iteration"] == 2
    assert [s["url"] for s in out["sources"]] == ["https://s1", "https://s2"]
    assert [f["claim"] for f in out["findings"]] == ["c1", "c2"]
    assert out["search_calls"] == 5


def test_run_pipeline_injected_graph_works_without_api_key(monkeypatch):
    from app.main import run_pipeline

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("WEBSCOUT_OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(app.main, "get_settings", lambda: Settings(_env_file=None))
    out = run_pipeline("Q?", graph=FakeGraph(DELTAS))
    assert out["answer"] == "Final [1]."


def test_require_openrouter_key_message(monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    with pytest.raises(SystemExit, match="OPENROUTER_API_KEY is not set"):
        require_openrouter_key(Settings(_env_file=None))


def test_write_report_markdown(tmp_path):
    from app.main import write_report

    out = {
        "answer": "Body text [1].",
        "sources": [{"url": "https://a.dev", "title": "A"}],
        "findings": [{"claim": "Claim A", "source_urls": ["https://a.dev"], "confidence": "high"}],
        "sufficient": True,
        "iteration": 1,
        "search_calls": 2,
        "total_tokens": 1234,
        "total_cost": 0.0123,
    }
    path = tmp_path / "report.md"
    write_report("Q?", out, str(path))
    text = path.read_text(encoding="utf-8")
    assert "# WebScout Report" in text
    assert "**Question:** Q?" in text
    assert "Body text [1]." in text
    assert "(high) Claim A" in text
    assert "[A](https://a.dev)" in text
    assert "Est. cost: $0.0123" in text


def test_run_pipeline_returns_usage_fields():
    from app.main import run_pipeline

    class FakeGraph:
        def stream(self, state, stream_mode="updates"):
            delta = {"answer": "done", "total_tokens": 500, "total_cost": 0.01}
            yield ("updates", {"answer": delta})
            yield ("values", {**state, **delta})

    out = run_pipeline("Q?", graph=FakeGraph())
    assert out["total_tokens"] == 500
    assert abs(out["total_cost"] - 0.01) < 1e-9


def test_list_models_prints_shortlist_and_skips_pipeline(monkeypatch, capsys):
    from app.config import MODEL_CHOICES

    def forbidden(*a, **k):
        raise AssertionError("--list-models must not run the pipeline")

    monkeypatch.setattr(app.main, "run_pipeline", forbidden)
    main(["--list-models"])
    out = capsys.readouterr().out
    for slug in MODEL_CHOICES:
        assert slug in out


def test_model_flag_overrides_every_role(monkeypatch, capsys):
    from app.config import ROLE_NAMES, get_settings

    seen = {}

    def fake_pipeline(question, graph=None):
        seen["model"] = get_settings().researcher.model
        return {"answer": "a", "sources": [], "findings": [], "search_calls": 0}

    monkeypatch.setattr(app.main, "run_pipeline", fake_pipeline)
    monkeypatch.setattr(app.main, "require_openrouter_key", lambda *a, **k: None)
    try:
        main(["--model", "google/gemma-4-31b-it:free", "why?"])
        assert seen["model"] == "google/gemma-4-31b-it:free"
        assert all(
            getattr(get_settings(), r).model == "google/gemma-4-31b-it:free"
            for r in ROLE_NAMES
        )
    finally:
        get_settings.cache_clear()


def test_model_flag_accepts_shortlist_number(monkeypatch):
    from app.config import MODEL_CHOICES, get_settings

    monkeypatch.setattr(
        app.main,
        "run_pipeline",
        lambda q, graph=None: {"answer": "", "sources": [], "findings": [], "search_calls": 0},
    )
    monkeypatch.setattr(app.main, "require_openrouter_key", lambda *a, **k: None)
    try:
        main(["--model", "2", "q"])
        assert get_settings().researcher.model == MODEL_CHOICES[1]
    finally:
        get_settings.cache_clear()


def test_repl_model_command_switches_without_researching(monkeypatch, capsys):
    from app.config import get_settings

    def forbidden(*a, **k):
        raise AssertionError("/model must not run the pipeline")

    answers = iter(["/model google/gemma-4-26b-a4b-it:free", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr(app.main, "run_pipeline", forbidden)
    monkeypatch.setattr(app.main, "require_openrouter_key", lambda *a, **k: None)
    try:
        main([])
        assert get_settings().researcher.model == "google/gemma-4-26b-a4b-it:free"
    finally:
        get_settings.cache_clear()
