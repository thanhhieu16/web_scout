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


class FakeGraph:
    def __init__(self, deltas):
        self._deltas = deltas

    def stream(self, state, stream_mode="updates"):
        yield from self._deltas


DELTAS = [
    {"research": {"findings": [{"claim": "c", "source_urls": [], "confidence": "high"}], "sources": [{"url": "https://s", "title": "S", "source_type": "secondary", "excerpt": ""}], "iteration": 1, "search_calls": 3, "weak_claims": []}},
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


def test_main_fast_fails_without_api_key(monkeypatch):
    def forbidden_input(prompt=""):
        raise AssertionError("input must not be called without an API key")

    def forbidden_pipeline(question, graph=None):
        raise AssertionError("run_pipeline must not run without an API key")

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("WEBCOUT_OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(app.main, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr("builtins.input", forbidden_input)
    monkeypatch.setattr(app.main, "run_pipeline", forbidden_pipeline)
    with pytest.raises(SystemExit):
        main(["q"])


def test_run_pipeline_injected_graph_works_without_api_key(monkeypatch):
    from app.main import run_pipeline

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("WEBCOUT_OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(app.main, "get_settings", lambda: Settings(_env_file=None))
    out = run_pipeline("Q?", graph=FakeGraph(DELTAS))
    assert out["answer"] == "Final [1]."


def test_require_openrouter_key_message():
    with pytest.raises(SystemExit, match="OPENROUTER_API_KEY is not set"):
        require_openrouter_key(Settings(_env_file=None))
