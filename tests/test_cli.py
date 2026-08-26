import pytest

import app.main
from app.main import main, run_question


class FakeAgent:
    def invoke(self, payload, config=None):
        from langchain_core.messages import AIMessage

        return {
            "messages": [
                AIMessage(
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
            ]
        }


def test_run_question_assembles_result():
    out = run_question("So sánh?", agent=FakeAgent())
    assert "[S1]" in out["answer"]
    assert out["sources"][0]["url"] == "https://docs.langchain.com/lg"
    assert out["sources"][0]["source_type"] == "secondary"
    assert out["findings"][0]["claim"].startswith("LG runs")


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
    main(["why?"])
    captured = capsys.readouterr()
    assert "=== ANSWER ===" in captured.out
    assert "canned answer" in captured.out


def test_interactive_eof_exits_cleanly(monkeypatch, capsys):
    def raise_eof(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)
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
