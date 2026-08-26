import pytest

from app.main import run_question


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
