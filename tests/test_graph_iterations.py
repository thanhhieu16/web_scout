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
