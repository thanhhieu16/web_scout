from types import SimpleNamespace

from app.config import Settings
from app.nodes.research import build_research_input, make_research_node


class FakeAgent:
    def __init__(self, msg):
        self._msg = msg

    def invoke(self, payload, config=None):
        self.last_payload = payload
        return {"messages": [self._msg]}


def _msg(content, citations):
    return SimpleNamespace(
        content=content,
        annotations=[{"url_citation": c} for c in citations],
        additional_kwargs={},
        response_metadata={"usage": {"server_tool_use": {"web_search_requests": 2}}},
    )


CONTENT = (
    "Body.\n\n## FINDINGS\n- [S1] Claim A | confidence: high\n"
)


def test_first_iteration_prompt_only_question():
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    text = build_research_input({"question": "Q?", "iteration": 0})
    assert "Q?" in text
    assert "Existing findings" not in text


def test_followup_prompt_includes_gaps():
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    text = build_research_input(
        {
            "question": "Q?",
            "iteration": 1,
            "findings": [],
            "gaps": ["replay semantics"],
        }
    )
    assert "replay semantics" in text
    assert "ONLY these points" in text


def test_node_updates_state():
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    agent = FakeAgent(
        _msg(CONTENT, [{"url": "https://x.dev", "title": "X", "content": "cx"}])
    )
    node = make_research_node(agent, s)
    delta = node(
        {"question": "Q?", "iteration": 0, "gaps": [], "weak_claims": []}
    )
    assert delta["iteration"] == 1
    assert delta["search_calls"] == 2
    assert delta["sources"][0]["url"] == "https://x.dev"
    assert delta["findings"][0]["source_urls"] == ["https://x.dev"]


def test_unknown_ref_becomes_weak_claim():
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    bad = "Body.\n\n## FINDINGS\n- [S9] Ghost claim | confidence: low\n"
    agent = FakeAgent(_msg(bad, [{"url": "https://y.dev", "title": "Y", "content": ""}]))
    delta = make_research_node(agent, s)(
        {"question": "Q?", "iteration": 1, "gaps": [], "weak_claims": []}
    )
    assert any("Ghost claim" in w for w in delta["weak_claims"])
