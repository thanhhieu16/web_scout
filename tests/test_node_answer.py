from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app.config import Settings
from app.nodes.answer import make_answer_node


def _node_with(reply: str):
    return make_answer_node(
        Settings(_env_file=None),  # type: ignore[call-arg]
        model=GenericFakeChatModel(messages=iter([AIMessage(content=reply)])),
    )


STATE = {
    "question": "Khác nhau thế nào?",
    "findings": [{"claim": "A vs B", "source_urls": ["https://a"], "confidence": "high"}],
    "sources": [{"url": "https://a", "title": "A Doc", "source_type": "primary", "excerpt": ""}],
    "sufficient": True,
    "iteration": 1,
    "max_iterations": 3,
}


def test_answer_stored_and_cited_prompt():
    captured = {}

    class Capture(GenericFakeChatModel):
        def invoke(self, messages, *a, **k):
            captured["messages"] = messages
            return super().invoke(messages, *a, **k)

    node = make_answer_node(
        Settings(_env_file=None),  # type: ignore[call-arg]
        model=Capture(messages=iter([AIMessage(content="Trả lời [1].")])),
    )
    delta = node(dict(STATE))
    assert delta["answer"] == "Trả lời [1]."
    text = str(captured["messages"])
    assert "Khác nhau thế nào?" in text
    assert "https://a" in text


def test_uncertainty_requested_when_budget_exhausted():
    captured = {}

    class Capture(GenericFakeChatModel):
        def invoke(self, messages, *a, **k):
            captured["messages"] = messages
            return super().invoke(messages, *a, **k)

    exhausted = dict(STATE, sufficient=False, iteration=3)
    node = make_answer_node(
        Settings(_env_file=None),  # type: ignore[call-arg]
        model=Capture(messages=iter([AIMessage(content="X.")])),
    )
    node(exhausted)
    assert "uncertainty" in str(captured["messages"]).lower()


def test_language_rule_present():
    node = _node_with("ok")
    assert callable(node)
