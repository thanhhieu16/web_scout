from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage

from app.config import Settings
from app.nodes.verify import make_verify_node


def _fake_model(payload_json: str):
    return GenericFakeChatModel(messages=iter([AIMessage(content=payload_json)]))


STATE = {
    "question": "Q?",
    "findings": [{"claim": "A", "source_urls": ["https://a"], "confidence": "high"}],
    "sources": [{"url": "https://a", "title": "A", "source_type": "secondary", "excerpt": ""}],
    "iteration": 1,
    "max_iterations": 3,
    "weak_claims": ["old weak"],
}


def test_verify_merges_result():
    payload = (
        '{"sufficient": false, "missing_information": ["replay"], '
        '"weak_claims": ["old weak", "new weak"], "contradictory_claims": []}'
    )
    node = make_verify_node(Settings(_env_file=None), model=_fake_model(payload))  # type: ignore[call-arg]
    delta = node(dict(STATE))
    assert delta["sufficient"] is False
    assert delta["gaps"] == ["replay"]
    assert delta["weak_claims"] == ["old weak", "new weak"]


def test_verify_sufficient_clears_gaps():
    payload = '{"sufficient": true, "missing_information": [], "weak_claims": [], "contradictory_claims": []}'
    node = make_verify_node(Settings(_env_file=None), model=_fake_model(payload))  # type: ignore[call-arg]
    delta = node(dict(STATE))
    assert delta["sufficient"] is True
    assert delta["gaps"] == []
