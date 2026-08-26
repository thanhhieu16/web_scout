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
    payload = (
        '{"sufficient": true, "missing_information": [], '
        '"weak_claims": [], "contradictory_claims": []}'
    )
    node = make_verify_node(Settings(_env_file=None), model=_fake_model(payload))  # type: ignore[call-arg]
    delta = node(dict(STATE))
    assert delta["sufficient"] is True
    assert delta["gaps"] == []


def test_verify_retries_once_after_garbage(capsys):
    good = (
        '{"sufficient": true, "missing_information": [], '
        '"weak_claims": [], "contradictory_claims": []}'
    )
    model = GenericFakeChatModel(
        messages=iter([AIMessage(content="not json at all"), AIMessage(content=good)])
    )
    node = make_verify_node(Settings(_env_file=None), model=model)  # type: ignore[call-arg]
    delta = node(dict(STATE))
    assert delta["sufficient"] is True
    assert delta["gaps"] == []
    assert capsys.readouterr().err == ""


def test_verify_degrades_after_failed_retry(capsys):
    model = GenericFakeChatModel(
        messages=iter([AIMessage(content="junk"), AIMessage(content="still junk")])
    )
    node = make_verify_node(Settings(_env_file=None), model=model)  # type: ignore[call-arg]
    delta = node(dict(STATE))
    assert delta == {
        "sufficient": False,
        "gaps": ["verifier parse error"],
        "contradictory_claims": [],
        "weak_claims": ["old weak"],
        "total_tokens": 0,
        "total_cost": 0.0,
    }
    assert "[warn]" in capsys.readouterr().err
