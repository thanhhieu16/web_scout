import pytest

from app.config import RoleConfig, Settings
from app.models import get_model


def _settings() -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]


def test_get_model_verifier_config():
    s = _settings()
    m = get_model("verifier", s)
    assert m.model_name == s.verifier.model
    assert m.temperature == s.verifier.temperature
    assert m.openai_api_base == "https://openrouter.ai/api/v1"


def test_get_model_custom_role_values():
    s = _settings()
    s.researcher = RoleConfig(model="some/model", temperature=0.55)
    m = get_model("researcher", s)
    assert m.model_name == "some/model"
    assert abs(m.temperature - 0.55) < 1e-9


def test_invalid_role_raises():
    with pytest.raises(ValueError):
        get_model("nonexistent", _settings())


def test_research_chat_openai_forces_chat_completions():
    from app.models import ResearchChatOpenAI

    m = ResearchChatOpenAI(api_key="k", base_url="https://x")
    assert m.use_responses_api is False


def test_create_chat_result_lifts_annotations():
    from langchain_core.messages import AIMessage

    from app.models import ResearchChatOpenAI

    m = ResearchChatOpenAI(api_key="k", base_url="https://x")
    response = {
        "id": "gen-1",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "hello",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "url_citation": {"url": "https://a.dev", "title": "A", "content": "ca"},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    result = m._create_chat_result(response)
    msg = result.generations[0].message
    assert isinstance(msg, AIMessage)
    anns = msg.additional_kwargs["annotations"]
    assert anns[0]["url_citation"]["url"] == "https://a.dev"


def test_get_model_retries_and_timeout():
    from app.models import get_model

    m = get_model("verifier")
    assert m.max_retries == 4
    assert m.request_timeout == 180


def test_payload_requests_usage_accounting():
    from langchain_core.messages import HumanMessage

    payload = get_model("verifier")._get_request_payload([HumanMessage("hi")])
    assert payload["extra_body"]["usage"] == {"include": True}


@pytest.mark.integration
def test_cost_present_in_response_metadata():
    import os

    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("needs OPENROUTER_API_KEY")
    reply = get_model("verifier").invoke([("human", "Reply with the single word: ok")])
    token_usage = reply.response_metadata.get("token_usage") or {}
    # The `cost` key only appears when the request asks for usage accounting.
    # Its VALUE depends on the model's price list — stealth/ox-alpha is priced
    # at 0/0, so assert presence and a real token spend, not a positive cost.
    assert "cost" in token_usage, f"no cost in {token_usage}"
    assert float(token_usage["cost"]) >= 0
    assert int(token_usage.get("total_tokens", 0)) > 0, f"no tokens consumed: {token_usage}"
