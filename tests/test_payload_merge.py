from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from app.models import ResearchChatOpenAI, get_model


@tool
def dummy_lookup(x: str) -> str:
    """Dummy client-side tool for tests."""
    return x


def test_bind_tools_keeps_single_client_tool():
    bound = get_model("researcher").bind_tools([dummy_lookup])
    assert set(bound.kwargs) >= {"tools"}
    assert len(bound.kwargs["tools"]) == 1


def test_payload_omits_tools_when_none_present():
    m = ResearchChatOpenAI(
        model="stealth/ox-alpha",
        api_key="not-set",
        base_url="https://openrouter.ai/api/v1",
    )
    payload = m._get_request_payload([HumanMessage("hi")])
    assert "tools" not in payload
