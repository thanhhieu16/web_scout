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


def test_payload_merge_keeps_client_and_server_tools():
    m = ResearchChatOpenAI(
        model="stealth/ox-alpha",
        api_key="not-set",
        base_url="https://openrouter.ai/api/v1",
        server_tools=[{"type": "openrouter:web_search", "parameters": {}}],
    )
    payload = m._get_request_payload(
        [HumanMessage("hi")],
        **{"tools": [{"type": "function", "function": {"name": "web_fetch"}}]},
    )
    types = [t.get("type") for t in payload["tools"]]
    assert len(payload["tools"]) == 2
    assert "function" in types
    assert "openrouter:web_search" in types
