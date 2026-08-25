from types import SimpleNamespace

from app.config import SearchConfig
from app.tools.search import build_search_spec, count_web_searches


def test_build_search_spec_shape():
    spec = build_search_spec(SearchConfig())
    assert spec["type"] == "openrouter:web_search"
    p = spec["parameters"]
    assert p == {"max_results": 5, "max_uses": 4, "max_characters": 4000}


def test_count_from_usage_dict():
    assert count_web_searches({"usage": {"server_tool_use": {"web_search_requests": 3}}}) == 3


def test_count_from_message_response_metadata():
    msg = SimpleNamespace(
        response_metadata={"usage": {"server_tool_use": {"web_search_requests": 2}}},
        additional_kwargs={},
    )
    assert count_web_searches(msg) == 2


def test_count_missing_is_zero():
    assert count_web_searches({"usage": {}}) == 0
    assert count_web_searches(None) == 0
