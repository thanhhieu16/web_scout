import httpx
from langchain_core.messages import AIMessage, ToolMessage

from app.config import Settings
from app.nodes.parsing import collect_search_tool_sources
from app.tools.search_tool import make_web_search


def _ok_response_body():
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "results",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "url_citation": {
                                "url": "https://x.dev",
                                "title": "X Title",
                                "content": "useful x content",
                            },
                        },
                        {
                            "type": "url_citation",
                            "url_citation": {"url": "https://x.dev", "title": "dup"},
                        },
                    ],
                }
            }
        ],
        "usage": {"server_tool_use_details": {"web_search_requests": 1}},
    }


def test_web_search_formats_annotations():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_response_body())

    tool = make_web_search(Settings(_env_file=None), transport=httpx.MockTransport(handler))  # type: ignore[call-arg]
    out = tool.invoke({"query": "test query"})
    assert out.startswith("SEARCH_RESULTS (1 results, 1 search executed)")
    assert "[SRC] https://x.dev | X Title" in out
    assert "EXCERPT: useful x content" in out


def test_web_search_http_error_returns_search_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    tool = make_web_search(Settings(_env_file=None), transport=httpx.MockTransport(handler))  # type: ignore[call-arg]
    out = tool.invoke({"query": "q"})
    assert out.startswith("SEARCH_ERROR:")


def test_web_search_no_results_returns_search_error():
    body = {"choices": [{"message": {"role": "assistant", "content": "nothing"}}], "usage": {}}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    tool = make_web_search(Settings(_env_file=None), transport=httpx.MockTransport(handler))  # type: ignore[call-arg]
    out = tool.invoke({"query": "q"})
    assert out == "SEARCH_ERROR: no results returned"


def test_collect_search_tool_sources_pairs_and_parses():
    content = (
        "SEARCH_RESULTS (2 results, 1 search executed):\n\n"
        "[SRC] https://a.dev | A Title\nEXCERPT: alpha excerpt\n\n"
        "[SRC] https://b.dev | B Title\nEXCERPT: beta excerpt"
    )
    messages = [
        AIMessage(
            content="",
            tool_calls=[{"name": "web_search", "args": {"query": "q"}, "id": "s1"}],
        ),
        ToolMessage(content=content, tool_call_id="s1"),
        AIMessage(content="## FINDINGS\n- [S1] claim | confidence: high\n"),
    ]
    sources = collect_search_tool_sources(messages)
    assert [s["url"] for s in sources] == ["https://a.dev", "https://b.dev"]
    assert sources[0]["title"] == "A Title"
    assert sources[0]["excerpt"] == "alpha excerpt"


def test_collect_search_tool_sources_ignores_unrelated_tools():
    messages = [
        AIMessage(
            content="",
            tool_calls=[{"name": "web_fetch", "args": {"url": "https://z"}, "id": "f1"}],
        ),
        ToolMessage(content="[SRC] https://noisy.dev | N\nEXCERPT: x", tool_call_id="f1"),
    ]
    assert collect_search_tool_sources(messages) == []


def test_web_search_records_usage():
    from app.usage import UsageCollector

    body = _ok_response_body()
    body["usage"]["total_tokens"] = 900
    body["usage"]["cost"] = 0.0042

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    collector = UsageCollector()
    tool = make_web_search(
        Settings(_env_file=None),  # type: ignore[call-arg]
        transport=httpx.MockTransport(handler),
        usage=collector,
    )
    tool.invoke({"query": "q"})
    tokens, cost, searches = collector.drain()
    assert tokens == 900
    assert abs(cost - 0.0042) < 1e-9
    assert searches == 1


def test_web_search_records_usage_even_with_no_results():
    from app.usage import UsageCollector

    body = {
        "choices": [{"message": {"role": "assistant", "content": "nothing"}}],
        "usage": {"total_tokens": 120, "cost": 0.0008},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    collector = UsageCollector()
    tool = make_web_search(
        Settings(_env_file=None),  # type: ignore[call-arg]
        transport=httpx.MockTransport(handler),
        usage=collector,
    )
    out = tool.invoke({"query": "q"})
    assert out == "SEARCH_ERROR: no results returned"
    tokens, cost, _ = collector.drain()
    assert tokens == 120
    assert abs(cost - 0.0008) < 1e-9


def test_web_search_body_requests_usage_accounting():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_ok_response_body())

    tool = make_web_search(
        Settings(_env_file=None),  # type: ignore[call-arg]
        transport=httpx.MockTransport(handler),
    )
    tool.invoke({"query": "q"})
    assert captured["usage"] == {"include": True}
