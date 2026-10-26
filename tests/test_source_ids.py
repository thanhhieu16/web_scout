import httpx
from langchain_core.messages import AIMessage, ToolMessage

from app.config import FetchConfig, Settings
from app.nodes.parsing import build_sources, map_refs_to_urls, parse_findings_block
from app.tools.cache import TTLCache
from app.tools.fetch import make_web_fetch
from app.tools.search_tool import make_web_search
from app.tools.sources import SourceRegistry


def _public_resolver(host):
    return ["93.184.216.34"]


def _search_body(urls):
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "results",
                    "annotations": [
                        {"type": "url_citation", "url_citation": {"url": u, "title": u}}
                        for u in urls
                    ],
                }
            }
        ],
        "usage": {"server_tool_use_details": {"web_search_requests": 1}},
    }


def _tools(registry, urls, cache=None):
    search = make_web_search(
        Settings(_env_file=None),  # type: ignore[call-arg]
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=_search_body(urls))
        ),
        registry=registry,
        cache=cache,
    )
    fetch = make_web_fetch(
        FetchConfig(),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                text=f"<p>body of {request.url}</p>",
                headers={"content-type": "text/html"},
            )
        ),
        resolve=_public_resolver,
        registry=registry,
    )
    return search, fetch


def _transcript(search, fetch, steps, final):
    """Run real tools and lay out the messages the agent loop would produce."""
    messages = []
    for i, (name, arg) in enumerate(steps):
        call_id = f"c{i}"
        key = "query" if name == "web_search" else "url"
        tool = search if name == "web_search" else fetch
        messages.append(
            AIMessage(content="", tool_calls=[{"name": name, "args": {key: arg}, "id": call_id}])
        )
        messages.append(ToolMessage(content=tool.invoke({key: arg}), tool_call_id=call_id))
    messages.append(AIMessage(content=final))
    return messages


def test_registry_assigns_stable_sequential_ids():
    registry = SourceRegistry()
    assert registry.id_for("https://a.dev") == "S1"
    assert registry.id_for("https://b.dev") == "S2"
    assert registry.id_for("https://a.dev") == "S1"


def test_fetch_of_a_search_result_reuses_its_id():
    registry = SourceRegistry()
    search, fetch = _tools(registry, ["https://a.dev/", "https://b.dev/", "https://c.dev/"])
    assert "[S3] https://c.dev/ | " in search.invoke({"query": "q"})
    assert fetch.invoke({"url": "https://c.dev/"}).startswith("[S3] https://c.dev/\n\n")


def test_cited_id_resolves_to_the_page_the_tool_labeled():
    # The agent fetches a page before searching. Positional numbering put all
    # search results first, so the fetched page was not S1 even though it was
    # the first source the agent saw; the tool-printed ID removes that guess.
    registry = SourceRegistry()
    search, fetch = _tools(registry, ["https://a.dev/", "https://b.dev/"])
    messages = _transcript(
        search,
        fetch,
        [("web_fetch", "https://first.dev/"), ("web_search", "q")],
        "Body.\n\n## FINDINGS\n- [S1] from the fetched page | confidence: high\n"
        "- [S3] from the second search hit | confidence: medium\n",
    )
    parsed = parse_findings_block(str(messages[-1].content))
    sources, ref_map = build_sources(messages)
    findings = map_refs_to_urls(parsed.findings, parsed.refs, ref_map)
    assert findings[0]["source_urls"] == ["https://first.dev/"]
    assert findings[1]["source_urls"] == ["https://b.dev/"]
    fetched = next(s for s in sources if s["url"] == "https://first.dev/")
    assert not fetched["excerpt"].startswith("[S1]")


def test_cached_search_hit_is_numbered_by_the_reading_turn():
    cache = TTLCache()
    first_turn, _ = _tools(SourceRegistry(), ["https://a.dev/"], cache=cache)
    first_turn.invoke({"query": "q"})

    registry = SourceRegistry()
    registry.id_for("https://other.dev/")
    second_turn, _ = _tools(registry, ["https://unused.dev/"], cache=cache)
    out = second_turn.invoke({"query": "q"})
    assert "[S2] https://a.dev/ | " in out


def test_cached_fetch_hit_still_carries_its_id():
    cache = TTLCache()
    registry = SourceRegistry()
    fetch = make_web_fetch(
        FetchConfig(),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, text="<p>cached content here</p>", headers={"content-type": "text/html"}
            )
        ),
        resolve=_public_resolver,
        cache=cache,
        registry=registry,
    )
    fetch.invoke({"url": "https://ok.example/page"})
    assert fetch.invoke({"url": "https://ok.example/page"}).startswith(
        "[S1] https://ok.example/page\n\n"
    )
