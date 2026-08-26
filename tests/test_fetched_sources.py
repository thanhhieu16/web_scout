from langchain_core.messages import AIMessage, ToolMessage

from app.nodes.parsing import (
    build_sources,
    collect_fetched_sources,
)


def _fetch_call(id_, url):
    return {"name": "web_fetch", "args": {"url": url}, "id": id_}


def test_collect_fetched_sources_pairs_calls_with_results():
    messages = [
        AIMessage(content="", tool_calls=[_fetch_call("c1", "https://a.dev")]),
        ToolMessage(content="page text about a", tool_call_id="c1"),
        AIMessage(
            content="",
            tool_calls=[
                _fetch_call("c2", "https://b.dev"),
                _fetch_call("c3", "https://c.dev"),
            ],
        ),
        ToolMessage(content="FETCH_ERROR: timeout", tool_call_id="c2"),
        AIMessage(content="final"),
    ]
    sources = collect_fetched_sources(messages)
    urls = [s["url"] for s in sources]
    assert urls == ["https://a.dev"]
    assert sources[0]["excerpt"].startswith("page text")


def test_build_sources_merges_annotations_and_fetched():
    messages = [
        AIMessage(
            content="",
            additional_kwargs={
                "annotations": [
                    {
                        "url_citation": {
                            "url": "https://ann.dev",
                            "title": "Ann",
                            "content": "ann excerpt",
                        }
                    }
                ]
            },
        ),
        AIMessage(content="", tool_calls=[_fetch_call("f1", "https://fetched.dev")]),
        ToolMessage(content="fetched body", tool_call_id="f1"),
        AIMessage(content="## FINDINGS\n- [S1] x | confidence: high\n"),
    ]
    sources, ref_order = build_sources(messages)
    assert [s["url"] for s in sources] == ["https://ann.dev", "https://fetched.dev"]
    assert ref_order == [{"url": "https://ann.dev"}, {"url": "https://fetched.dev"}]
    assert sources[0]["title"] == "Ann"
    assert sources[1]["excerpt"] == "fetched body"


def test_build_sources_empty_messages():
    sources, ref_order = build_sources([AIMessage(content="no tools here")])
    assert sources == []
    assert ref_order == []
