import json
import sys
from collections.abc import Callable

from app.config import Settings
from app.nodes.parsing import (
    build_sources,
    collect_source_texts,
    count_total_searches,
    find_unknown_refs,
    ground_findings,
    map_refs_to_urls,
    parse_findings_block,
    sum_usage,
)
from app.state import ResearchState
from app.tools.events import ToolEventCallback


def _resolve_tool_event_writer():
    """Resolve the outer graph's stream writer, if we're running inside one.

    Must be called from the research node's own body, before invoking the
    (itself compiled-graph) research agent — see ToolEventCallback's
    docstring for why a lazy resolution from inside the agent call fails.
    """
    from langgraph.config import get_stream_writer

    try:
        return get_stream_writer()
    except RuntimeError:
        return None


def build_research_input(state: ResearchState) -> str:
    parts = [f"Original question:\n{state['question']}"]
    if state.get("iteration", 0) > 0:
        existing = state.get("findings") or []
        if existing:
            parts.append(
                "Existing findings:\n"
                + json.dumps(existing, ensure_ascii=False, indent=2)
            )
        gaps = state.get("gaps") or []
        if gaps:
            parts.append(
                "Missing evidence — research ONLY these points:\n- "
                + "\n- ".join(gaps)
            )
    parts.append(
        "Reminder: you MUST invoke the web_search tool at least once before "
        "fetching anything; then end your final reply with the ## FINDINGS "
        "block exactly as your instructions specify (one '- [Sn] claim | "
        "confidence: ... | quote: \"...\"' line per finding)."
    )
    return "\n\n".join(parts)


def _excerpts_from_quotes(sources: list, findings: list) -> list:
    """Use a grounded quote as its source's excerpt.

    A fetched page's default excerpt is its first 500 characters — usually
    navigation or a header — and the excerpt is all the answer writer (and
    the citation_support eval) ever see of a source. The quote is the passage
    that actually supports a claim.
    """
    quote_for: dict[str, str] = {}
    for finding in findings:
        if finding.get("grounded"):
            for url in finding.get("source_urls", []):
                quote_for.setdefault(url, finding["quote"])
    return [
        dict(source, excerpt=quote_for[source["url"]]) if source["url"] in quote_for else source
        for source in sources
    ]


def make_research_node(
    agent, settings: Settings, usage=None
) -> Callable[[ResearchState], dict]:
    def research(state: ResearchState) -> dict:
        prompt = build_research_input(state)
        writer = _resolve_tool_event_writer()
        # No call_with_backoff here: retrying agent.invoke would replay every
        # search and fetch so far. Rate limits are retried per model call
        # instead (ResearchChatOpenAI.rate_limit_attempts).
        result = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config={
                "recursion_limit": 50,
                "callbacks": [ToolEventCallback(writer=writer)],
            },
        )
        messages = result["messages"]
        message = messages[-1]
        parsed = parse_findings_block(str(message.content))
        sources, ref_map = build_sources(messages)
        findings = map_refs_to_urls(parsed.findings, parsed.refs, ref_map)
        findings, ungrounded = ground_findings(
            findings, parsed.quotes, collect_source_texts(messages)
        )
        sources = _excerpts_from_quotes(sources, findings)
        unknown = find_unknown_refs(findings, ref_map)
        unknown_set = set(unknown)
        seen_weak: set[str] = set()
        weak: list[str] = []

        def _note(item: str) -> None:
            if item not in seen_weak:
                seen_weak.add(item)
                weak.append(item)

        if not parsed.block_found:
            _note("research reply had no ## FINDINGS block")
            print(
                "[warn] research reply had no ## FINDINGS block",
                file=sys.stderr,
                flush=True,
            )
        for line in parsed.dropped:
            _note(f"unparseable FINDINGS line: {line}")
            print(f"[warn] unparseable FINDINGS line: {line}", file=sys.stderr, flush=True)
        for item in ungrounded:
            _note(item)
        for finding in findings:
            for u in finding.get("source_urls", []):
                if u in unknown_set and u.startswith("unresolved:"):
                    _note(
                        f"claim '{finding.get('claim', '')}' references "
                        f"unknown citation: {u.replace('unresolved:', '[ref] ')}"
                    )
        tokens, cost = sum_usage(messages)
        tool_tokens, tool_cost, tool_searches = (
            usage.drain() if usage is not None else (0, 0.0, 0)
        )
        return {
            "findings": findings,
            "sources": sources,
            "weak_claims": weak,
            "iteration": 1,
            "search_calls": count_total_searches(messages) + tool_searches,
            "total_tokens": tokens + tool_tokens,
            "total_cost": cost + tool_cost,
        }

    return research
