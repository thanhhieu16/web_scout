import json
from typing import Callable

from app.config import Settings
from app.nodes.parsing import (
    extract_url_citations,
    map_refs_to_urls,
    parse_findings_block,
    reconcile_sources,
)
from app.state import ResearchState
from app.tools.search import count_web_searches


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
    return "\n\n".join(parts)


def make_research_node(agent, settings: Settings) -> Callable[[ResearchState], dict]:
    def research(state: ResearchState) -> dict:
        prompt = build_research_input(state)
        result = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"recursion_limit": 50},
        )
        message = result["messages"][-1]
        findings, refs, _ = parse_findings_block(str(message.content))
        citations = extract_url_citations(message)
        findings, _ = map_refs_to_urls(findings, refs, citations)
        sources, unknown = reconcile_sources(findings, citations)
        unknown_set = set(unknown)
        seen_weak: set[str] = set()
        weak: list[str] = []
        for finding in findings:
            for u in finding.get("source_urls", []):
                if u in unknown_set and u.startswith("unresolved:"):
                    item = (
                        f"claim '{finding.get('claim', '')}' references "
                        f"unknown citation: {u.replace('unresolved:', '[ref] ')}"
                    )
                    if item not in seen_weak:
                        seen_weak.add(item)
                        weak.append(item)
        prior_weak = state.get("weak_claims") or []
        return {
            "findings": findings,
            "sources": sources,
            "weak_claims": prior_weak + weak,
            "iteration": state.get("iteration", 0) + 1,
            "search_calls": state.get("search_calls", 0)
            + count_web_searches(message),
        }

    return research
