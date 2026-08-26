import json
from typing import Callable

from app.backoff import call_with_backoff
from app.config import Settings
from app.nodes.parsing import (
    build_sources,
    count_total_searches,
    map_refs_to_urls,
    parse_findings_block,
    reconcile_sources,
)
from app.state import ResearchState


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
        "confidence: ...' line per finding)."
    )
    return "\n\n".join(parts)


def make_research_node(agent, settings: Settings) -> Callable[[ResearchState], dict]:
    def research(state: ResearchState) -> dict:
        prompt = build_research_input(state)
        result = call_with_backoff(
            agent.invoke,
            {"messages": [{"role": "user", "content": prompt}]},
            config={"recursion_limit": 50},
        )
        messages = result["messages"]
        message = messages[-1]
        findings, refs, _ = parse_findings_block(str(message.content))
        sources, ref_order = build_sources(messages)
        findings, _ = map_refs_to_urls(findings, refs, ref_order)
        _, unknown = reconcile_sources(findings, ref_order)
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
            + count_total_searches(messages),
        }

    return research
