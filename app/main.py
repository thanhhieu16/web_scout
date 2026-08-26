import argparse
import sys

from app.agent import build_research_agent
from app.config import get_settings
from app.graph import build_graph
from app.nodes.parsing import (
    collect_citations,
    count_total_searches,
    map_refs_to_urls,
    parse_findings_block,
    reconcile_sources,
)


def run_question(question: str, agent=None, settings=None) -> dict:
    s = settings or get_settings()
    deep = agent or build_research_agent(s)
    result = deep.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"recursion_limit": 50},
    )
    messages = result["messages"]
    message = messages[-1]
    findings, refs, narrative = parse_findings_block(str(message.content))
    citations = collect_citations(messages)
    findings, _ = map_refs_to_urls(findings, refs, citations)
    sources, unknown = reconcile_sources(findings, citations)
    searches = count_total_searches(messages)
    if unknown:
        print(f"[warn] {len(unknown)} uncited URL(s) ignored", file=sys.stderr)
    return {
        "answer": narrative.strip(),
        "sources": sources,
        "findings": findings,
        "search_calls": searches,
    }


def run_pipeline(question: str, graph=None) -> dict:
    g = graph or build_graph()
    s = get_settings()
    state = {"question": question, "iteration": 0, "max_iterations": s.max_iterations}
    final = dict(state)
    for update in g.stream(state, stream_mode="updates"):
        for node, delta in update.items():
            print(f"[{node}] ...", flush=True)
            if isinstance(delta, dict):
                final.update(delta)
    return {
        "answer": final.get("answer", ""),
        "sources": final.get("sources", []),
        "findings": final.get("findings", []),
        "search_calls": final.get("search_calls", 0),
        "sufficient": final.get("sufficient", False),
        "iteration": final.get("iteration", 0),
    }


def _print_result(out: dict) -> None:
    print("\n=== ANSWER ===\n")
    print(out["answer"])
    if out["sources"]:
        print("\n=== SOURCES ===")
        for i, src in enumerate(out["sources"], 1):
            print(f"[{i}] {src['title']} — {src['url']}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="webscout")
    parser.add_argument("question", nargs="*", help="research question")
    args = parser.parse_args(argv)
    if args.question:
        _print_result(run_pipeline(" ".join(args.question)))
        return
    while True:
        try:
            q = input("\nwebscout> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() in {"exit", "quit"}:
            break
        _print_result(run_pipeline(q))


if __name__ == "__main__":
    main()
