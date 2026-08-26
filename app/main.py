import argparse
import sys

from app.agent import build_research_agent
from app.config import get_settings
from app.nodes.parsing import (
    extract_url_citations,
    map_refs_to_urls,
    parse_findings_block,
    reconcile_sources,
)
from app.tools.search import count_web_searches


def run_question(question: str, agent=None, settings=None) -> dict:
    s = settings or get_settings()
    deep = agent or build_research_agent(s)
    result = deep.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"recursion_limit": 50},
    )
    message = result["messages"][-1]
    findings, refs, narrative = parse_findings_block(str(message.content))
    citations = extract_url_citations(message)
    findings, _ = map_refs_to_urls(findings, refs, citations)
    sources, unknown = reconcile_sources(findings, citations)
    searches = count_web_searches(message)
    if unknown:
        print(f"[warn] {len(unknown)} uncited URL(s) ignored", file=sys.stderr)
    return {
        "answer": narrative.strip(),
        "sources": sources,
        "findings": findings,
        "search_calls": searches,
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
        _print_result(run_question(" ".join(args.question)))
        return
    while True:
        try:
            q = input("\nwebscout> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() in {"exit", "quit"}:
            break
        _print_result(run_question(q))


if __name__ == "__main__":
    main()
