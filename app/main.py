import argparse
import sys
from datetime import datetime
from pathlib import Path

from app.agent import build_research_agent
from app.config import get_settings
from app.graph import build_graph
from app.nodes.parsing import (
    build_sources,
    count_total_searches,
    map_refs_to_urls,
    parse_findings_block,
    reconcile_sources,
)


def require_openrouter_key(settings=None) -> None:
    s = settings or get_settings()
    if not s.openrouter_api_key:
        raise SystemExit(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill it in."
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
    sources, ref_order = build_sources(messages)
    findings, _ = map_refs_to_urls(findings, refs, ref_order)
    _, unknown = reconcile_sources(findings, ref_order)
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
    if graph is None:
        require_openrouter_key()
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
        "total_tokens": final.get("total_tokens", 0),
        "total_cost": round(final.get("total_cost", 0.0), 4),
    }


def _print_result(out: dict) -> None:
    print("\n=== ANSWER ===\n")
    print(out["answer"])
    if out["sources"]:
        print("\n=== SOURCES ===")
        for i, src in enumerate(out["sources"], 1):
            print(f"[{i}] {src['title']} — {src['url']}")
    print(
        "\n=== METRICS ===\n"
        f"iterations: {out.get('iteration', 0)} | "
        f"searches: {out.get('search_calls', 0)} | "
        f"sources: {len(out.get('sources', []))} | "
        f"tokens: {out.get('total_tokens', 0)} | "
        f"est_cost: ${out.get('total_cost', 0.0):.4f}"
    )


def write_report(question: str, out: dict, path: str) -> None:
    lines = [
        f"# WebScout Report",
        "",
        f"- **Question:** {question}",
        f"- **Generated:** {datetime.now().isoformat(timespec='seconds')}",
        f"- **Sufficient:** {out.get('sufficient', False)} | "
        f"Iterations: {out.get('iteration', 0)} | "
        f"Searches: {out.get('search_calls', 0)} | "
        f"Tokens: {out.get('total_tokens', 0)} | "
        f"Est. cost: ${out.get('total_cost', 0.0):.4f}",
        "",
        "## Answer",
        "",
        out.get("answer", ""),
        "",
    ]
    findings = out.get("findings") or []
    if findings:
        lines += ["## Findings", ""]
        for f in findings:
            refs = ", ".join(f.get("source_urls") or [])
            lines.append(
                f"- ({f.get('confidence', '?')}) {f.get('claim', '')}"
                + (f" — {refs}" if refs else "")
            )
        lines.append("")
    sources = out.get("sources") or []
    if sources:
        lines += ["## Sources", ""]
        for i, s in enumerate(sources, 1):
            lines.append(f"{i}. [{s.get('title', '')}]({s.get('url', '')})")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="webscout")
    parser.add_argument("question", nargs="*", help="research question")
    parser.add_argument("--out", default=None, help="write a markdown report to this path (one-shot mode)")
    args = parser.parse_args(argv)
    require_openrouter_key()
    if args.question:
        question = " ".join(args.question)
        out = run_pipeline(question)
        _print_result(out)
        if args.out:
            write_report(question, out, args.out)
            print(f"[report] written to {args.out}")
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
