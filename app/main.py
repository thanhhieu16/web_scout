import argparse
import sys
from datetime import datetime
from pathlib import Path

from langchain_core.tracers.langchain import LangChainTracer

from app.agent import build_research_agent
from app.config import MODEL_CHOICES, get_settings, override_model
from app.graph import build_graph
from app.nodes.parsing import (
    build_sources,
    count_total_searches,
    find_unknown_refs,
    map_refs_to_urls,
    parse_findings_block,
)
from app.usage import UsageCollector


def require_openrouter_key(settings=None) -> None:
    s = settings or get_settings()
    if not s.openrouter_api_key:
        raise SystemExit(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill it in."
        )


def run_question(question: str, agent=None, settings=None, usage=None) -> dict:
    s = settings or get_settings()
    if usage is None:
        usage = UsageCollector()
    deep = agent or build_research_agent(s, usage=usage)
    result = deep.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"recursion_limit": 50},
    )
    messages = result["messages"]
    message = messages[-1]
    parsed = parse_findings_block(str(message.content))
    sources, ref_order = build_sources(messages)
    findings = map_refs_to_urls(parsed.findings, parsed.refs, ref_order)
    unknown = find_unknown_refs(findings, ref_order)
    _, _, tool_searches = usage.drain()
    searches = count_total_searches(messages) + tool_searches
    if unknown:
        print(f"[warn] {len(unknown)} uncited URL(s) ignored", file=sys.stderr)
    return {
        "answer": parsed.narrative.strip(),
        "sources": sources,
        "findings": findings,
        "search_calls": searches,
    }


def stream_pipeline(question: str, graph=None, max_iterations: int | None = None):
    """Drive the LangGraph pipeline, yielding progress then the final result.

    Yields ("status", node_name) once per completed node, in order, then yields
    exactly one ("result", out_dict) as the last item.
    """
    if graph is None:
        require_openrouter_key()
    g = graph or build_graph()
    s = get_settings()
    mi = max_iterations if max_iterations is not None else s.max_iterations
    state = {"question": question, "iteration": 0, "max_iterations": mi}
    final = dict(state)
    tracer = LangChainTracer()
    for mode, chunk in g.stream(
        state, stream_mode=["updates", "values"], config={"callbacks": [tracer]}
    ):
        if mode == "updates":
            for node in chunk:
                yield ("status", node)
        else:
            final = chunk
    try:
        trace_url = tracer.get_run_url()
        trace_run_id = str(tracer.latest_run.id) if tracer.latest_run else None
    except Exception:
        trace_url = None
        trace_run_id = None
    yield (
        "result",
        {
            "answer": final.get("answer", ""),
            "sources": final.get("sources", []),
            "findings": final.get("findings", []),
            "search_calls": final.get("search_calls", 0),
            "sufficient": final.get("sufficient", False),
            "iteration": final.get("iteration", 0),
            "total_tokens": final.get("total_tokens", 0),
            "total_cost": round(final.get("total_cost", 0.0), 4),
            "trace_url": trace_url,
            "trace_run_id": trace_run_id,
        },
    )


def run_pipeline(question: str, graph=None, max_iterations: int | None = None) -> dict:
    out = None
    for kind, payload in stream_pipeline(question, graph=graph, max_iterations=max_iterations):
        if kind == "status":
            print(f"[{payload}] ...", flush=True)
        else:
            out = payload
    return out


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


def render_report_markdown(question: str, out: dict) -> str:
    lines = [
        "# WebScout Report",
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
    return "\n".join(lines)


def write_report(question: str, out: dict, path: str) -> None:
    Path(path).write_text(render_report_markdown(question, out), encoding="utf-8")


def _print_models(current: str) -> None:
    print("Available models (any OpenRouter slug also works):")
    for i, slug in enumerate(MODEL_CHOICES, 1):
        mark = " *" if slug == current else ""
        print(f"  {i}. {slug}{mark}")
    print("  * = currently selected")


def _resolve_choice(raw: str) -> str:
    """Accept either a shortlist index or a raw slug."""
    if raw.isdigit() and 1 <= int(raw) <= len(MODEL_CHOICES):
        return MODEL_CHOICES[int(raw) - 1]
    return raw


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="webscout")
    parser.add_argument("question", nargs="*", help="research question")
    parser.add_argument(
        "--out",
        default=None,
        help="write a markdown report to this path (one-shot mode)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="OpenRouter model slug (or shortlist number) for every role",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="print the model shortlist and exit",
    )
    args = parser.parse_args(argv)
    if args.list_models:
        _print_models(get_settings().researcher.model)
        return
    if args.model:
        chosen = _resolve_choice(args.model)
        override_model(chosen)
        print(f"[model] {chosen}")
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
        if q.split()[0].lower() in {"/model", "/models"}:
            parts = q.split(maxsplit=1)
            if len(parts) == 1:
                _print_models(get_settings().researcher.model)
            else:
                chosen = _resolve_choice(parts[1].strip())
                override_model(chosen)
                print(f"[model] {chosen}")
            continue
        _print_result(run_pipeline(q))


if __name__ == "__main__":
    main()
