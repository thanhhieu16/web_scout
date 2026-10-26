from collections.abc import Callable

from app.backoff import call_with_backoff
from app.config import Settings
from app.nodes.parsing import sum_usage
from app.state import ResearchState

ANSWER_SYSTEM_PROMPT = """You are WebScout's answer writer.

Write the final answer to the question using ONLY the verified findings and
sources provided. Requirements:
- Reply in the same language as the question.
- Cite claims inline as [n], using the source numbers listed after each
  finding; they match the numbered source list given.
- Distinguish fact from inference. State uncertainty when present.
- Use findings listed under "Uncertain findings" only with explicit hedging,
  or leave them out.
- Do not add unsupported claims. Do not perform new research."""


def _finding_line(finding: dict, number_for: dict[str, int]) -> str | None:
    """Render one finding with its sources as [n] numbers, or None when no
    cited URL is in the source list (every ref unresolved) — such a claim has
    nothing to cite and must not reach the answer."""
    numbers = sorted(
        {number_for[u] for u in finding.get("source_urls") or [] if u in number_for}
    )
    if not numbers:
        return None
    line = (
        f"- ({finding.get('confidence')}) {finding.get('claim')} "
        + "".join(f"[{n}]" for n in numbers)
    )
    if finding.get("grounded") and finding.get("quote"):
        line += f'\n  quote: "{finding["quote"]}"'
    return line


def render_findings(findings: list, sources: list) -> str:
    number_for = {s.get("url"): i for i, s in enumerate(sources, 1)}
    solid: list[str] = []
    uncertain: list[str] = []
    for finding in findings:
        line = _finding_line(finding, number_for)
        if line is None:
            continue
        (uncertain if finding.get("confidence") == "low" else solid).append(line)
    text = "Findings:\n" + ("\n".join(solid) or "(none)")
    if uncertain:
        text += (
            "\n\nUncertain findings (low confidence — hedge or omit):\n"
            + "\n".join(uncertain)
        )
    return text


def make_answer_node(settings: Settings, model=None) -> Callable[[ResearchState], dict]:
    from app.models import get_model

    llm = model or get_model("answer", settings)

    def answer(state: ResearchState) -> dict:
        lines = [
            f"[{i}] {s.get('title', '')} — {s.get('url', '')}\n    excerpt: {s.get('excerpt', '')}"
            for i, s in enumerate(state.get("sources") or [], 1)
        ]
        sections = [
            f"Question:\n{state.get('question', '')}",
            render_findings(state.get("findings") or [], state.get("sources") or []),
            "Numbered sources:\n" + ("\n".join(lines) or "(none)"),
        ]
        exhausted = (
            state.get("iteration", 0) >= state.get("max_iterations", 3)
            and not state.get("sufficient", False)
        )
        if exhausted:
            sections.append(
                "The verification budget was exhausted without full confidence. "
                "Explicitly state the remaining uncertainty in the answer."
            )
        reply = call_with_backoff(
            llm.invoke,
            [("system", ANSWER_SYSTEM_PROMPT), ("human", "\n\n".join(sections))],
        )
        tokens, cost = sum_usage([reply])
        return {
            "answer": str(reply.content).strip(),
            "total_tokens": tokens,
            "total_cost": cost,
        }

    return answer
