from collections.abc import Callable

from app.backoff import call_with_backoff
from app.config import Settings
from app.nodes.parsing import sum_usage
from app.state import ResearchState

ANSWER_SYSTEM_PROMPT = """You are WebScout's answer writer.

Write the final answer to the question using ONLY the verified findings and
sources provided. Requirements:
- Reply in the same language as the question.
- Cite claims inline as [n] where n matches the numbered source list given.
- Distinguish fact from inference. State uncertainty when present.
- Do not add unsupported claims. Do not perform new research."""


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
            "Findings:\n"
            + "\n".join(
                f"- ({f.get('confidence')}) {f.get('claim')} "
                f"[refs: {', '.join(f.get('source_urls') or [])}]"
                for f in (state.get("findings") or [])
            ),
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
            "total_tokens": state.get("total_tokens", 0) + tokens,
            "total_cost": round(state.get("total_cost", 0.0) + cost, 6),
        }

    return answer
