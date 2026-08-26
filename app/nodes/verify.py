import json
import re
import sys
from collections.abc import Callable

from pydantic import ValidationError

from app.backoff import call_with_backoff
from app.config import Settings
from app.nodes.parsing import sum_usage
from app.schemas import VerificationResult
from app.state import ResearchState

VERIFY_SYSTEM_PROMPT = """You are an independent evidence auditor.

Given a research question, findings with cited sources, decide whether the
evidence is sufficient to write a correct, well-cited answer.

Checklist:
- Does the evidence cover every part of the question?
- Are important claims backed by sources?
- Are there contradictions between credible sources?
- Are claims relying on weak/secondary-only evidence flagged?

Be strict but pragmatic: minor stylistic gaps are fine to pass.

Reply with JSON only, exactly this shape:
{"sufficient": bool, "missing_information": [str], "weak_claims": [str],
 "contradictory_claims": [str]}"""


def _parse_verdict(raw: str) -> VerificationResult:
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        raise ValueError(f"no JSON object in verifier reply: {raw[:200]}")
    try:
        return VerificationResult.model_validate_json(match.group(0))
    except ValidationError as exc:
        raise ValueError(f"verifier JSON invalid: {exc}") from exc


def make_verify_node(settings: Settings, model=None) -> Callable[[ResearchState], dict]:
    from app.models import get_model

    llm = model or get_model("verifier", settings)

    def verify(state: ResearchState) -> dict:
        human = (
            f"Question:\n{state['question']}\n\n"
            "Findings:\n"
            + json.dumps(state.get("findings") or [], ensure_ascii=False, indent=2)
            + "\n\nSources:\n"
            + json.dumps(state.get("sources") or [], ensure_ascii=False, indent=2)
        )
        reply = call_with_backoff(
            llm.invoke, [("system", VERIFY_SYSTEM_PROMPT), ("human", human)]
        )
        try:
            result = _parse_verdict(str(reply.content))
        except ValueError:
            retry_human = (
                human
                + "\n\nYour previous reply was not valid JSON. "
                "Reply with JSON only, exactly the specified shape."
            )
            reply = call_with_backoff(
                llm.invoke, [("system", VERIFY_SYSTEM_PROMPT), ("human", retry_human)]
            )
            try:
                result = _parse_verdict(str(reply.content))
            except ValueError as exc:
                print(f"[warn] verifier parse failed after retry: {exc}", file=sys.stderr)
                result = None
        tokens, cost = sum_usage([reply])
        usage_delta = {"total_tokens": tokens, "total_cost": cost}
        if result is None:
            return {
                "sufficient": False,
                "gaps": ["verifier parse error"],
                "contradictory_claims": [],
                **usage_delta,
            }
        return {
            "sufficient": result.sufficient,
            "gaps": result.missing_information,
            "weak_claims": result.weak_claims,
            "contradictory_claims": result.contradictory_claims,
            **usage_delta,
        }

    return verify
