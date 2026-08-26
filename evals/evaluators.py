import json
import re

from langsmith.evaluation import EvaluationResult

from app.models import get_model

_CITE_RE = re.compile(r"\[(\d+)\]")


def correctness_evaluator(run, example) -> EvaluationResult:
    question = (run.inputs or {}).get("question", "")
    answer = (run.outputs or {}).get("answer", "")
    notes = (example.outputs or {}).get("reference_notes", "")
    judge = get_model("verifier")
    verdict = judge.invoke(
        [
            (
                "system",
                "You grade research answers. Return JSON only: "
                '{"score": <1-5>, "reason": "<short>"} where 5 means fully '
                "correct, well-scoped and honest about uncertainty.",
            ),
            (
                "human",
                f"Question:\n{question}\n\nReference notes:\n{notes}\n\n"
                f"Answer:\n{answer}",
            ),
        ]
    )
    try:
        parsed = json.loads(re.search(r"\{[\s\S]*\}", str(verdict.content)).group(0))
        score = float(parsed.get("score", 0)) / 5.0
        reason = str(parsed.get("reason", ""))
    except Exception as exc:
        score, reason = 0.0, f"judge-parse-error: {exc}"
    return EvaluationResult(key="correctness", score=score, comment=reason)


def citation_support_evaluator(run, example) -> EvaluationResult:
    outputs = run.outputs or {}
    answer = outputs.get("answer", "")
    sources = outputs.get("sources") or []
    refs = {int(n) for n in _CITE_RE.findall(answer)}
    unresolved = [n for n in refs if n < 1 or n > len(sources)]
    if not refs or unresolved:
        base = 0.0
    else:
        base = 1.0
    return EvaluationResult(
        key="citation_support",
        score=base,
        comment=f"unresolved={unresolved}, refs={sorted(refs)}",
    )


def metrics_evaluator(run, example) -> list[EvaluationResult]:
    outputs = run.outputs or {}
    latency = 0.0
    if getattr(run, "start_time", None) and getattr(run, "end_time", None):
        latency = (run.end_time - run.start_time).total_seconds()
    values = {
        "latency_s": latency,
        "total_tokens": int(getattr(run, "total_tokens", 0) or 0),
        "search_calls": int(outputs.get("search_calls", 0) or 0),
        "num_sources": len(outputs.get("sources") or []),
    }
    return [
        EvaluationResult(key=k, score=float(v), value=float(v))
        for k, v in values.items()
    ]
