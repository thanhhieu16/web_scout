from app.config import Settings, get_settings
from app.models import get_model

_MAX_HISTORY_TURNS = 3

_CONDENSE_SYSTEM_PROMPT = (
    "You rewrite a follow-up question into a standalone question, using the "
    "conversation so far. Preserve the user's language and intent exactly. "
    "Output ONLY the rewritten question — no preamble, no quotes, no explanation. "
    "If the latest question is already standalone, return it unchanged."
)


def condense_question(
    history: list[dict],
    question: str,
    settings: Settings | None = None,
    model=None,
) -> str:
    """Rewrite `question` into a standalone question using prior turns.

    Returns `question` unchanged (no model call) when history is empty, and
    falls back to `question` unchanged if the rewrite call raises for any
    reason — a broken rewrite must never block the chat turn.
    """
    if not history:
        return question
    active_model = model or get_model("verifier", settings or get_settings())
    turns = history[-_MAX_HISTORY_TURNS:]
    transcript = "\n\n".join(
        f"Q: {t.get('question', '')}\nA: {t.get('answer', '')}" for t in turns
    )
    try:
        result = active_model.invoke(
            [
                ("system", _CONDENSE_SYSTEM_PROMPT),
                ("human", f"Conversation so far:\n{transcript}\n\nLatest question:\n{question}"),
            ]
        )
        rewritten = str(result.content).strip()
        return rewritten or question
    except Exception:
        return question
