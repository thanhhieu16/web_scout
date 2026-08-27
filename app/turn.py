import sys
from collections.abc import Iterator

from app.conversation import condense_question
from app.graph import build_graph
from app.main import stream_pipeline
from web import store


def run_chat_turn(
    db_path: str,
    conversation_id: int,
    question: str,
    max_iterations: int | None = None,
) -> Iterator[tuple[str, dict]]:
    """Run one conversation turn, yielding progress then the final result.

    Mirrors `stream_pipeline`'s contract: yields ("status", {"node": name})
    zero or more times, then exactly one ("result", out_dict) or one
    ("error", {"message": str}) as the last item. On "result", the turn is
    persisted via `store.append_message` before it's yielded. On "error",
    nothing is persisted — a failed turn must never appear in history.

    Raises KeyError if `conversation_id` doesn't exist in `db_path` (only
    once the generator is actually iterated, since this is a generator
    function — nothing in this body runs until the first `next()`).
    """
    conversation = store.get_conversation(db_path, conversation_id)
    if conversation is None:
        raise KeyError(f"conversation {conversation_id} not found")
    try:
        history = [
            {"question": m["question"], "answer": m["out"].get("answer", "")}
            for m in conversation["messages"]
        ]
        condensed = condense_question(history, question)
        graph = build_graph()
        for kind, payload in stream_pipeline(
            condensed, graph=graph, max_iterations=max_iterations
        ):
            if kind == "status":
                yield ("status", {"node": payload})
            else:
                try:
                    store.append_message(db_path, conversation_id, question, payload)
                except Exception as persist_exc:
                    print(f"[warn] failed to persist turn: {persist_exc}", file=sys.stderr)
                yield ("result", payload)
    except Exception as exc:
        yield ("error", {"message": str(exc)})
