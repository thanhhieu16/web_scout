import asyncio

from acp import (
    PROTOCOL_VERSION,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    plan_entry,
    run_agent,
    update_agent_message_text,
    update_plan,
)

from app.config import get_settings
from app.main import require_openrouter_key
from app.turn import run_chat_turn
from web import store

_STAGE_LABELS = {"research": "Research", "verify": "Verify", "answer": "Answer"}


def _stage_label(node: str, occurrence: int) -> str:
    base = _STAGE_LABELS.get(node, node)
    return base if occurrence == 1 else f"{base} — vòng {occurrence}"


def _db_path() -> str:
    path = get_settings().conversations_db_path
    store.init_db(path)
    return path


class WebScoutAcpAgent:
    """ACP agent exposing WebScout's research pipeline over stdio, for Zed."""

    def __init__(self) -> None:
        self.conn = None
        self._cancelled: set[str] = set()

    def on_connect(self, conn) -> None:
        self.conn = conn

    async def initialize(
        self, protocol_version, client_capabilities=None, client_info=None, **kwargs
    ) -> InitializeResponse:
        return InitializeResponse(protocol_version=min(protocol_version, PROTOCOL_VERSION))

    async def new_session(
        self, cwd, additional_directories=None, mcp_servers=None, **kwargs
    ) -> NewSessionResponse:
        conversation_id = store.create_conversation(_db_path())
        return NewSessionResponse(session_id=str(conversation_id))

    async def cancel(self, session_id, **kwargs) -> None:
        self._cancelled.add(session_id)

    async def prompt(self, session_id, prompt, **kwargs) -> PromptResponse:
        question = "\n".join(
            block.text for block in prompt if getattr(block, "type", None) == "text"
        )
        if not question.strip():
            await self.conn.session_update(
                session_id,
                update_agent_message_text("(không nhận được nội dung câu hỏi dạng văn bản)"),
            )
            return PromptResponse(stop_reason="end_turn")

        db_path = _db_path()
        conversation_id = int(session_id)

        entries = [plan_entry(_stage_label("research", 1), status="in_progress")]
        await self.conn.session_update(session_id, update_plan(entries))
        occurrences = {"research": 1}
        bootstrap_open = True

        answer_text = "(không có câu trả lời)"
        stop_reason = "end_turn"
        turn_events = run_chat_turn(db_path, conversation_id, question)
        sentinel = object()
        while True:
            item = await asyncio.to_thread(next, turn_events, sentinel)
            if item is sentinel:
                break
            if session_id in self._cancelled:
                self._cancelled.discard(session_id)
                stop_reason = "cancelled"
                answer_text = "(đã hủy)"
                break
            kind, payload = item
            if kind == "status":
                # stream_pipeline only reports node COMPLETIONS, never starts, and
                # verify can loop back to research an unknown number of times, so
                # the next node to run can't be predicted here. Every event after
                # the bootstrap entry simply appends a new completed entry.
                node = payload["node"]
                n = occurrences.get(node, 0) + 1
                occurrences[node] = n
                label = _stage_label(node, n)
                if bootstrap_open:
                    entries[-1] = plan_entry(label, status="completed")
                    bootstrap_open = False
                else:
                    entries.append(plan_entry(label, status="completed"))
                await self.conn.session_update(session_id, update_plan(entries))
            elif kind == "result":
                answer_text = payload["answer"]
            else:
                answer_text = f"⚠ Lỗi: {payload['message']}"

        # A turn that fails or is cancelled before its next status event leaves the
        # bootstrap (or last-appended) entry stuck at "in_progress" forever in Zed's
        # plan panel. ACP's PlanEntryStatus has no "failed" value, so the best honest
        # signal available is to close it out — the answer_text (or the cancellation)
        # already communicates what actually happened.
        if entries and entries[-1].status == "in_progress":
            entries[-1] = plan_entry(entries[-1].content, status="completed")
            await self.conn.session_update(session_id, update_plan(entries))

        await self.conn.session_update(session_id, update_agent_message_text(answer_text))
        return PromptResponse(stop_reason=stop_reason)


def main() -> None:
    """Entry point for the `webscout-acp` console script."""
    require_openrouter_key()
    asyncio.run(run_agent(WebScoutAcpAgent()))


if __name__ == "__main__":
    main()
