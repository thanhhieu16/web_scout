import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import MODEL_CHOICES, get_settings, override_model
from app.conversation import condense_question
from app.graph import build_graph
from app.main import render_report_markdown, stream_pipeline
from web import store

STATIC_DIR = Path(__file__).parent / "static"


def _db_path() -> str:
    """Read the configured DB path and ensure its schema exists.

    Called per-request (not once at import time) so tests can point
    CONVERSATIONS_DB_PATH at a tmp file via monkeypatch + cache_clear()
    before the first conversation route runs, instead of the real
    data/webscout.db getting created as a side effect of merely
    importing this module.
    """
    path = get_settings().conversations_db_path
    store.init_db(path)
    return path


class ChatRequest(BaseModel):
    conversation_id: int
    question: str
    model: str | None = None
    max_iterations: int | None = Field(default=None, ge=1, le=10)


class ReportRequest(BaseModel):
    question: str
    out: dict


class RenameRequest(BaseModel):
    title: str


app = FastAPI(title="WebScout Chat")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/api/models")
def list_models():
    s = get_settings()
    return {
        "choices": list(MODEL_CHOICES),
        "current": s.researcher.model,
        "key_configured": bool(s.openrouter_api_key),
    }


@app.get("/api/conversations")
def list_conversations():
    return store.list_conversations(_db_path())


@app.post("/api/conversations")
def create_conversation():
    db_path = _db_path()
    conv_id = store.create_conversation(db_path)
    return {"id": conv_id, "title": store.DEFAULT_TITLE}


@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: int):
    conv = store.get_conversation(_db_path(), conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return conv


@app.patch("/api/conversations/{conversation_id}")
def rename_conversation(conversation_id: int, body: RenameRequest):
    title = body.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="title cannot be empty")
    ok = store.rename_conversation(_db_path(), conversation_id, title)
    if not ok:
        raise HTTPException(status_code=404, detail="conversation not found")
    return {"id": conversation_id, "title": title}


@app.delete("/api/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: int):
    ok = store.delete_conversation(_db_path(), conversation_id)
    if not ok:
        raise HTTPException(status_code=404, detail="conversation not found")
    return Response(status_code=204)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/api/chat")
def chat(body: ChatRequest):
    db_path = _db_path()
    conversation = store.get_conversation(db_path, body.conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation not found")

    if body.model:
        override_model(body.model)

    def gen():
        try:
            history = [
                {"question": m["question"], "answer": m["out"].get("answer", "")}
                for m in conversation["messages"]
            ]
            question = condense_question(history, body.question)
            graph = build_graph()
            for kind, payload in stream_pipeline(
                question, graph=graph, max_iterations=body.max_iterations
            ):
                if kind == "status":
                    yield _sse("status", {"node": payload})
                else:
                    store.append_message(db_path, body.conversation_id, body.question, payload)
                    yield _sse("result", payload)
        except Exception as exc:
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/report")
def report(body: ReportRequest):
    md = render_report_markdown(body.question, body.out)
    return Response(
        md,
        media_type="text/markdown",
        headers={"Content-Disposition": 'attachment; filename="report.md"'},
    )


def main() -> None:
    """Entry point for the `webscout-web` console script."""
    import uvicorn

    uvicorn.run("web.server:app", host="127.0.0.1", port=8000, reload=False)
