import json
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import MODEL_CHOICES, get_settings, override_model
from app.conversation import condense_question
from app.graph import build_graph
from app.main import render_report_markdown, stream_pipeline

STATIC_DIR = Path(__file__).parent / "static"


class HistoryTurn(BaseModel):
    question: str
    answer: str


class ChatRequest(BaseModel):
    question: str
    history: list[HistoryTurn] = []
    model: str | None = None
    max_iterations: int | None = None


class ReportRequest(BaseModel):
    question: str
    out: dict


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


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/api/chat")
def chat(body: ChatRequest):
    if body.model:
        override_model(body.model)

    def gen():
        try:
            history = [t.model_dump() for t in body.history]
            question = condense_question(history, body.question)
            graph = build_graph()
            for kind, payload in stream_pipeline(
                question, graph=graph, max_iterations=body.max_iterations
            ):
                yield _sse(kind, {"node": payload} if kind == "status" else payload)
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
