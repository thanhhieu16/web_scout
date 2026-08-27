import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_TITLE = "Cuộc hội thoại mới"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=5)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                question TEXT NOT NULL,
                answer_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def create_conversation(db_path: str, title: str = DEFAULT_TITLE) -> int:
    now = _now()
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)",
            (title, now, now),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def list_conversations(db_path: str) -> list[dict]:
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, title, updated_at FROM conversations ORDER BY updated_at DESC, id DESC"
        ).fetchall()
        return [{"id": r[0], "title": r[1], "updated_at": r[2]} for r in rows]
    finally:
        conn.close()


def get_conversation(db_path: str, conversation_id: int) -> dict | None:
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT id, title FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if row is None:
            return None
        messages = conn.execute(
            "SELECT question, answer_json FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        ).fetchall()
        return {
            "id": row[0],
            "title": row[1],
            "messages": [{"question": q, "out": json.loads(a)} for q, a in messages],
        }
    finally:
        conn.close()


def append_message(db_path: str, conversation_id: int, question: str, out: dict) -> None:
    now = _now()
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT title FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"conversation {conversation_id} not found")
        count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (conversation_id,)
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO messages (conversation_id, question, answer_json, created_at) "
            "VALUES (?, ?, ?, ?)",
            (conversation_id, question, json.dumps(out), now),
        )
        if count == 0 and row[0] == DEFAULT_TITLE:
            title = question[:40] + ("…" if len(question) > 40 else "")
            conn.execute(
                "UPDATE conversations SET updated_at = ?, title = ? WHERE id = ?",
                (now, title, conversation_id),
            )
        else:
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
        conn.commit()
    finally:
        conn.close()


def rename_conversation(db_path: str, conversation_id: int, title: str) -> bool:
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id)
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def delete_conversation(db_path: str, conversation_id: int) -> bool:
    conn = _connect(db_path)
    try:
        cur = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()
