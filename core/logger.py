"""
core/logger.py
--------------
Centralised SQLite logging for every agent event.
"""

import sqlite3
import json
import os
from datetime import datetime, timezone


def _get_db() -> sqlite3.Connection:
    db_name = os.getenv("DB_NAME", "business_agent.db")
    conn = sqlite3.connect(db_name, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def log_message(conversation_id: str, role: str, content: str) -> None:
    conn = _get_db()
    try:
        conn.execute(
            "INSERT INTO conversations (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def log_tool_call(
    conversation_id: str,
    tool_name: str,
    inputs: dict,
    outputs: dict,
    activated: bool = True,
) -> None:
    conn = _get_db()
    try:
        conn.execute(
            """
            INSERT INTO tool_calls
                (conversation_id, tool_name, inputs, outputs, activated, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                tool_name,
                json.dumps(inputs, ensure_ascii=False),
                json.dumps(outputs, ensure_ascii=False),
                1 if activated else 0,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def log_chunk_retrieval(conversation_id: str, chunk_ids: list[int]) -> None:
    conn = _get_db()
    try:
        conn.execute(
            "INSERT INTO agent_logs (conversation_id, event_type, details, timestamp) VALUES (?, ?, ?, ?)",
            (
                conversation_id,
                "chunk_retrieval",
                json.dumps({"chunk_ids": chunk_ids}),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def log_agent_event(
    conversation_id: str, event_type: str, details: dict | str
) -> None:
    conn = _get_db()
    try:
        if isinstance(details, dict):
            details = json.dumps(details, ensure_ascii=False)
        conn.execute(
            "INSERT INTO agent_logs (conversation_id, event_type, details, timestamp) VALUES (?, ?, ?, ?)",
            (conversation_id, event_type, details, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def get_conversation_history(conversation_id: str, limit: int = 20) -> list[dict]:
    conn = _get_db()
    try:
        rows = conn.execute(
            """
            SELECT role, content, timestamp
            FROM conversations
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]
    finally:
        conn.close()
def log_llm_call(
    conversation_id: str,
    prompt: str,
    response: str,
    model: str = "unknown",
) -> None:
    conn = _get_db()
    try:
        details = {
            "prompt": prompt,
            "response": response,
            "model": model,
        }
        conn.execute(
            "INSERT INTO agent_logs (conversation_id, event_type, details, timestamp) VALUES (?, ?, ?, ?)",
            (
                conversation_id,
                "llm_call",
                json.dumps(details, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def log_db_write(
    conversation_id: str,
    table_name: str,
    operation: str,
    row_id: int | None = None,
    key_values: dict | None = None,
) -> None:
    """Log an individual SQL write operation (INSERT/UPDATE/DELETE) for audit tracing."""
    conn = _get_db()
    try:
        details = {
            "table": table_name,
            "operation": operation,
            "row_id": row_id,
            "key_values": key_values or {},
        }
        conn.execute(
            "INSERT INTO agent_logs (conversation_id, event_type, details, timestamp) VALUES (?, ?, ?, ?)",
            (
                conversation_id,
                "db_write",
                json.dumps(details, ensure_ascii=False, default=str),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()
