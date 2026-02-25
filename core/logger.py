"""
core/logger.py
--------------
Centralised SQLite logging for every agent event.
"""

import sqlite3
import json
import os
from datetime import datetime


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
            (conversation_id, role, content, datetime.utcnow().isoformat()),
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
                datetime.utcnow().isoformat(),
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
                datetime.utcnow().isoformat(),
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
            (conversation_id, event_type, details, datetime.utcnow().isoformat()),
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
