"""
processing/pdf_processor.py
---------------------------
Reads a PDF file and splits it into meaningful chunks.
Import path changed from: pdf_processor → processing.pdf_processor
"""

import re
import os
import sqlite3
from pathlib import Path
import fitz  # PyMuPDF

DB_NAME = os.getenv("DB_NAME", "business_agent.db")


def process_pdf(pdf_path: str) -> list[dict]:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    pages = _extract_pages(str(path))
    chunks = _intelligent_chunk(pages)
    _save_chunks(chunks)
    return chunks


def load_chunks_from_db() -> list[dict]:
    db_name = os.getenv("DB_NAME", "business_agent.db")
    conn = sqlite3.connect(db_name, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM pdf_chunks ORDER BY chunk_index").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def chunks_already_stored() -> bool:
    db_name = os.getenv("DB_NAME", "business_agent.db")
    conn = sqlite3.connect(db_name, check_same_thread=False)
    try:
        count = conn.execute("SELECT COUNT(*) FROM pdf_chunks").fetchone()[0]
        return count > 0
    finally:
        conn.close()


def _extract_pages(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        structured_blocks = []
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                line_text = ""
                is_bold = False
                font_size = 0.0
                for span in line.get("spans", []):
                    line_text += span.get("text", "")
                    if "Bold" in span.get("font", "") or "bold" in span.get("font", ""):
                        is_bold = True
                    font_size = max(font_size, span.get("size", 0.0))

                line_text = line_text.strip()
                if not line_text:
                    continue

                is_header = is_bold or font_size >= 13.0
                is_bullet = bool(re.match(r"^[\•\-\*\◆\►\▶\–\—\d]+[\.\)]\s+", line_text))

                structured_blocks.append(
                    {
                        "text": line_text,
                        "is_header": is_header,
                        "is_bullet": is_bullet,
                        "font_size": font_size,
                    }
                )
        pages.append({"page_num": page_num, "blocks": structured_blocks})
    doc.close()
    return pages


def _intelligent_chunk(pages: list[dict]) -> list[dict]:
    chunks = []
    chunk_index = 0
    current_section = "Introduction"
    current_page = 1
    buffer_lines: list[str] = []
    in_bullet_block = False

    def flush(section: str, page: int) -> None:
        nonlocal chunk_index, buffer_lines
        text = " ".join(buffer_lines).strip()
        if text:
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "page_num": page,
                    "section_title": section,
                    "text": text,
                    "topic_tags": None,
                }
            )
            chunk_index += 1
        buffer_lines = []

    for page in pages:
        current_page = page["page_num"]
        for block in page["blocks"]:
            text = block["text"]

            if block["is_header"]:
                flush(current_section, current_page)
                in_bullet_block = False
                current_section = text
                buffer_lines = [f"[SECTION: {text}]"]

            elif block["is_bullet"]:
                if not in_bullet_block:
                    if buffer_lines:
                        flush(current_section, current_page)
                    in_bullet_block = True
                buffer_lines.append(text)

            else:
                if in_bullet_block:
                    flush(current_section, current_page)
                    in_bullet_block = False
                buffer_lines.append(text)

                if len(text) < 80 and buffer_lines:
                    flush(current_section, current_page)

    flush(current_section, current_page)
    return chunks


def _save_chunks(chunks: list[dict]) -> None:
    db_name = os.getenv("DB_NAME", "business_agent.db")
    conn = sqlite3.connect(db_name, check_same_thread=False)
    try:
        conn.execute("DELETE FROM pdf_chunks")
        for chunk in chunks:
            conn.execute(
                "INSERT INTO pdf_chunks (chunk_index, page_num, section_title, text, topic_tags) VALUES (?, ?, ?, ?, ?)",
                (
                    chunk["chunk_index"],
                    chunk["page_num"],
                    chunk["section_title"],
                    chunk["text"],
                    chunk.get("topic_tags"),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def get_pdf_summary(chunks: list[dict], max_chars: int = 3000) -> str:
    texts = [c["text"] for c in chunks[:30]]
    combined = " ".join(texts)
    return combined[:max_chars]
