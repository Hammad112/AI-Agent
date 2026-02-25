"""
processing/pdf_processor.py
----------------------------
PDF and text file ingestion with intelligent chunking.
Supports: .pdf (via PyMuPDF), .md, .txt files.
"""

import os
import re
import sqlite3


def process_pdf(file_path: str) -> list[dict]:
    """
    Process a business description file (PDF, MD, or TXT).
    Returns list of chunk dicts.
    """
    file_path = str(file_path)
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        pages = _extract_pages_pdf(file_path)
    elif ext in (".md", ".txt", ".text"):
        pages = _extract_pages_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .pdf, .md, or .txt")

    all_chunks = []
    for page_num, page_text in enumerate(pages, 1):
        chunks = _intelligent_chunk(page_text, page_num)
        all_chunks.extend(chunks)

    # Save to database
    _save_chunks_to_db(all_chunks)
    return all_chunks


def _extract_pages_pdf(pdf_path: str) -> list[str]:
    """Extract text from a PDF using PyMuPDF."""
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages.append(text)
    doc.close()
    return pages


def _extract_pages_text(file_path: str) -> list[str]:
    """Extract text from markdown or plain text files, splitting on headings."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split on top-level headings (##) to simulate "pages"
    sections = re.split(r'\n(?=## )', content)
    pages = [s.strip() for s in sections if s.strip()]
    if not pages:
        pages = [content]
    return pages


def _intelligent_chunk(text: str, page_num: int) -> list[dict]:
    """
    Split text into meaningful chunks based on structure:
    - Headers
    - Bullet/numbered lists
    - Paragraphs
    - Tables (keep together)
    Does NOT use overlapping fixed-size windows.
    """
    chunks = []
    current_section = ""
    current_text_lines: list[str] = []

    lines = text.split("\n")

    for line in lines:
        stripped = line.strip()

        # Detect headers (markdown or uppercase titles)
        is_header = False
        if stripped.startswith("#"):
            is_header = True
            header_text = stripped.lstrip("#").strip()
        elif stripped.isupper() and len(stripped) > 3 and not stripped.startswith("|"):
            is_header = True
            header_text = stripped

        if is_header:
            # Save previous chunk
            if current_text_lines:
                chunk_text = "\n".join(current_text_lines).strip()
                if chunk_text and len(chunk_text) > 20:
                    chunks.append({
                        "page_num": page_num,
                        "section_title": current_section or "General",
                        "text": chunk_text,
                    })
                current_text_lines = []
            current_section = header_text
            continue

        # Detect table rows (keep together with current section)
        if stripped.startswith("|"):
            current_text_lines.append(stripped)
            continue

        # Detect bullet points
        if re.match(r'^[-*+] ', stripped) or re.match(r'^\d+\.\s', stripped):
            current_text_lines.append(stripped)
            continue

        # Empty line = potential paragraph break
        if not stripped:
            joined = "\n".join(current_text_lines).strip()
            if joined and len(joined) > 20:
                # If chunk is getting large, save it
                if len(joined) > 800:
                    chunks.append({
                        "page_num": page_num,
                        "section_title": current_section or "General",
                        "text": joined,
                    })
                    current_text_lines = []
            continue

        current_text_lines.append(stripped)

        # Safety: if accumulated text is very large, flush
        joined = "\n".join(current_text_lines)
        if len(joined) > 1200:
            chunks.append({
                "page_num": page_num,
                "section_title": current_section or "General",
                "text": joined.strip(),
            })
            current_text_lines = []

    # Final chunk
    if current_text_lines:
        chunk_text = "\n".join(current_text_lines).strip()
        if chunk_text and len(chunk_text) > 10:
            chunks.append({
                "page_num": page_num,
                "section_title": current_section or "General",
                "text": chunk_text,
            })

    # Number the chunks
    for i, chunk in enumerate(chunks):
        chunk["chunk_index"] = i

    return chunks


def _save_chunks_to_db(chunks: list[dict]) -> None:
    """Save processed chunks to the database."""
    db_name = os.getenv("DB_NAME", "business_agent.db")
    conn = sqlite3.connect(db_name)
    try:
        # Clear old chunks
        conn.execute("DELETE FROM pdf_chunks")

        for chunk in chunks:
            conn.execute(
                "INSERT INTO pdf_chunks (chunk_index, page_num, section_title, text) VALUES (?, ?, ?, ?)",
                (
                    chunk.get("chunk_index", 0),
                    chunk.get("page_num", 0),
                    chunk.get("section_title", ""),
                    chunk.get("text", ""),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def load_chunks() -> list[dict]:
    """Load all chunks from the database."""
    db_name = os.getenv("DB_NAME", "business_agent.db")
    conn = sqlite3.connect(db_name)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, chunk_index, page_num, section_title, text FROM pdf_chunks ORDER BY chunk_index"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
