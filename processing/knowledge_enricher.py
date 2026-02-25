"""
processing/knowledge_enricher.py
---------------------------------
Uses LLM to generate supplementary knowledge not in the PDF.
Import path changed from: knowledge_enricher → processing.knowledge_enricher
"""

import os
import json
import sqlite3
from core.llm_client import llm_call


def _get_db() -> sqlite3.Connection:
    db_name = os.getenv("DB_NAME", "business_agent.db")
    conn = sqlite3.connect(db_name, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def enrichment_already_done() -> bool:
    conn = _get_db()
    try:
        count = conn.execute("SELECT COUNT(*) FROM enriched_knowledge").fetchone()[0]
        return count > 0
    finally:
        conn.close()


def detect_business_type(pdf_summary: str) -> tuple[str, str]:
    prompt = f"""
Analyse this business document excerpt and identify the business:

---
{pdf_summary[:2000]}
---

Return ONLY a JSON object like:
{{
  "business_name": "Glow Beauty Salon",
  "business_type": "beauty salon"
}}

No markdown, just JSON.
"""
    raw = llm_call(prompt, temperature=0.1)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(raw)
        return data.get("business_name", "The Business"), data.get("business_type", "general business")
    except json.JSONDecodeError:
        return "The Business", "general business"


def enrich_knowledge(
    chunks: list[dict],
    business_name: str,
    business_type: str,
) -> list[dict]:
    chunk_texts = "\n\n".join(c["text"] for c in chunks[:40])
    topics_prompt = f"""
You are a knowledge base enricher for "{business_name}" ({business_type}).

Here is the existing knowledge from their PDF:
---
{chunk_texts[:3000]}
---

List 8 important topics that would be valuable to add to this knowledge base
but are NOT fully covered in the existing content. Focus on:
- FAQs customers commonly ask
- Detailed product/service specs or comparisons
- Pricing tier breakdowns
- Loyalty and discount explanations
- Booking and cancellation policies
- Common tips or best practices for this business type
- Seasonal offers or promotions structure
- Staff qualifications or certifications

Return ONLY a JSON array of topic names:
["Topic 1", "Topic 2", ...]
No markdown, just JSON.
"""
    raw = llm_call(topics_prompt, temperature=0.4)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        topics = json.loads(raw)
        if not isinstance(topics, list):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        topics = [
            "Frequently Asked Questions",
            "Pricing & Packages",
            "Booking & Cancellation Policy",
            "Loyalty Rewards Program",
            "Staff Qualifications",
            "Product/Service Comparisons",
            "Seasonal Promotions",
            "Tips & Best Practices",
        ]

    enriched = []
    for topic in topics:
        content = _enrich_topic(topic, business_name, business_type)
        if content:
            enriched.append(
                {
                    "topic": topic,
                    "content": content,
                    "source": "llm_enrichment",
                    "topic_tags": topic.lower().replace(" ", ","),
                }
            )

    _save_enriched(enriched)
    return enriched


def _enrich_topic(topic: str, business_name: str, business_type: str) -> str:
    prompt = f"""
You are a knowledgeable assistant for "{business_name}" ({business_type}).

Write a detailed, helpful, and realistic knowledge base article about:
"{topic}"

This content will be shown to customers as part of a customer service response system.
Make it specific to {business_type} businesses. Be factual, practical, and thorough.
Use clear paragraphs. Write in the third person. Approximately 200-350 words.
"""
    try:
        return llm_call(prompt, temperature=0.5, max_tokens=600)
    except Exception:
        return f"Information about {topic} for {business_name}."


def _save_enriched(enriched: list[dict]) -> None:
    conn = _get_db()
    try:
        for item in enriched:
            conn.execute(
                "INSERT INTO enriched_knowledge (topic, content, source, topic_tags) VALUES (?, ?, ?, ?)",
                (
                    item["topic"],
                    item["content"],
                    item.get("source", "llm_enrichment"),
                    item.get("topic_tags"),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def load_enriched_from_db() -> list[dict]:
    conn = _get_db()
    try:
        rows = conn.execute("SELECT * FROM enriched_knowledge ORDER BY id").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
