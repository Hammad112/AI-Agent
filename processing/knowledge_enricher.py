"""
processing/knowledge_enricher.py
---------------------------------
Uses LLM to:
  1) detect_business_type() from PDF text
  2) enrich_knowledge() — generate supplementary knowledge not in the PDF
     (called "skills" in the spec): FAQs, product comparisons, tips, etc.
"""

import os
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
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


def detect_business_type(pdf_summary: str) -> dict:
    """
    Detect business name and type from PDF content.
    Returns dict: {"business_name": "...", "business_type": "..."}
    """
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
        return {
            "business_name": data.get("business_name", "The Business"),
            "business_type": data.get("business_type", "general business"),
        }
    except json.JSONDecodeError:
        return {"business_name": "The Business", "business_type": "general business"}


def enrich_knowledge(pdf_text: str, business_type: str, business_name: str = "") -> list[dict]:
    """
    Generate supplementary "skills" — knowledge topics not in the PDF.
    Uses LLM to identify gaps and then write detailed articles for each.
    
    Args:
        pdf_text: Sample text from the ingested PDF for context
        business_type: e.g. "restaurant", "dental clinic"
        business_name: Optional name of the business
    """
    if enrichment_already_done():
        return load_enriched_from_db()

    if not business_name:
        business_name = "the business"

    topics_prompt = f"""
You are a knowledge base enricher for "{business_name}" ({business_type}).

Here is the existing knowledge from their PDF:
---
{pdf_text[:3000]}
---

List 8-10 important topics that would be valuable to add to this knowledge base
but are NOT fully covered in the existing content. These are called "skills" —
supplementary knowledge that helps the AI agent handle customer requests better.

Focus on:
- FAQs customers commonly ask for this type of business
- Detailed product/service specifications or comparisons
- Cross-selling and upselling strategies
- Pricing tier breakdowns and value explanations
- Booking and cancellation policies and best practices
- Common tips or best practices for this business type
- Seasonal offers or promotions structure
- Staff qualifications and expertise areas
- Customer care and complaint handling procedures
- Industry-standard practices and quality markers

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
    
    # Use parallel execution for faster enrichment
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_enrich_topic, topic, business_name, business_type): topic for topic in topics}
        
        for future in futures:
            topic = futures[future]
            try:
                content = future.result()
                if content:
                    enriched.append({
                        "topic": topic,
                        "content": content,
                        "source": "llm_enrichment",
                        "topic_tags": topic.lower().replace(" ", ","),
                    })
            except Exception as e:
                print(f"Error enriching topic {topic}: {e}")

    _save_enriched(enriched)
    return enriched


def _enrich_topic(topic: str, business_name: str, business_type: str) -> str:
    """Generate a detailed knowledge article on the given topic."""
    prompt = f"""
You are a knowledgeable assistant for "{business_name}" ({business_type}).

Write a detailed, helpful, and realistic knowledge base article about:
"{topic}"

This content will be used by an AI customer service agent to answer customer questions.
Make it specific to {business_type} businesses. Be factual, practical, and thorough.
Include:
- Relevant tips and best practices
- Common questions and their answers
- Industry-standard information
- Practical examples where helpful

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
