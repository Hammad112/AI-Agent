"""
agent/rag_engine.py
-------------------
LLM-only Retrieval-Augmented Generation — no embeddings, no vector DB.
Import path changed from: rag_engine → agent.rag_engine
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


def chunk_query_into_topics(query: str, history: list[dict]) -> list[str]:
    history_text = ""
    if history:
        recent = history[-4:]
        history_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)

    prompt = f"""
Analyse this customer service query and conversation history.
Extract 3-6 key topics/intents that a knowledge base search should cover.

Conversation history:
{history_text or 'None'}

Current query: "{query}"

Return ONLY a JSON array of short topic strings:
["pricing", "booking", "cancellation policy"]
No markdown, just JSON.
"""
    raw = llm_call(prompt, temperature=0.1, max_tokens=200)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        topics = json.loads(raw)
        return [str(t).lower().strip() for t in topics if t]
    except (json.JSONDecodeError, TypeError):
        return [query.lower()]


def retrieve_relevant_chunks(
    query_topics: list[str],
    chunks: list[dict],
    top_n: int = 5,
) -> tuple[list[dict], list[int]]:
    if not chunks:
        return [], []

    chunk_index_map = {i: chunk for i, chunk in enumerate(chunks)}

    chunk_summaries = []
    for i, chunk in enumerate(chunks[:60]):
        text_preview = chunk["text"][:200].replace("\n", " ")
        section = chunk.get("section_title", "")
        chunk_summaries.append(f"[{i}] Section: {section} | {text_preview}")

    summaries_text = "\n".join(chunk_summaries)
    topics_text = ", ".join(query_topics)

    prompt = f"""
You are a context retrieval engine for a customer service system.

The customer is asking about these topics: {topics_text}

Below are numbered knowledge chunks. Select the {top_n} most relevant ones.

Chunks:
{summaries_text}

Return ONLY a JSON array of the selected chunk indices (integers):
[0, 3, 7, 12, 15]
No markdown, just JSON.
"""
    raw = llm_call(prompt, temperature=0.1, max_tokens=100)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        indices = json.loads(raw)
        selected_indices = [int(i) for i in indices if isinstance(i, (int, float)) and int(i) in chunk_index_map]
        selected_chunks = [chunk_index_map[i] for i in selected_indices]
        chunk_ids = [chunk.get("id", chunk.get("chunk_index", i)) for i, chunk in zip(selected_indices, selected_chunks)]
        return selected_chunks, chunk_ids
    except (json.JSONDecodeError, TypeError, ValueError):
        fallback = chunks[:top_n]
        return fallback, [c.get("id", c.get("chunk_index", 0)) for c in fallback]


def get_customer_context(user_id: int) -> dict:
    conn = _get_db()
    try:
        user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if not user:
            return {}

        orders = conn.execute(
            """
            SELECT o.*, s.name as service_name, s.price, sp.name as provider_name
            FROM orders o
            LEFT JOIN services s ON o.service_id = s.id
            LEFT JOIN service_providers sp ON o.provider_id = sp.id
            WHERE o.user_id = ?
            ORDER BY o.scheduled_at DESC
            LIMIT 10
            """,
            (user_id,),
        ).fetchall()

        loyalty = conn.execute(
            "SELECT * FROM loyalty_points WHERE user_id = ?", (user_id,)
        ).fetchone()

        order_list = [dict(o) for o in orders]
        return {
            "user_id": user_id,
            "username": user["username"],
            "full_name": user["full_name"],
            "email": user["email"],
            "phone": user["phone"],
            "loyalty_points": loyalty["points"] if loyalty else 0,
            "loyalty_tier": loyalty["tier"] if loyalty else "bronze",
            "order_history": order_list,
            "total_orders": len(order_list),
            "completed_orders": sum(1 for o in order_list if o["status"] == "completed"),
        }
    finally:
        conn.close()


def build_composite_prompt(
    query: str,
    retrieved_chunks: list[dict],
    customer_context: dict,
    global_stats: dict,
    tool_results: dict,
    conversation_history: list[dict],
    business_name: str,
    business_type: str,
) -> str:
    # Knowledge context
    knowledge_text = ""
    if retrieved_chunks:
        knowledge_parts = []
        for chunk in retrieved_chunks:
            section = chunk.get("section_title", "")
            text = chunk.get("text", "")
            topic = chunk.get("topic", "")
            if topic:
                knowledge_parts.append(f"[{topic}]\n{text}")
            else:
                knowledge_parts.append(f"[{section}]\n{text}")
        knowledge_text = "\n\n".join(knowledge_parts)

    # Customer context
    customer_text = ""
    if customer_context:
        orders_summary = ""
        if customer_context.get("order_history"):
            recent = customer_context["order_history"][:5]
            items = [
                f"  - {o.get('service_name','Service')} | {o.get('status','?')} | {o.get('scheduled_at','?')[:10]}"
                for o in recent
            ]
            orders_summary = "\n".join(items)

        customer_text = f"""Customer Profile:
  Name: {customer_context.get('full_name', 'Unknown')}
  Loyalty: {customer_context.get('loyalty_points', 0)} points ({customer_context.get('loyalty_tier', 'bronze')} tier)
  Total bookings: {customer_context.get('total_orders', 0)}
Recent orders:
{orders_summary or '  No recent orders'}""".strip()

    # Business stats
    stats_text = ""
    if global_stats:
        stats_text = (
            f"Business overview: {global_stats.get('total_customers',0)} customers, "
            f"{global_stats.get('total_orders',0)} total orders, "
            f"${global_stats.get('total_revenue',0):.2f} total revenue, "
            f"{global_stats.get('total_providers',0)} staff members, "
            f"{global_stats.get('total_services',0)} services available."
        )

    # Tool results
    tools_text = ""
    if tool_results:
        tool_parts = [
            f"[{name} result]\n{json.dumps(result, indent=2, default=str)}"
            for name, result in tool_results.items()
        ]
        tools_text = "\n\n".join(tool_parts)

    # Conversation history
    history_text = ""
    if conversation_history:
        recent = conversation_history[-6:]
        history_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)

    prompt = f"""You are a helpful, friendly customer service assistant for "{business_name}" ({business_type}).
Always be professional, empathetic, and solution-oriented. Stay relevant to what the customer asked.

════════════════════════════════════════
BUSINESS KNOWLEDGE BASE
════════════════════════════════════════
{knowledge_text or 'No specific knowledge retrieved.'}

════════════════════════════════════════
CUSTOMER INFORMATION
════════════════════════════════════════
{customer_text or 'No customer profile available.'}

════════════════════════════════════════
BUSINESS STATISTICS
════════════════════════════════════════
{stats_text or 'No statistics available.'}

════════════════════════════════════════
TOOL RESULTS (Actions already performed)
════════════════════════════════════════
{tools_text or 'No tools were activated for this query.'}

════════════════════════════════════════
CONVERSATION HISTORY
════════════════════════════════════════
{history_text or 'This is the start of the conversation.'}

════════════════════════════════════════
CUSTOMER QUERY
════════════════════════════════════════
{query}

════════════════════════════════════════
YOUR RESPONSE (be concise, helpful, and specific):
"""
    return prompt
