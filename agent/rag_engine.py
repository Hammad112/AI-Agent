"""
agent/rag_engine.py
-------------------
LLM-only Retrieval-Augmented Generation (no embeddings, no vector DB).
Uses LLM calls for:
  - Chunking queries into topics
  - Dividing customer information into logical topics
  - Retrieving relevant (query-topic, PDF-chunk) pairs
  - Building composite prompt for final response
"""

import os
import json
import sqlite3
from core.llm_client import llm_call
from core.logger import log_agent_event


def chunk_query_into_topics(query: str, history: list[dict] | None = None) -> list[str]:
    """Use a separate LLM call to chunk/divide the query into logical topics."""
    history_text = ""
    if history:
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in history[-4:]
        )

    prompt = f"""Divide the following customer query into distinct logical topics for information retrieval.
Each topic should be a short phrase that can be matched against a knowledge base.

Conversation context:
{history_text or 'None'}

Customer query: "{query}"

Return ONLY a JSON array of topic strings, e.g.: ["topic 1", "topic 2"]
No markdown, just JSON."""

    try:
        raw = llm_call(prompt, temperature=0.1, max_tokens=200)
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start != -1 and end > start:
            topics = json.loads(raw[start:end])
            return [str(t) for t in topics if t]
    except Exception:
        pass
    return [query]


def retrieve_relevant_chunks(
    topics: list[str],
    all_chunks: list[dict],
    top_n: int = 5,
) -> tuple[list[dict], list[int]]:
    """
    Use LLM calls to find relevant pairs (query-topic, PDF-chunk).
    No 60-chunk hard limit. Processes ALL chunks via batched summaries.
    """
    if not all_chunks or not topics:
        return [], []

    # Build chunk summaries in batches to handle large PDFs
    batch_size = 40
    relevant_indices: set[int] = set()

    for batch_start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[batch_start:batch_start + batch_size]
        chunk_summaries = "\n".join(
            f"[{batch_start + i}] {c.get('section_title', '')} : {c.get('text', '')[:150]}"
            for i, c in enumerate(batch)
        )

        prompt = f"""Given these search topics: {json.dumps(topics)}

And these knowledge chunks:
{chunk_summaries}

Which chunks are relevant to the topics? Return ONLY a JSON array of chunk indices (the numbers in brackets), e.g.: [0, 3, 7]
Return at most {top_n} most relevant indices per batch. If none are relevant, return [].
No markdown, just JSON."""

        try:
            raw = llm_call(prompt, temperature=0.0, max_tokens=150)
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start != -1 and end > start:
                indices = json.loads(raw[start:end])
                for idx in indices:
                    if isinstance(idx, int) and 0 <= idx < len(all_chunks):
                        relevant_indices.add(idx)
        except Exception:
            continue

    # Limit to top_n
    selected_indices = sorted(relevant_indices)[:top_n]
    retrieved = [all_chunks[i] for i in selected_indices]
    chunk_ids = [c.get("id", i) for i, c in zip(selected_indices, retrieved)]

    # Also fetch enriched knowledge that matches
    enriched = _fetch_enriched_knowledge(topics)
    for ek in enriched:
        retrieved.append({"text": ek["content"], "section_title": f"[Enriched] {ek['topic']}", "id": -1})

    return retrieved, chunk_ids


def _fetch_enriched_knowledge(topics: list[str]) -> list[dict]:
    """Fetch enriched knowledge entries matching the query topics."""
    db_name = os.getenv("DB_NAME", "business_agent.db")
    conn = sqlite3.connect(db_name)
    conn.row_factory = sqlite3.Row
    try:
        results = []
        for topic in topics:
            keywords = topic.lower().split()
            for kw in keywords:
                if len(kw) < 3:
                    continue
                rows = conn.execute(
                    "SELECT topic, content FROM enriched_knowledge WHERE LOWER(topic) LIKE ? OR LOWER(content) LIKE ? LIMIT 2",
                    (f"%{kw}%", f"%{kw}%"),
                ).fetchall()
                for r in rows:
                    if not any(existing["topic"] == r["topic"] for existing in results):
                        results.append(dict(r))
        return results[:5]
    finally:
        conn.close()


def chunk_customer_info(customer_context: dict) -> list[dict]:
    """
    Use a separate LLM call to chunk/divide customer information into logical topics.
    (Requirement III from the spec)
    """
    if not customer_context or all(not v for v in customer_context.values()):
        return []

    ctx_text = json.dumps(customer_context, indent=2, default=str)
    prompt = f"""Divide this customer's information into logical topic groups.
Each group should have a label and the relevant data.

Customer information:
{ctx_text}

Return ONLY JSON array: [{{"topic": "Contact Info", "data": "..."}}, {{"topic": "Order History", "data": "..."}}]
No markdown, just JSON."""

    try:
        raw = llm_call(prompt, temperature=0.1, max_tokens=400)
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start != -1 and end > start:
            return json.loads(raw[start:end])
    except Exception:
        pass

    return [{"topic": "Full Profile", "data": ctx_text}]


def get_customer_context(user_id: int, session_id: str = "") -> dict:
    """Gather all customer data for context window."""
    db_name = os.getenv("DB_NAME", "business_agent.db")
    conn = sqlite3.connect(db_name)
    conn.row_factory = sqlite3.Row
    try:
        user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if not user:
            return {}

        orders = conn.execute(
            """SELECT o.id, o.status, o.order_type, o.scheduled_at, o.total_price,
                      o.delivery_type, o.items_json,
                      s.name as service_name
               FROM orders o LEFT JOIN services s ON o.service_id = s.id
               WHERE o.user_id = ? ORDER BY o.scheduled_at DESC LIMIT 8""",
            (user_id,),
        ).fetchall()

        loyalty = conn.execute(
            "SELECT points, tier FROM loyalty_points WHERE user_id = ?",
            (user_id,),
        ).fetchone()

        events = conn.execute(
            """SELECT e.*, s.name as service_name FROM calendar_events e
               LEFT JOIN orders o ON e.order_id = o.id
               LEFT JOIN services s ON o.service_id = s.id
               WHERE e.user_id = ? AND e.status != 'cancelled' 
               ORDER BY e.start_time LIMIT 5""",
            (user_id,),
        ).fetchall()

        if session_id:
            cart_items = conn.execute(
                """SELECT c.*, s.name as service_name FROM cart c
                   LEFT JOIN services s ON c.service_id = s.id
                   WHERE c.user_id = ? AND c.session_id = ?""",
                (user_id, session_id),
            ).fetchall()
        else:
            cart_items = conn.execute(
                """SELECT c.*, s.name as service_name FROM cart c
                   LEFT JOIN services s ON c.service_id = s.id
                   WHERE c.user_id = ?""",
                (user_id,),
            ).fetchall()

        # Recent complaints
        complaints = conn.execute(
            """SELECT complaint_type, description, status, created_at FROM complaints 
               WHERE user_id = ? ORDER BY created_at DESC LIMIT 3""",
            (user_id,),
        ).fetchall()

        # Most ordered service
        fav = conn.execute(
            """SELECT s.name, COUNT(o.id) as cnt FROM orders o
               JOIN services s ON o.service_id = s.id
               WHERE o.user_id = ? GROUP BY s.name ORDER BY cnt DESC LIMIT 1""",
            (user_id,),
        ).fetchone()

        context = {
            "user_id": user_id,
            "full_name": user["full_name"],
            "email": user["email"],
            "phone": user["phone"],
            "address": user["address"],
            "postal_code": user["postal_code"],
            "city": user["city"],
            "family_members": user["family_members"],
            "loyalty_points": loyalty["points"] if loyalty else 0,
            "loyalty_tier": loyalty["tier"] if loyalty else "bronze",
            "usual_favorite": fav["name"] if fav and fav["name"] else "our standard services",
            "usual_favorite_count": fav["cnt"] if fav else 0,
            "recent_orders": [dict(o) for o in orders],
            "recent_complaints": [dict(c) for c in complaints],
            "upcoming_events": [dict(e) for e in events],
            "current_cart": [dict(c) for c in cart_items],
        }
        return context
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
    detected_intents: list[str] | None = None,
    conversation_state: dict | None = None,
) -> str:
    """Build the final composite prompt from all sources."""

    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in (conversation_history or [])[-10:]
    )

    chunks_text = "\n\n".join(
        f"[{c.get('section_title', 'Info')}]\n{c.get('text', '')}"
        for c in retrieved_chunks
    )
    if not chunks_text:
        chunks_text = "No specific knowledge retrieved."

    # Chunk customer info into logical topics
    customer_topics = chunk_customer_info(customer_context)
    customer_text = "\n".join(
        f"[{t.get('topic', 'Info')}] {t.get('data', '')}"
        for t in customer_topics
    ) if customer_topics else json.dumps(customer_context, indent=2, default=str)

    tools_text = ""
    if tool_results:
        for tool_name, result in tool_results.items():
            tools_text += f"\n--- {tool_name} ---\n{json.dumps(result, indent=2, default=str)}\n"

    # Complaints text
    complaints_list = customer_context.get("recent_complaints", [])
    complaints_text = ""
    if complaints_list:
        complaints_text = "PAST DISPUTES/COMPLAINTS:\n" + "\n".join(
            f"- [{c['created_at']}] {c['complaint_type']}: {c['description']} (Status: {c['status']})"
            for c in complaints_list
        )

    # Active conversational flow state
    conv_state = conversation_state or {}
    active_flow = conv_state.get("active_flow")
    pending_slots = conv_state.get("pending_slots", {})
    cart_active = conv_state.get("cart_active", False)
    satisfied_reqs = conv_state.get("satisfied_requirements", [])

    flow_instructions = ""
    if active_flow == "ordering" and cart_active:
        flow_instructions = """
FLOW: Customer is building an order. Ask if they want to add more items, specify
delivery/pickup, or confirm the order. Be a good salesperson — suggest related items
or popular additions."""
        if "allergies_checked" not in satisfied_reqs:
            flow_instructions += "\nCRITICAL: Ask if the customer has any food allergies or dietary restrictions if you haven't already."
        if "group_size_checked" not in satisfied_reqs:
            flow_instructions += "\nCRITICAL: Ask if the customer is ordering for themselves or a group if you haven't already."
    elif active_flow == "booking" and pending_slots:
        flow_instructions = f"""
FLOW: Customer is booking an appointment. Still missing: {list(pending_slots.keys())}.
Ask for the missing information naturally."""
    elif active_flow == "ordering":
        flow_instructions = """
FLOW: Customer was ordering. The cart may be empty now. Ask if they want to start a new order."""

    # Check if customer has usual favorites
    usual_fav = customer_context.get("usual_favorite")
    fav_instruction = ""
    if usual_fav:
        fav_instruction = f"\nNote: This customer usually orders '{usual_fav}'. Remind them if relevant."

    # Family cross-sell (Relationship-aware)
    family_members = customer_context.get("family_members")
    family_instruction = ""
    if family_members and family_members != "[]":
        try:
            fam_list = json.loads(family_members) if isinstance(family_members, str) else family_members
            if fam_list:
                # Relation filter: Match query term to relationship
                match_found = False
                q_low = query.lower()
                rel_map = {"wife": ("spouse", "Female"), "husband": ("spouse", "Male"), "son": ("child", "Male"), "daughter": ("child", "Female")}
                
                for term, (rel, gen) in rel_map.items():
                    if term in q_low:
                        matches = [m for m in fam_list if m.get("relation") == rel and m.get("gender") == gen]
                        if matches:
                            names = ", ".join(m.get("name", "") for m in matches)
                            family_instruction = f"\nNote: The customer mentioned '{term}'. Suggest {names} (on file) if relevant."
                            match_found = True
                            break
                        else:
                            family_instruction = f"\nNote: The customer mentioned '{term}', but you don't have a {term} on file. Ask if they'd like to add her/him."
                            match_found = True
                            break
                
                if not match_found:
                    # Default generic family mention
                    names = ", ".join(f"{m.get('name', '')} ({m.get('relation', '')})" for m in fam_list)
                    family_instruction = f"\nFamily members on file: {names}. Suggest services for them when appropriate."
        except (json.JSONDecodeError, TypeError):
            pass

    prompt = f"""You are a friendly, professional customer service agent for "{business_name}" ({business_type}).

STRICT RULES:
- SERVICE NAMES: NEVER use internal IDs like "Service 4". ALWAYS use the natural service name provided in tool results or context.
- PRICING GAPS: If price is missing or "$0.00", explicitly say "Please call us for current pricing" or "Contact us for a quote". 
- NATURAL UPSELL: Suggest ONE relevant item per session only if it adds real value (e.g. "Since you're getting a pizza, would you like a drink?"). DO NOT repeat upsells.
- CONFIRMATION: Always include Order/Booking/Complaint Reference numbers (e.g., ORDER-123, APPT-456, REF-789) provided by the tools.
- ADDRESSES: Acknowledge the street and city when confirming delivery.
- COMPLAINTS: Acknowledge grievances warmly, confirm they are logged, and provide a reference number (REF-123).
- LOYALTY: Mention point balance and tier if relevant to the turn.
- GROUNDING: NEVER GUESS or INVENT prices. If information is missing, ask the customer — do NOT guess.
- Address ALL detected intents: {detected_intents or ['general']}
{flow_instructions}{fav_instruction}{family_instruction}

KNOWLEDGE BASE:
{chunks_text}

CUSTOMER PROFILE:
{customer_text}

{complaints_text}

TOOL RESULTS:
{tools_text or 'None'}

CONVERSATION HISTORY:
{history_text or 'No prior messages.'}

CUSTOMER MESSAGE: "{query}"

Respond naturally as the agent. Address all parts of the customer's message."""

    return prompt
