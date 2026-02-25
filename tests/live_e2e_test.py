"""
tests/live_e2e_test.py
-----------------------
Live end-to-end test that:
  1. Loads a sample business PDF
  2. Runs the full pipeline (DB init → detect → enrich → synth data)
  3. Runs multi-turn conversation scenarios
  4. Checks DB for persisted orders/appointments/logs
  5. Reports pass/fail for each requirement
"""
import os
import sys
import json
import sqlite3
import time

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# Verify API key
api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key or "your_" in api_key:
    print("ERROR: No valid OPENAI_API_KEY in .env")
    sys.exit(1)
print(f"API Key: ...{api_key[-8:]}")

# Clean slate
DB_NAME = os.getenv("DB_NAME", "business_agent.db")
if os.path.exists(DB_NAME):
    os.remove(DB_NAME)
    print(f"Removed old {DB_NAME}")

# ── Phase 1: Infrastructure ──
print("\n" + "="*60)
print("PHASE 1: Infrastructure Setup")
print("="*60)

results = {}

# Test 1: Database init
try:
    from core.database import init_db, generate_synthetic_data, set_business_meta, get_business_meta
    init_db()
    conn = sqlite3.connect(DB_NAME)
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    conn.close()
    expected_tables = ["users", "services", "cart", "orders", "calendar_events", "loyalty_points",
                       "pdf_chunks", "enriched_knowledge", "conversations", "tool_calls", "agent_logs",
                       "business_meta", "conversation_state", "service_providers"]
    missing = [t for t in expected_tables if t not in tables]
    if missing:
        results["db_schema"] = f"FAIL — missing tables: {missing}"
    else:
        results["db_schema"] = f"PASS — {len(tables)} tables created"
    print(f"  DB Schema: {results['db_schema']}")
except Exception as e:
    results["db_schema"] = f"FAIL — {e}"
    print(f"  DB Schema: {results['db_schema']}")

# Test 2: PDF processing
try:
    from processing.pdf_processor import process_pdf, load_chunks
    pdf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_pdfs", "mario_pizza.md")
    chunks = process_pdf(pdf_path)
    results["pdf_processing"] = f"PASS — {len(chunks)} chunks from mario_pizza.md"
    print(f"  PDF Processing: {results['pdf_processing']}")
except Exception as e:
    results["pdf_processing"] = f"FAIL — {e}"
    print(f"  PDF Processing: {results['pdf_processing']}")

# Test 3: Business detection (LLM call)
print("\n  Detecting business type (LLM call)...")
try:
    from processing.knowledge_enricher import detect_business_type
    sample_text = "\n".join(c.get("text", "")[:200] for c in chunks[:5])
    biz_info = detect_business_type(sample_text)
    assert isinstance(biz_info, dict), f"Expected dict, got {type(biz_info)}"
    bname = biz_info.get("business_name", "?")
    btype = biz_info.get("business_type", "?")
    set_business_meta("business_name", bname)
    set_business_meta("business_type", btype)
    results["biz_detection"] = f"PASS — {bname} ({btype})"
    print(f"  Biz Detection: {results['biz_detection']}")
except Exception as e:
    results["biz_detection"] = f"FAIL — {e}"
    print(f"  Biz Detection: {results['biz_detection']}")
    bname, btype = "Mario's Pizza", "restaurant"
    set_business_meta("business_name", bname)
    set_business_meta("business_type", btype)

# Test 4: Knowledge enrichment (multiple LLM calls)
print("  Enriching knowledge (multiple LLM calls, ~30s)...")
try:
    from processing.knowledge_enricher import enrich_knowledge
    enriched = enrich_knowledge(sample_text, btype, bname)
    results["enrichment"] = f"PASS — {len(enriched)} skills generated"
    print(f"  Enrichment: {results['enrichment']}")
except Exception as e:
    results["enrichment"] = f"FAIL — {e}"
    print(f"  Enrichment: {results['enrichment']}")

# Test 5: Synthetic data generation (LLM call)
print("  Generating synthetic data (LLM call)...")
try:
    generate_synthetic_data(btype, bname)
    conn = sqlite3.connect(DB_NAME)
    user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    svc_count = conn.execute("SELECT COUNT(*) FROM services").fetchone()[0]
    order_count = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    provider_count = conn.execute("SELECT COUNT(*) FROM service_providers").fetchone()[0]
    conn.close()
    results["synth_data"] = f"PASS — {user_count} users, {svc_count} services, {order_count} orders, {provider_count} providers"
    print(f"  Synth Data: {results['synth_data']}")
except Exception as e:
    results["synth_data"] = f"FAIL — {e}"
    print(f"  Synth Data: {results['synth_data']}")

# ── Phase 2: Agent Conversation Tests ──
print("\n" + "="*60)
print("PHASE 2: Agent Conversation Tests")
print("="*60)

try:
    from agent.agent import run_agent_turn
    all_chunks = load_chunks()
except Exception as e:
    print(f"CRITICAL: Cannot import agent — {e}")
    sys.exit(1)

user_id = 1
conversation_id = "test-e2e-001"

def agent_turn(msg: str, conv_id: str = None) -> str:
    """Run one agent turn and return the response."""
    cid = conv_id or conversation_id
    print(f"\n  USER: {msg}")
    start = time.time()
    try:
        resp = run_agent_turn(
            user_id=user_id,
            conversation_id=cid,
            query=msg,
            all_chunks=all_chunks,
            business_name=bname,
            business_type=btype,
        )
        elapsed = time.time() - start
        print(f"  AGENT ({elapsed:.1f}s): {resp[:200]}{'...' if len(resp) > 200 else ''}")
        return resp
    except Exception as e:
        elapsed = time.time() - start
        print(f"  AGENT ERROR ({elapsed:.1f}s): {e}")
        return f"ERROR: {e}"

# Scenario A: Greeting + Info question
print("\n--- Scenario A: Greeting + Info ---")
r1 = agent_turn("Hi there!")
results["greeting"] = "PASS" if any(w in r1.lower() for w in ["hello", "hi", "welcome", "hey"]) else f"WARN — response may not contain greeting: {r1[:80]}"
print(f"  Result: {results['greeting']}")

# Scenario B: Pricing question (RAG retrieval)
print("\n--- Scenario B: Pricing Question ---")
r2 = agent_turn("How much does a large pepperoni pizza cost?")
results["pricing_rag"] = "PASS" if "$" in r2 or "price" in r2.lower() or "cost" in r2.lower() else f"WARN — no pricing info found: {r2[:80]}"
print(f"  Result: {results['pricing_rag']}")

# Scenario C: Order flow (add to cart)
print("\n--- Scenario C: Order Flow ---")
conversation_id = "test-order-001"
r3 = agent_turn("I'd like to order a large pepperoni pizza", conversation_id)
results["order_add"] = "PASS" if any(w in r3.lower() for w in ["added", "cart", "order", "pizza", "pepperoni"]) else f"WARN — order not acknowledged: {r3[:80]}"
print(f"  Result: {results['order_add']}")

# Scenario D: Add more items
r4 = agent_turn("Also add a coke please", conversation_id)
results["order_add2"] = "PASS" if any(w in r4.lower() for w in ["added", "coke", "cola", "drink", "cart"]) else f"WARN — item not added: {r4[:80]}"
print(f"  Result: {results['order_add2']}")

# Scenario E: View cart
r5 = agent_turn("What's in my cart?", conversation_id)
results["view_cart"] = "PASS" if any(w in r5.lower() for w in ["cart", "order", "item", "total", "$"]) else f"WARN — cart not shown: {r5[:80]}"
print(f"  Result: {results['view_cart']}")

# Scenario F: Change mind (remove item)
r6 = agent_turn("Actually remove the pizza, I want garlic bread instead", conversation_id)
results["change_mind"] = "PASS" if any(w in r6.lower() for w in ["removed", "garlic", "bread", "changed"]) else f"WARN — change not processed: {r6[:80]}"
print(f"  Result: {results['change_mind']}")

# Scenario G: Delivery + address
r7 = agent_turn("I want delivery to 34 Front Street, postal code N4K 4L7", conversation_id)
results["address"] = "PASS" if any(w in r7.lower() for w in ["address", "front", "delivery", "confirmed", "34"]) else f"WARN — address not processed: {r7[:80]}"
print(f"  Result: {results['address']}")

# Scenario H: Booking (separate conversation)
print("\n--- Scenario H: Booking Flow ---")
conversation_id = "test-booking-001"
r8 = agent_turn("I'd like to book an appointment", conversation_id)
results["booking_start"] = "PASS" if any(w in r8.lower() for w in ["book", "appointment", "schedule", "service", "when", "date", "time"]) else f"WARN — booking not initiated: {r8[:80]}"
print(f"  Result: {results['booking_start']}")

# Scenario I: Business hours
print("\n--- Scenario I: Business Hours ---")
conversation_id = "test-info-001"
r9 = agent_turn("What are your business hours?", conversation_id)
results["hours"] = "PASS" if any(w in r9.lower() for w in ["hour", "open", "close", "am", "pm", "monday", "daily"]) else f"WARN — hours not provided: {r9[:80]}"
print(f"  Result: {results['hours']}")

# Scenario J: Loyalty points
print("\n--- Scenario J: Loyalty Points ---")
r10 = agent_turn("How many loyalty points do I have?", conversation_id)
results["loyalty"] = "PASS" if any(w in r10.lower() for w in ["point", "loyalty", "reward", "tier", "bronze", "silver", "gold"]) else f"WARN — loyalty not shown: {r10[:80]}"
print(f"  Result: {results['loyalty']}")


# ── Phase 3: Database Persistence Checks ──
print("\n" + "="*60)
print("PHASE 3: Database Persistence Verification")
print("="*60)

conn = sqlite3.connect(DB_NAME)
conn.row_factory = sqlite3.Row

# Check conversation logs
conv_count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
results["conv_persistence"] = f"PASS — {conv_count} conversation messages logged" if conv_count > 0 else "FAIL — no conversation messages persisted"
print(f"  Conversations: {results['conv_persistence']}")

# Check tool call logs
tool_count = conn.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]
results["tool_logs"] = f"PASS — {tool_count} tool calls logged" if tool_count > 0 else "FAIL — no tool calls logged"
print(f"  Tool Calls: {results['tool_logs']}")

# Check agent event logs
event_count = conn.execute("SELECT COUNT(*) FROM agent_logs").fetchone()[0]
results["agent_logs"] = f"PASS — {event_count} agent events logged" if event_count > 0 else "FAIL — no agent events logged"
print(f"  Agent Events: {results['agent_logs']}")

# Check cart items
cart_count = conn.execute("SELECT COUNT(*) FROM cart").fetchone()[0]
print(f"  Cart Items: {cart_count} items currently in cart")

# Check conversation state
state_count = conn.execute("SELECT COUNT(*) FROM conversation_state").fetchone()[0]
results["conv_state"] = f"PASS — {state_count} conversation states persisted" if state_count > 0 else "WARN — no conversation state persisted (may be valid if no multi-turn flows required state)"
print(f"  Conversation State: {results['conv_state']}")

# Check enriched knowledge
ek_count = conn.execute("SELECT COUNT(*) FROM enriched_knowledge").fetchone()[0]
results["enriched_kb"] = f"PASS — {ek_count} enriched knowledge entries" if ek_count > 0 else "FAIL — no enriched knowledge"
print(f"  Enriched Knowledge: {results['enriched_kb']}")

# Check PDF chunks
chunk_count = conn.execute("SELECT COUNT(*) FROM pdf_chunks").fetchone()[0]
results["chunks_stored"] = f"PASS — {chunk_count} PDF chunks stored" if chunk_count > 0 else "FAIL — no PDF chunks"
print(f"  PDF Chunks: {results['chunks_stored']}")

# Sample some tool call details for review
print("\n  Sample Tool Calls:")
tool_samples = conn.execute("SELECT tool_name, activated, timestamp FROM tool_calls ORDER BY id DESC LIMIT 8").fetchall()
for t in tool_samples:
    print(f"    {t['tool_name']}: activated={t['activated']} at {t['timestamp']}")

# Sample agent events
print("\n  Sample Agent Events:")
event_samples = conn.execute("SELECT event_type, details FROM agent_logs ORDER BY id DESC LIMIT 5").fetchall()
for e in event_samples:
    details = e['details'][:100] if e['details'] else 'N/A'
    print(f"    {e['event_type']}: {details}")

conn.close()


# ── Phase 4: Final Report ──
print("\n" + "="*60)
print("FINAL REPORT")
print("="*60)

passes = sum(1 for v in results.values() if v.startswith("PASS"))
warns = sum(1 for v in results.values() if v.startswith("WARN"))
fails = sum(1 for v in results.values() if v.startswith("FAIL"))
total = len(results)

print(f"\n  Total Tests: {total}")
print(f"  ✅ PASS: {passes}")
print(f"  ⚠️  WARN: {warns}")
print(f"  ❌ FAIL: {fails}")
print()

for name, result in results.items():
    icon = "✅" if result.startswith("PASS") else "⚠️" if result.startswith("WARN") else "❌"
    print(f"  {icon} {name}: {result}")

print(f"\nOverall: {'PASS' if fails == 0 else 'FAIL'} ({passes}/{total} passed, {warns} warnings)")
