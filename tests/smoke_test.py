"""
tests/smoke_test.py
-------------------
Automated smoke test that runs scripted conversations
against two sample businesses (restaurant + dental clinic).

Usage:
    python tests/smoke_test.py

Validates that the agent pipeline runs end-to-end without crashing.
"""

import os
import sys
import time
import json
import traceback

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from core.database import init_db, generate_synthetic_data, set_business_meta, get_business_meta
from processing.pdf_processor import process_pdf, load_chunks
from processing.knowledge_enricher import detect_business_type, enrich_knowledge
from agent.agent import run_agent_turn


# ─── Test Scenarios ────────────────────────────

PIZZA_SCENARIOS = [
    ("Greeting", "Hi there, I'd like to order some food"),
    ("Add item", "Can I get a large pepperoni pizza?"),
    ("Add another", "Also add a Caesar salad and two Sprites please"),
    ("Change mind", "Actually, remove the salad. Add garlic bread instead"),
    ("View cart", "What do I have so far?"),
    ("Ask delivery", "Is this for delivery. My address is 34 Front Street, Toronto, M4K 6B2"),
    ("Ask price", "How much is my total?"),
    ("Confirm", "Yes, confirm my order please"),
    ("Complaint", "Last time I ordered 3 pizzas but only got 2"),
    ("Hours", "What are your hours on Saturday?"),
    ("Loyalty", "How many loyalty points do I have?"),
    ("Goodbye", "Thanks, bye!"),
]

DENTAL_SCENARIOS = [
    ("Greeting", "Hello, I need to book an appointment"),
    ("Ask availability", "When is Dr. Chen available?"),
    ("Book", "I'd like to book a dental cleaning for next Tuesday at 10 AM"),
    ("Ask price", "How much does teeth whitening cost?"),
    ("Reschedule", "Can I reschedule my appointment to Friday at 2 PM?"),
    ("Family", "Does my wife need a cleaning too?"),
    ("Cancel", "I need to cancel my appointment"),
    ("Promotions", "Do you have any special offers?"),
    ("Provider info", "Tell me about your orthodontist"),
    ("Mixed intent", "What's the price for a crown and can I also book a consultation for next week?"),
]


# ─── Test Runner ───────────────────────────────

def run_test_suite(
    business_file: str,
    scenarios: list[tuple[str, str]],
    suite_name: str,
) -> dict:
    """Run a suite of test scenarios against the agent."""
    print(f"\n{'='*60}")
    print(f"  TEST SUITE: {suite_name}")
    print(f"  Business File: {business_file}")
    print(f"{'='*60}")

    results = {
        "suite": suite_name,
        "total": len(scenarios),
        "passed": 0,
        "failed": 0,
        "errors": [],
    }

    # Check if file exists
    if not os.path.exists(business_file):
        print(f"  [SKIP] File not found: {business_file}")
        results["errors"].append(f"File not found: {business_file}")
        return results

    # Delete old DB for clean test
    db_name = f"test_{suite_name.lower().replace(' ', '_')}.db"
    os.environ["DB_NAME"] = db_name
    if os.path.exists(db_name):
        os.remove(db_name)

    try:
        # Setup
        print(f"  [SETUP] Initializing database...")
        init_db()

        print(f"  [SETUP] Processing business file...")
        all_chunks = process_pdf(business_file)
        print(f"  [SETUP] Created {len(all_chunks)} chunks")

        sample_text = "\n".join(c.get("text", "")[:200] for c in all_chunks[:5])
        business_info = detect_business_type(sample_text)
        business_name = business_info.get("business_name", "TestBusiness")
        business_type = business_info.get("business_type", "general")
        print(f"  [SETUP] Detected: {business_name} ({business_type})")

        set_business_meta("business_name", business_name)
        set_business_meta("business_type", business_type)

        print(f"  [SETUP] Enriching knowledge...")
        enrich_knowledge(sample_text, business_type)

        print(f"  [SETUP] Generating synthetic data...")
        generate_synthetic_data(business_type, business_name)

        conv_id = f"test_{suite_name}_{int(time.time())}"
        user_id = 1

        # Run scenarios
        for i, (label, message) in enumerate(scenarios, 1):
            print(f"\n  [{i}/{len(scenarios)}] {label}")
            print(f"    USER: {message}")
            try:
                start = time.time()
                response = run_agent_turn(
                    user_id=user_id,
                    conversation_id=conv_id,
                    query=message,
                    all_chunks=all_chunks,
                    business_name=business_name,
                    business_type=business_type,
                )
                elapsed = time.time() - start

                # Basic validation
                if response and len(response) > 10:
                    print(f"    AGENT: {response[:120]}{'...' if len(response) > 120 else ''}")
                    print(f"    [PASS] ({elapsed:.1f}s)")
                    results["passed"] += 1
                else:
                    print(f"    AGENT: {response}")
                    print(f"    [FAIL] Response too short or empty")
                    results["failed"] += 1
                    results["errors"].append(f"{label}: Empty/short response")

            except Exception as e:
                print(f"    [FAIL] {str(e)[:100]}")
                results["failed"] += 1
                results["errors"].append(f"{label}: {str(e)[:200]}")

    except Exception as e:
        print(f"  [SETUP FAILED] {str(e)}")
        traceback.print_exc()
        results["errors"].append(f"Setup: {str(e)[:200]}")

    # Summary
    print(f"\n  {'─'*40}")
    print(f"  Results: {results['passed']}/{results['total']} passed, {results['failed']} failed")
    if results["errors"]:
        print(f"  Errors:")
        for err in results["errors"]:
            print(f"    - {err[:100]}")

    return results


def main():
    print("=" * 60)
    print("  GENERIC AI AGENT — SMOKE TEST")
    print("=" * 60)

    # Check API keys
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\n[ERROR] No API keys found. Set GEMINI_API_KEY or OPENAI_API_KEY in .env")
        sys.exit(1)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_dir = os.path.join(base_dir, "sample_pdfs")

    all_results = []

    # Test 1: Pizza Restaurant
    pizza_file = os.path.join(sample_dir, "mario_pizza.md")
    if not os.path.exists(pizza_file):
        pizza_file = os.path.join(sample_dir, "mario_pizza.pdf")
    r1 = run_test_suite(pizza_file, PIZZA_SCENARIOS, "Mario Pizza")
    all_results.append(r1)

    # Test 2: Dental Clinic
    dental_file = os.path.join(sample_dir, "bright_smile_dental.md")
    if not os.path.exists(dental_file):
        dental_file = os.path.join(sample_dir, "bright_smile_dental.pdf")
    r2 = run_test_suite(dental_file, DENTAL_SCENARIOS, "Bright Smile Dental")
    all_results.append(r2)

    # Final summary
    total_tests = sum(r["total"] for r in all_results)
    total_passed = sum(r["passed"] for r in all_results)
    total_failed = sum(r["failed"] for r in all_results)

    print("\n" + "=" * 60)
    print(f"  FINAL SUMMARY: {total_passed}/{total_tests} passed, {total_failed} failed")
    print("=" * 60)

    # Save results
    results_path = os.path.join(os.path.dirname(__file__), "test_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {results_path}")

    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()
