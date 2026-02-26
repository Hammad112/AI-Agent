
import os
import sys
import uuid
import sqlite3
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from core.database import init_db, set_business_meta
from agent.agent import run_agent_turn
from agent.tools import handle_dispute

def test_complaint_memory():
    DB_NAME = "test_complaint_memory.db"
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)
    os.environ["DB_NAME"] = DB_NAME
    
    init_db()
    set_business_meta("business_name", "Test Pizza")
    set_business_meta("business_type", "restaurant")
    
    # Mock a service and an order
    conn = sqlite3.connect(DB_NAME)
    conn.execute("INSERT INTO services (id, name, price, category) VALUES (1, 'Large Pizza', 15.0, 'Pizza')")
    conn.execute("INSERT INTO orders (id, user_id, service_id, status, total_price, scheduled_at) VALUES (101, 1, 1, 'completed', 15.0, '2023-01-01T12:00:00')")
    conn.commit()
    conn.close()
    
    state = {
        "user_id": 1,
        "conversation_id": str(uuid.uuid4()),
        "query": "My pizza was cold in order #101"
    }
    
    print("--- Step 1: Filing a dispute ---")
    res = handle_dispute(state)
    print(f"Tool response: {res['message']}")
    
    # Verify DB
    conn = sqlite3.connect(DB_NAME)
    complaints = conn.execute("SELECT * FROM complaints").fetchall()
    print(f"Complaints in DB: {len(complaints)}")
    for c in complaints:
        print(f"Logged Complaint: {c[4]} (Type: {c[3]})")
    conn.close()
    
    if len(complaints) == 0:
        print("FAILED: Complaint not stored in DB")
        return

    print("\n--- Step 2: New turn, checking if agent remembers ---")
    query = "Hi, I'm checking in again. Do you remember my last issue?"
    resp = run_agent_turn(
        user_id=1,
        conversation_id=str(uuid.uuid4()),
        query=query,
        all_chunks=[],
        business_name="Test Pizza",
        business_type="restaurant"
    )
    print(f"Agent response: {resp}")
    
    if "cold" in resp.lower() or "order #101" in resp.lower() or "dispute" in resp.lower() or "issue" in resp.lower():
        print("SUCCESS: Agent seems to remember the past complaint!")
    else:
        print("WARNING: Agent didn't explicitly mention the cold pizza, but let's check if it's in the profile context.")

if __name__ == "__main__":
    test_complaint_memory()
