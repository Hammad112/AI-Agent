
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

def test_single_turn_add_and_confirm():
    DB_NAME = "test_regression.db"
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)
    os.environ["DB_NAME"] = DB_NAME
    
    init_db()
    set_business_meta("business_name", "Test Pizza")
    set_business_meta("business_type", "restaurant")
    
    # Mock some services
    conn = sqlite3.connect(DB_NAME)
    conn.execute("INSERT INTO services (name, price, category) VALUES ('Large Pepperoni Pizza', 15.99, 'Pizza')")
    conn.commit()
    conn.close()
    
    user_id = 1
    session_id = str(uuid.uuid4())
    
    print(f"--- Single Turn: Add and Confirm ---")
    query = "Add a large pepperoni pizza and confirm my order please"
    
    # We expect the agent to:
    # 1. Detect 'order_item' and 'confirm_order' intents
    # 2. Activate 'add_to_cart' and 'confirm_order' tools
    # 3. Execute 'add_to_cart' FIRST (priority 1)
    # 4. Execute 'confirm_order' SECOND (priority 9)
    # 5. Respond with a successful order ID
    
    resp = run_agent_turn(
        user_id=user_id,
        conversation_id=session_id,
        query=query,
        all_chunks=[],
        business_name="Test Pizza",
        business_type="restaurant"
    )
    
    print(f"Agent Response: {resp}")
    
    # Verify order in DB
    conn = sqlite3.connect(DB_NAME)
    orders = conn.execute("SELECT * FROM orders").fetchall()
    cart = conn.execute("SELECT * FROM cart").fetchall()
    conn.close()
    
    print(f"Orders in DB: {len(orders)}")
    print(f"Cart in DB: {len(cart)}")
    
    if len(orders) > 0 and len(cart) == 0:
        print("SUCCESS: Single-turn Add and Confirm worked correctly!")
    else:
        print(f"FAILED: Expected order count > 0 (got {len(orders)}) and cart count 0 (got {len(cart)})")

if __name__ == "__main__":
    test_single_turn_add_and_confirm()
