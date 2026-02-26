import sqlite3
import os
import json

db_name = "business_agent.db"
if not os.path.exists(db_name):
    print(f"Database {db_name} not found.")
else:
    conn = sqlite3.connect(db_name)
    conn.row_factory = sqlite3.Row
    
    print("\n--- TOOL CALLS (Last 10) ---")
    rows = conn.execute("SELECT id, tool_name, inputs, activated FROM tool_calls ORDER BY id DESC LIMIT 10").fetchall()
    for row in rows:
        print(f"ID:{row['id']} | Tool:{row['tool_name']} | Inputs:{row['inputs']} | Activated:{row['activated']}")

    print("\n--- ORDERS (Last 5) ---")
    rows = conn.execute("SELECT * FROM orders ORDER BY id DESC LIMIT 5").fetchall()
    for row in rows:
        print(dict(row))
        
    print("\n--- CART (All) ---")
    rows = conn.execute("SELECT * FROM cart").fetchall()
    for row in rows:
        print(dict(row))
        
    conn.close()
