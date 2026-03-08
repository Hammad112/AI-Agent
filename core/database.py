"""
core/database.py
----------------
SQLite schema creation and synthetic data generation.
Includes: cart system, address fields, calendar events.
"""

import sqlite3
import os
import json
import random
import bcrypt
from datetime import datetime, timezone, timedelta
from faker import Faker
from core.llm_client import llm_call

fake = Faker()


SCHEMA_SQL = """
-- Users (customers who can log in)
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    username    TEXT    UNIQUE NOT NULL,
    email       TEXT    UNIQUE NOT NULL,
    password_hash TEXT  NOT NULL,
    full_name   TEXT,
    phone       TEXT,
    address     TEXT,
    postal_code TEXT,
    city        TEXT,
    family_members TEXT,
    created_at  TEXT    DEFAULT (datetime('now'))
);

-- Service providers / staff
CREATE TABLE IF NOT EXISTS service_providers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    business_name TEXT,
    name        TEXT    NOT NULL,
    specialty   TEXT,
    rating      REAL    DEFAULT 4.5,
    available   INTEGER DEFAULT 1,
    schedule    TEXT
);

-- Services / products offered by the business
CREATE TABLE IF NOT EXISTS services (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    business_name TEXT,
    name        TEXT    NOT NULL,
    description TEXT,
    price       REAL    NOT NULL,
    duration_min INTEGER,
    category    TEXT,
    modifiers   TEXT
);

-- Shopping cart (active order being built)
CREATE TABLE IF NOT EXISTS cart (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    business_name TEXT,
    user_id     INTEGER REFERENCES users(id),
    session_id  TEXT    NOT NULL,
    service_id  INTEGER REFERENCES services(id),
    service_name TEXT,
    quantity    INTEGER DEFAULT 1,
    unit_price  REAL,
    modifiers   TEXT,
    notes       TEXT,
    created_at  TEXT    DEFAULT (datetime('now'))
);

-- Confirmed orders
CREATE TABLE IF NOT EXISTS orders (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER REFERENCES users(id),
    service_id      INTEGER REFERENCES services(id),
    provider_id     INTEGER REFERENCES service_providers(id),
    status          TEXT    DEFAULT 'completed',
    order_type      TEXT    DEFAULT 'order',
    scheduled_at    TEXT,
    completed_at    TEXT,
    notes           TEXT,
    total_price     REAL,
    delivery_type   TEXT,
    delivery_address TEXT,
    delivery_postal TEXT,
    items_json      TEXT
);

-- Calendar events (local .ics compatible)
CREATE TABLE IF NOT EXISTS calendar_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER REFERENCES users(id),
    order_id    INTEGER REFERENCES orders(id),
    title       TEXT    NOT NULL,
    description TEXT,
    start_time  TEXT    NOT NULL,
    end_time    TEXT    NOT NULL,
    location    TEXT,
    provider    TEXT,
    status      TEXT    DEFAULT 'confirmed',
    created_at  TEXT    DEFAULT (datetime('now'))
);

-- Loyalty programme
CREATE TABLE IF NOT EXISTS loyalty_points (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    business_name TEXT,
    user_id     INTEGER REFERENCES users(id),
    points      INTEGER DEFAULT 0,
    tier        TEXT    DEFAULT 'bronze',
    updated_at  TEXT    DEFAULT (datetime('now')),
    UNIQUE(user_id, business_name)
);

-- PDF knowledge chunks
CREATE TABLE IF NOT EXISTS pdf_chunks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    business_name   TEXT,
    chunk_index     INTEGER,
    page_num        INTEGER,
    section_title   TEXT,
    text            TEXT    NOT NULL,
    topic_tags      TEXT,
    created_at      TEXT    DEFAULT (datetime('now'))
);

-- LLM-enriched supplementary knowledge
CREATE TABLE IF NOT EXISTS enriched_knowledge (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    business_name TEXT,
    topic       TEXT    NOT NULL,
    content     TEXT    NOT NULL,
    source      TEXT    DEFAULT 'llm_enrichment',
    topic_tags  TEXT,
    created_at  TEXT    DEFAULT (datetime('now'))
);

-- Customer complaints and disputes
CREATE TABLE IF NOT EXISTS complaints (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER REFERENCES users(id),
    order_id        INTEGER REFERENCES orders(id),
    complaint_type  TEXT,
    description     TEXT,
    suggested_resolution TEXT,
    status          TEXT    DEFAULT 'open',
    created_at      TEXT    DEFAULT (datetime('now'))
);

-- Chat messages
CREATE TABLE IF NOT EXISTS conversations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    business_name   TEXT,
    conversation_id TEXT    NOT NULL,
    role            TEXT    NOT NULL,
    content         TEXT    NOT NULL,
    timestamp       TEXT    NOT NULL
);

-- Tool call log
CREATE TABLE IF NOT EXISTS tool_calls (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT    NOT NULL,
    tool_name       TEXT    NOT NULL,
    inputs          TEXT,
    outputs         TEXT,
    activated       INTEGER DEFAULT 1,
    timestamp       TEXT    NOT NULL
);

-- Generic agent events
CREATE TABLE IF NOT EXISTS agent_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT    NOT NULL,
    event_type      TEXT    NOT NULL,
    details         TEXT,
    timestamp       TEXT    NOT NULL
);

-- Business metadata
CREATE TABLE IF NOT EXISTS business_meta (
    key     TEXT PRIMARY KEY,
    value   TEXT
);

-- Conversation state (for multi-turn slot filling)
CREATE TABLE IF NOT EXISTS conversation_state (
    conversation_id TEXT PRIMARY KEY,
    state_json      TEXT NOT NULL,
    updated_at      TEXT DEFAULT (datetime('now'))
);
"""


def init_db() -> None:
    conn = _get_db()
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def _get_db() -> sqlite3.Connection:
    db_name = os.getenv("DB_NAME", "business_agent.db")
    conn = sqlite3.connect(db_name, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def set_business_meta(key: str, value: str) -> None:
    conn = _get_db()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO business_meta (key, value) VALUES (?, ?)",
            (key, value),
        )
        conn.commit()
    finally:
        conn.close()


def get_business_meta(key: str) -> str | None:
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT value FROM business_meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None
    finally:
        conn.close()


def _already_seeded(business_name: str) -> bool:
    conn = _get_db()
    try:
        key = f"seeded_{business_name.lower().replace(' ', '_')}"
        row = conn.execute(
            "SELECT value FROM business_meta WHERE key = ?", (key,)
        ).fetchone()
        return row is not None and row["value"] == "1"
    finally:
        conn.close()


# ── Conversation state persistence ──────────────

def save_conversation_state(conversation_id: str, state: dict) -> None:
    conn = _get_db()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO conversation_state (conversation_id, state_json, updated_at) VALUES (?, ?, ?)",
            (conversation_id, json.dumps(state, default=str), datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def load_conversation_state(conversation_id: str) -> dict | None:
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT state_json FROM conversation_state WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        return json.loads(row["state_json"]) if row else None
    finally:
        conn.close()


# ── Synthetic data generation ──────────────

def generate_synthetic_data(business_type: str, business_name: str, chunks: list[dict] = None) -> None:
    if _already_seeded(business_name):
        return

    context_text = ""
    if chunks:
        # Use first few chunks that likely contain services/pricing
        context_text = "\n".join(c.get("text", "")[:500] for c in chunks[:10])

    prompt = f"""
You are a database seeder. The business is: "{business_name}" ({business_type}).
Here is some context about the business from its documentation:
{context_text}

Return ONLY a JSON object with these keys:
{{
  "services": [
    {{"name": "...", "description": "...", "price": 0.00, "duration_min": 0, "category": "...", "modifiers": "size:small/medium/large;extras:cheese/bacon"}}
    // 10-12 items
  ],
  "provider_specialties": ["...", "...", "...", "...", "..."],
  "order_status_options": ["completed", "completed", "completed", "cancelled", "pending"]
}}

Make prices, durations, names, and modifiers realistic for this type of business. 
CRITICAL: Ensure prices are typical for the industry (e.g., pizzas should be $10-$25, not $100+).
Include modifier options where they make sense (sizes, toppings, extras, options).
Respond with only the JSON, no markdown.
"""
    raw = llm_call(prompt)
    import re
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Better fallbacks based on type with realistic pricing
        if "pizza" in business_type.lower() or "restaurant" in business_type.lower():
            fallback_items = [
                ("Margherita Pizza", 12.00, 18.00), ("Pepperoni Pizza", 14.00, 20.00),
                ("Caesar Salad", 8.00, 12.00), ("Garlic Bread", 4.00, 7.00),
                ("Sprite", 2.50, 3.50), ("Coke", 2.50, 3.50),
                ("Pasta Carbonara", 15.00, 22.00), ("Tiramisu", 7.00, 10.00)
            ]
        elif "dental" in business_type.lower() or "clinic" in business_type.lower():
            fallback_items = [
                ("Dental Cleaning", 80.00, 150.00), ("Teeth Whitening", 150.00, 400.00),
                ("Checkup", 50.00, 100.00), ("X-Ray", 40.00, 90.00),
                ("Consultation", 30.00, 80.00), ("Filling", 100.00, 250.00),
                ("Crown", 800.00, 1500.00), ("Root Canal", 600.00, 1200.00)
            ]
        elif "cleaner" in business_type.lower() or "laundry" in business_type.lower():
             fallback_items = [
                ("Shirt Laundry", 3.00, 6.00), ("Suit Dry Clean", 15.00, 25.00),
                ("Dress Dry Clean", 12.00, 22.00), ("Pants Dry Clean", 7.00, 12.00),
                ("Coat Dry Clean", 18.00, 35.00), ("Wash & Fold (lb)", 1.50, 2.50)
            ]
        else:
            fallback_items = [(f"General Service {i}", 20.00, 100.00) for i in range(1, 9)]
        
        data = {
            "services": [
                {
                    "name": name, 
                    "description": "Standard service", 
                    "price": round(random.uniform(min_p, max_p), 2), 
                    "duration_min": 30, 
                    "category": "General", 
                    "modifiers": ""
                }
                for name, min_p, max_p in fallback_items
            ],
            "provider_specialties": ["General", "Senior", "Expert"],
            "order_status_options": ["completed", "completed", "completed", "cancelled", "pending"],
        }

    conn = _get_db()
    try:
        service_ids = []
        for svc in data.get("services", []):
            cur = conn.execute(
                "INSERT INTO services (business_name, name, description, price, duration_min, category, modifiers) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    business_name,
                    svc.get("name", "Service"),
                    svc.get("description", ""),
                    float(svc.get("price", 50.0)),
                    int(svc.get("duration_min", 30)),
                    svc.get("category", "General"),
                    svc.get("modifiers", ""),
                ),
            )
            service_ids.append(cur.lastrowid)

        specialties = data.get("provider_specialties", ["General"] * 5)
        provider_ids = []
        provider_names = []
        # Default schedule logic based on business type
        default_sched = {"mon": "9-17", "tue": "9-17", "wed": "9-17", "thu": "9-17", "fri": "9-17"}
        if "dental" in business_type.lower() or "clinic" in business_type.lower():
             # Align with RAG knowledge for Bright Smile
             default_sched = {"mon": "9-17", "wed": "9-17", "fri": "9-17", "sat": "9-13"}
        elif "pizza" in business_type.lower() or "restaurant" in business_type.lower():
             default_sched = {"mon": "11-23", "tue": "11-23", "wed": "11-23", "thu": "11-23", "fri": "11-23", "sat": "11-23", "sun": "11-23"}

        for spec in specialties:
            name = fake.name()
            # Special case: Ensure Dr. Sarah Chen is in the DB if it's dental
            if "dental" in business_type.lower() and spec == specialties[0]:
                 name = "Dr. Sarah Chen"

            provider_names.append(name)
            cur = conn.execute(
                "INSERT INTO service_providers (business_name, name, specialty, rating, available, schedule) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    business_name,
                    name,
                    spec,
                    round(random.uniform(3.8, 5.0), 1),
                    1 if name == "Dr. Sarah Chen" else random.choice([1, 1, 1, 0]),
                    json.dumps(default_sched),
                ),
            )
            provider_ids.append(cur.lastrowid)

        # Create synthetic users with family data
        user_ids = []
        families = [
            ([("spouse", "Female"), ("child", "Male")], "married_with_child"),
            ([("spouse", "Male")], "married"),
            ([("child", "Female"), ("child", "Male")], "children"),
            ([("spouse", "Female"), ("child", "Female"), ("child", "Male")], "large_family"),
            ([], "single"),
            ([("parent", "Female")], "parent_on_file"),
        ]
        for _ in range(20):
            pw_hash = bcrypt.hashpw(b"password123", bcrypt.gensalt()).decode()
            family_info, _label = random.choice(families)
            family_json = json.dumps([
                {"relation": r, "gender": g, "name": fake.first_name_female() if g == "Female" else fake.first_name_male()} 
                for r, g in family_info
            ]) if family_info else "[]"
            cur = conn.execute(
                "INSERT OR IGNORE INTO users (username, email, password_hash, full_name, phone, address, postal_code, city, family_members) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    fake.user_name(),
                    fake.email(),
                    pw_hash,
                    fake.name(),
                    fake.phone_number(),
                    fake.street_address(),
                    fake.postcode(),
                    fake.city(),
                    family_json,
                ),
            )
            if cur.lastrowid:
                user_ids.append(cur.lastrowid)

        statuses = data.get("order_status_options", ["completed"] * 5)
        for user_id in user_ids:
            num_orders = random.randint(1, 6)
            for _ in range(num_orders):
                days_ago = random.randint(1, 365)
                scheduled = datetime.utcnow() - timedelta(days=days_ago)
                svc_id = random.choice(service_ids)
                prov_id = random.choice(provider_ids)
                status = random.choice(statuses)
                svc_row = conn.execute("SELECT price, name FROM services WHERE id = ?", (svc_id,)).fetchone()
                price = svc_row["price"] if svc_row else 50.0
                delivery = random.choice(["pickup", "delivery", "in-person"])
                conn.execute(
                    "INSERT INTO orders (user_id, service_id, provider_id, status, order_type, scheduled_at, completed_at, notes, total_price, delivery_type, items_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        user_id,
                        svc_id,
                        prov_id,
                        status,
                        random.choice(["order", "appointment"]),
                        scheduled.isoformat(),
                        (scheduled + timedelta(hours=1)).isoformat() if status == "completed" else None,
                        fake.sentence(nb_words=6),
                        price,
                        delivery,
                        json.dumps([{"name": svc_row["name"], "qty": 1, "price": price}]) if svc_row else "[]",
                    ),
                )

        for user_id in user_ids:
            points = random.randint(0, 1500)
            tier = "gold" if points > 1000 else "silver" if points > 500 else "bronze"
            conn.execute(
                "INSERT OR IGNORE INTO loyalty_points (business_name, user_id, points, tier) VALUES (?, ?, ?, ?)",
                (business_name, user_id, points, tier),
            )

        key = f"seeded_{business_name.lower().replace(' ', '_')}"
        conn.execute("INSERT OR REPLACE INTO business_meta (key, value) VALUES (?, '1')", (key,))
        conn.commit()
    finally:
        conn.close()


def get_global_stats() -> dict:
    conn = _get_db()
    try:
        stats = {}
        stats["total_customers"] = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        stats["total_orders"] = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        stats["total_revenue"] = conn.execute("SELECT COALESCE(SUM(total_price),0) FROM orders WHERE status='completed'").fetchone()[0]
        stats["avg_order_value"] = conn.execute("SELECT COALESCE(AVG(total_price),0) FROM orders WHERE status='completed'").fetchone()[0]
        stats["total_providers"] = conn.execute("SELECT COUNT(*) FROM service_providers").fetchone()[0]
        stats["total_services"] = conn.execute("SELECT COUNT(*) FROM services").fetchone()[0]
        stats["completed_orders"] = conn.execute("SELECT COUNT(*) FROM orders WHERE status='completed'").fetchone()[0]
        stats["cancelled_orders"] = conn.execute("SELECT COUNT(*) FROM orders WHERE status='cancelled'").fetchone()[0]

        # Most popular service
        popular = conn.execute(
            "SELECT s.name, COUNT(o.id) as cnt FROM orders o JOIN services s ON o.service_id=s.id GROUP BY s.name ORDER BY cnt DESC LIMIT 1"
        ).fetchone()
        stats["most_popular_service"] = popular["name"] if popular else "N/A"

        # Most popular provider
        top_prov = conn.execute(
            "SELECT sp.name, COUNT(o.id) as cnt FROM orders o JOIN service_providers sp ON o.provider_id=sp.id GROUP BY sp.name ORDER BY cnt DESC LIMIT 1"
        ).fetchone()
        stats["most_popular_provider"] = top_prov["name"] if top_prov else "N/A"

        return stats
    finally:
        conn.close()
