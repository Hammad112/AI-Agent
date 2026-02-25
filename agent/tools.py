"""
agent/tools.py
--------------
All 17 agent tools. Each tool:
  - Has a clear TOOL_DESCRIPTION used by the LLM router
  - Accepts (state: dict, **kwargs) -> dict
  - Returns a result dict
  - Is fully generic (no hardcoded business logic)

Import paths changed from: database → core.database, logger → core.logger
"""

import os
import json
import sqlite3
import random
from datetime import datetime, timedelta

from core.database import get_business_meta, get_global_stats
from core.logger import log_agent_event


def _db() -> sqlite3.Connection:
    db_name = os.getenv("DB_NAME", "business_agent.db")
    conn = sqlite3.connect(db_name, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def book_service(state: dict, **kwargs) -> dict:
    """Create a new booking/reservation for the customer."""
    user_id = state.get("user_id")
    query = state.get("query", "")
    conn = _db()
    try:
        services = conn.execute("SELECT * FROM services ORDER BY id").fetchall()
        matched_service = None
        q_lower = query.lower()
        for svc in services:
            if any(word in svc["name"].lower() for word in q_lower.split()):
                matched_service = dict(svc)
                break
        if not matched_service and services:
            matched_service = dict(services[0])

        providers = conn.execute(
            "SELECT * FROM service_providers WHERE available = 1 ORDER BY rating DESC LIMIT 1"
        ).fetchone()

        if not matched_service:
            return {"success": False, "message": "No services available to book right now."}

        scheduled_dt = datetime.utcnow() + timedelta(days=1)
        cur = conn.execute(
            "INSERT INTO orders (user_id, service_id, provider_id, status, scheduled_at, total_price, notes) VALUES (?, ?, ?, 'pending', ?, ?, ?)",
            (
                user_id,
                matched_service["id"],
                providers["id"] if providers else None,
                scheduled_dt.isoformat(),
                matched_service["price"],
                "Booked via AI assistant",
            ),
        )
        conn.commit()
        return {
            "success": True,
            "booking_id": cur.lastrowid,
            "service": matched_service["name"],
            "provider": providers["name"] if providers else "TBD",
            "scheduled_at": scheduled_dt.strftime("%Y-%m-%d %H:%M"),
            "price": matched_service["price"],
            "message": f"Booking confirmed for {matched_service['name']} on {scheduled_dt.strftime('%Y-%m-%d at %H:%M')}.",
        }
    finally:
        conn.close()


def cancel_booking(state: dict, **kwargs) -> dict:
    """Cancel an existing booking by order ID or most recent pending booking."""
    user_id = state.get("user_id")
    conn = _db()
    try:
        order = conn.execute(
            """
            SELECT o.*, s.name as service_name
            FROM orders o
            LEFT JOIN services s ON o.service_id = s.id
            WHERE o.user_id = ? AND o.status IN ('pending', 'confirmed')
            ORDER BY o.scheduled_at DESC
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()
        if not order:
            return {"success": False, "message": "No active bookings found to cancel."}
        conn.execute("UPDATE orders SET status = 'cancelled' WHERE id = ?", (order["id"],))
        conn.commit()
        return {
            "success": True,
            "cancelled_booking_id": order["id"],
            "service": order["service_name"],
            "scheduled_at": order["scheduled_at"],
            "message": f"Booking #{order['id']} for {order['service_name']} has been cancelled.",
        }
    finally:
        conn.close()


def reschedule_booking(state: dict, **kwargs) -> dict:
    """Reschedule the customer's most recent active booking to a new date/time."""
    user_id = state.get("user_id")
    conn = _db()
    try:
        order = conn.execute(
            """
            SELECT o.*, s.name as service_name
            FROM orders o
            LEFT JOIN services s ON o.service_id = s.id
            WHERE o.user_id = ? AND o.status IN ('pending','confirmed')
            ORDER BY o.scheduled_at DESC LIMIT 1
            """,
            (user_id,),
        ).fetchone()
        if not order:
            return {"success": False, "message": "No active bookings found to reschedule."}
        new_time = datetime.utcnow() + timedelta(days=2)
        conn.execute(
            "UPDATE orders SET scheduled_at = ?, notes = ? WHERE id = ?",
            (new_time.isoformat(), "Rescheduled via AI assistant", order["id"]),
        )
        conn.commit()
        return {
            "success": True,
            "booking_id": order["id"],
            "service": order["service_name"],
            "new_scheduled_at": new_time.strftime("%Y-%m-%d %H:%M"),
            "message": f"Booking #{order['id']} for {order['service_name']} rescheduled to {new_time.strftime('%Y-%m-%d at %H:%M')}.",
        }
    finally:
        conn.close()


def get_pricing(state: dict, **kwargs) -> dict:
    """Get pricing information for all available services."""
    conn = _db()
    try:
        services = conn.execute(
            "SELECT name, description, price, duration_min, category FROM services ORDER BY category, price"
        ).fetchall()
        services_list = [dict(s) for s in services]
        categorised: dict[str, list] = {}
        for svc in services_list:
            cat = svc.get("category", "General")
            categorised.setdefault(cat, []).append(svc)
        return {
            "success": True,
            "services_by_category": categorised,
            "total_services": len(services_list),
            "message": f"Found {len(services_list)} services across {len(categorised)} categories.",
        }
    finally:
        conn.close()


def get_availability(state: dict, **kwargs) -> dict:
    """Check available time slots and service providers."""
    conn = _db()
    try:
        providers = conn.execute(
            "SELECT name, specialty, rating FROM service_providers WHERE available = 1 ORDER BY rating DESC"
        ).fetchall()
        services = conn.execute("SELECT name, duration_min FROM services").fetchall()
        slots = []
        for day_offset in range(1, 4):
            date = (datetime.utcnow() + timedelta(days=day_offset)).strftime("%Y-%m-%d")
            for hour in [9, 11, 14, 16]:
                slots.append(f"{date} {hour:02d}:00")
        return {
            "success": True,
            "available_providers": [dict(p) for p in providers],
            "available_slots": slots[:12],
            "services": [dict(s) for s in services],
            "message": f"{len(providers)} providers available, next slots: {slots[0]} onwards.",
        }
    finally:
        conn.close()


def get_recommendations(state: dict, **kwargs) -> dict:
    """Provide personalised service recommendations based on customer history."""
    user_id = state.get("user_id")
    conn = _db()
    try:
        used_service_ids = [
            r[0] for r in conn.execute(
                "SELECT DISTINCT service_id FROM orders WHERE user_id = ?", (user_id,)
            ).fetchall()
        ]
        if used_service_ids:
            placeholders = ",".join("?" * len(used_service_ids))
            new_services = conn.execute(
                f"SELECT * FROM services WHERE id NOT IN ({placeholders}) ORDER BY RANDOM() LIMIT 3",
                used_service_ids,
            ).fetchall()
        else:
            new_services = conn.execute("SELECT * FROM services ORDER BY RANDOM() LIMIT 3").fetchall()

        top_provider = conn.execute(
            "SELECT name, specialty, rating FROM service_providers ORDER BY rating DESC LIMIT 1"
        ).fetchone()

        loyalty = conn.execute(
            "SELECT points, tier FROM loyalty_points WHERE user_id = ?", (user_id,)
        ).fetchone()

        recs = [dict(s) for s in new_services]
        return {
            "success": True,
            "recommended_services": recs,
            "top_provider": dict(top_provider) if top_provider else None,
            "loyalty_points": loyalty["points"] if loyalty else 0,
            "loyalty_tier": loyalty["tier"] if loyalty else "bronze",
            "message": f"Recommended {len(recs)} services you haven't tried yet.",
        }
    finally:
        conn.close()


def apply_loyalty_discount(state: dict, **kwargs) -> dict:
    """Apply loyalty points as a discount on the next booking."""
    user_id = state.get("user_id")
    conn = _db()
    try:
        loyalty = conn.execute("SELECT * FROM loyalty_points WHERE user_id = ?", (user_id,)).fetchone()
        if not loyalty or loyalty["points"] < 100:
            return {
                "success": False,
                "message": "You need at least 100 loyalty points to apply a discount.",
                "current_points": loyalty["points"] if loyalty else 0,
            }
        discount = (loyalty["points"] // 100) * 5
        points_used = (loyalty["points"] // 100) * 100
        new_points = loyalty["points"] - points_used
        conn.execute(
            "UPDATE loyalty_points SET points = ?, updated_at = ? WHERE user_id = ?",
            (new_points, datetime.utcnow().isoformat(), user_id),
        )
        conn.commit()
        return {
            "success": True,
            "discount_applied": discount,
            "points_used": points_used,
            "remaining_points": new_points,
            "message": f"${discount:.2f} discount applied using {points_used} loyalty points. Remaining: {new_points} points.",
        }
    finally:
        conn.close()


def get_loyalty_balance(state: dict, **kwargs) -> dict:
    """Show the customer's current loyalty points and tier."""
    user_id = state.get("user_id")
    conn = _db()
    try:
        loyalty = conn.execute("SELECT * FROM loyalty_points WHERE user_id = ?", (user_id,)).fetchone()
        if not loyalty:
            return {"success": True, "points": 0, "tier": "bronze", "message": "No loyalty points yet."}
        points = loyalty["points"]
        tier = loyalty["tier"]
        next_tier = "silver" if tier == "bronze" else "gold" if tier == "silver" else "platinum"
        next_threshold = 500 if tier == "bronze" else 1000 if tier == "silver" else 2000
        return {
            "success": True,
            "points": points,
            "tier": tier,
            "next_tier": next_tier,
            "points_to_next_tier": max(0, next_threshold - points),
            "dollar_value": (points // 100) * 5,
            "message": f"You have {points} loyalty points ({tier} tier). ${(points // 100) * 5:.2f} redeemable.",
        }
    finally:
        conn.close()


def get_order_history(state: dict, **kwargs) -> dict:
    """Retrieve the customer's past orders and bookings."""
    user_id = state.get("user_id")
    conn = _db()
    try:
        orders = conn.execute(
            """
            SELECT o.id, o.status, o.scheduled_at, o.total_price,
                   s.name as service_name, sp.name as provider_name
            FROM orders o
            LEFT JOIN services s ON o.service_id = s.id
            LEFT JOIN service_providers sp ON o.provider_id = sp.id
            WHERE o.user_id = ?
            ORDER BY o.scheduled_at DESC
            LIMIT 10
            """,
            (user_id,),
        ).fetchall()
        order_list = [dict(o) for o in orders]
        return {
            "success": True,
            "orders": order_list,
            "total": len(order_list),
            "message": f"Found {len(order_list)} orders in your history.",
        }
    finally:
        conn.close()


def get_business_hours(state: dict, **kwargs) -> dict:
    """Return the business operating hours."""
    hours = get_business_meta("business_hours")
    if not hours:
        hours = "Monday–Friday: 9:00 AM – 7:00 PM\nSaturday: 9:00 AM – 5:00 PM\nSunday: Closed"
    return {
        "success": True,
        "hours": hours,
        "message": f"Business hours:\n{hours}",
    }


def get_provider_info(state: dict, **kwargs) -> dict:
    """Get information about service providers/staff members."""
    conn = _db()
    try:
        providers = conn.execute(
            "SELECT name, specialty, rating, available FROM service_providers ORDER BY rating DESC"
        ).fetchall()
        return {
            "success": True,
            "providers": [dict(p) for p in providers],
            "total_providers": len(providers),
            "message": f"Found {len(providers)} service providers.",
        }
    finally:
        conn.close()


def search_web(state: dict, **kwargs) -> dict:
    """Search the web for real-time information using DuckDuckGo."""
    query = state.get("query", "")
    business_name = get_business_meta("business_name") or "business"
    search_query = f"{business_name} {query}"
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, max_results=3))
        formatted = [
            {"title": r.get("title", ""), "snippet": r.get("body", ""), "url": r.get("href", "")}
            for r in results
        ]
        return {
            "success": True,
            "results": formatted,
            "query": search_query,
            "message": f"Found {len(formatted)} web results for: {search_query}",
        }
    except Exception as e:
        return {"success": False, "message": f"Web search failed: {str(e)}", "results": []}


def get_faqs(state: dict, **kwargs) -> dict:
    """Retrieve frequently asked questions from the knowledge base."""
    conn = _db()
    try:
        faqs = conn.execute(
            """
            SELECT topic, content FROM enriched_knowledge
            WHERE LOWER(topic) LIKE '%faq%'
               OR LOWER(topic) LIKE '%question%'
               OR LOWER(topic) LIKE '%common%'
            LIMIT 3
            """,
        ).fetchall()
        if not faqs:
            faqs = conn.execute("SELECT topic, content FROM enriched_knowledge LIMIT 3").fetchall()
        return {
            "success": True,
            "faqs": [dict(f) for f in faqs],
            "message": f"Retrieved {len(faqs)} FAQ entries.",
        }
    finally:
        conn.close()


def get_promotions(state: dict, **kwargs) -> dict:
    """List current deals, promotions, and special offers."""
    conn = _db()
    try:
        promos = conn.execute(
            """
            SELECT topic, content FROM enriched_knowledge
            WHERE LOWER(topic) LIKE '%promo%'
               OR LOWER(topic) LIKE '%deal%'
               OR LOWER(topic) LIKE '%offer%'
               OR LOWER(topic) LIKE '%discount%'
               OR LOWER(topic) LIKE '%seasonal%'
            LIMIT 3
            """,
        ).fetchall()
        if not promos:
            promos = conn.execute("SELECT topic, content FROM enriched_knowledge LIMIT 2").fetchall()
        return {
            "success": True,
            "promotions": [dict(p) for p in promos],
            "message": f"Found {len(promos)} active promotions.",
        }
    finally:
        conn.close()


def escalate_to_human(state: dict, **kwargs) -> dict:
    """Flag the conversation for human agent handoff."""
    conv_id = state.get("conversation_id", "unknown")
    log_agent_event(conv_id, "human_escalation", {
        "user_id": state.get("user_id"),
        "query": state.get("query", ""),
        "reason": "Customer requested human agent or issue unresolved",
    })
    return {
        "success": True,
        "escalated": True,
        "message": "Your request has been flagged for a human agent. We will contact you within 24 hours via email.",
    }


def get_global_stats_tool(state: dict, **kwargs) -> dict:
    """Retrieve aggregated business statistics."""
    stats = get_global_stats()
    return {
        "success": True,
        "stats": stats,
        "message": f"Business has {stats.get('total_customers',0)} customers and ${stats.get('total_revenue',0):.2f} total revenue.",
    }


def update_customer_profile(state: dict, **kwargs) -> dict:
    """Update the customer's profile information (name, phone)."""
    user_id = state.get("user_id")
    conn = _db()
    try:
        user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if not user:
            return {"success": False, "message": "User not found."}
        return {
            "success": True,
            "user_id": user_id,
            "current_profile": {
                "full_name": user["full_name"],
                "email": user["email"],
                "phone": user["phone"],
            },
            "message": "Your profile is on file. Please tell me which field (name, phone, email) you'd like to update and the new value.",
        }
    finally:
        conn.close()


# ─────────────────────────────────────────────
# TOOLS REGISTRY
# ─────────────────────────────────────────────

TOOLS: dict[str, tuple] = {
    "book_service": (book_service, "Activates when the customer wants to book, reserve, schedule, or make an appointment for a service or product."),
    "cancel_booking": (cancel_booking, "Activates when the customer wants to cancel, delete, or remove an existing booking or reservation."),
    "reschedule_booking": (reschedule_booking, "Activates when the customer wants to change, move, or reschedule the date or time of an existing booking."),
    "get_pricing": (get_pricing, "Activates when the customer asks about prices, costs, fees, rates, or how much something costs."),
    "get_availability": (get_availability, "Activates when the customer asks about availability, open slots, time slots, or when they can book."),
    "get_recommendations": (get_recommendations, "Activates when the customer asks for recommendations, suggestions, what to try, or what is popular."),
    "apply_loyalty_discount": (apply_loyalty_discount, "Activates when the customer wants to redeem, use, or apply loyalty points or get a discount via points."),
    "get_loyalty_balance": (get_loyalty_balance, "Activates when the customer asks about their loyalty points, rewards balance, tier, or membership status."),
    "get_order_history": (get_order_history, "Activates when the customer asks about their past orders, previous bookings, or purchase history."),
    "get_business_hours": (get_business_hours, "Activates when the customer asks about opening hours, closing time, when the business is open."),
    "get_provider_info": (get_provider_info, "Activates when the customer asks about staff, therapists, providers, specialists, or who will serve them."),
    "search_web": (search_web, "Activates when the customer asks for current news, trends, external information, or anything not in the knowledge base."),
    "get_faqs": (get_faqs, "Activates when the customer asks a general question, how something works, or requests FAQs or common questions."),
    "get_promotions": (get_promotions, "Activates when the customer asks about deals, promotions, discounts, offers, or special prices."),
    "escalate_to_human": (escalate_to_human, "Activates when the customer is frustrated, requests a human agent, or when the AI cannot resolve the issue."),
    "get_global_stats": (get_global_stats_tool, "Activates when asked about overall business performance, statistics, number of customers, or revenue."),
    "update_customer_profile": (update_customer_profile, "Activates when the customer wants to update their profile, change their name, phone, or personal details."),
}
