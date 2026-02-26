"""
agent/tools.py
--------------
All agent tools. Each tool:
  - Has a clear TOOL_DESCRIPTION used by the LLM router
  - Accepts (state: dict, **kwargs) -> dict
  - Returns a result dict
  - Is fully generic (no hardcoded business logic)

Tools include: cart management, booking, scheduling, calendar,
address validation, recommendations, loyalty, dispute handling, etc.
"""

import os
import json
import sqlite3
import random
import re
from datetime import datetime, timedelta, timezone

from core.database import get_business_meta, get_global_stats
from core.logger import log_agent_event
from core.llm_client import llm_call

try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False


def _db() -> sqlite3.Connection:
    db_name = os.getenv("DB_NAME", "business_agent.db")
    conn = sqlite3.connect(db_name, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _resolve_service_name(conn, service_id: int) -> str:
    """Helper to get a human-readable service name for any ID."""
    if not service_id:
        return "Unknown Service"
    row = conn.execute("SELECT name FROM services WHERE id = ?", (service_id,)).fetchone()
    return row["name"] if row else f"Service {service_id}"


def _generate_ref(prefix: str, db_id: int) -> str:
    """Generate a customer-facing reference number."""
    return f"{prefix}-{str(db_id).zfill(5)}"


# ─────────────────────────────────────────────
# Cart / Order Management Tools
# ─────────────────────────────────────────────

def add_to_cart(state: dict, **kwargs) -> dict:
    """Add an item to the customer's shopping cart."""
    user_id = state.get("user_id")
    query = state.get("query", "")
    session_id = state.get("conversation_id", "")
    conn = _db()
    try:
        services = conn.execute("SELECT * FROM services ORDER BY id").fetchall()
        if not services:
            return {"success": False, "message": "No services or items available."}

        # Use LLM to match the query to a service
        svc_list = "\n".join(
            f"- ID:{s['id']} | {s['name']} | ${s['price']:.2f} | {s['category'] or 'General'} | Modifiers: {s['modifiers'] or 'none'}"
            for s in services
        )
        match_prompt = f"""Given this customer request: "{query}"
And these available services/items:
{svc_list}

Which services/items does the customer want? Extract ALL items, their quantities, and any modifiers/options.
Return ONLY JSON: {{"items": [{{"service_id": 1, "quantity": 1, "modifiers": "large, extra cheese", "matched": true}}]}}
If no items match: {{"items": [{{"matched": false, "message": "reason"}}]}}
No markdown, just JSON."""

        raw = llm_call(match_prompt, temperature=0.1, max_tokens=400)
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            match_data = json.loads(raw)
            items_to_add = match_data.get("items", [])
        except json.JSONDecodeError:
            items_to_add = []

        if not items_to_add:
            return {"success": False, "message": "I couldn't identify any items to add. Could you be more specific?"}

        added_log = []
        for item in items_to_add:
            if not item.get("matched", False):
                continue
            
            svc_id = item.get("service_id")
            quantity = max(1, int(item.get("quantity", 1)))
            modifiers = item.get("modifiers", "")

            svc = conn.execute("SELECT * FROM services WHERE id = ?", (svc_id,)).fetchone()
            if not svc:
                continue

            conn.execute(
                "INSERT INTO cart (user_id, session_id, service_id, service_name, quantity, unit_price, modifiers, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, session_id, svc_id, svc["name"], quantity, svc["price"], modifiers, ""),
            )
            added_log.append(f"{quantity}x {svc['name']}")

        conn.commit()

        if not added_log:
             return {"success": False, "message": "I couldn't find those specific items in our menu."}

        # Get current cart summary
        cart_items = conn.execute(
            "SELECT * FROM cart WHERE user_id = ? AND session_id = ?",
            (user_id, session_id),
        ).fetchall()
        total = sum(item["quantity"] * item["unit_price"] for item in cart_items)

        return {
            "success": True,
            "added_items": added_log,
            "cart_items": [
                {"name": c["service_name"], "qty": c["quantity"], "price": c["unit_price"], "modifiers": c["modifiers"]}
                for c in cart_items
            ],
            "cart_total": round(total, 2),
            "message": f"Added {', '.join(added_log)} to your cart. Current total: ${total:.2f}",
        }
    finally:
        conn.close()


def remove_from_cart(state: dict, **kwargs) -> dict:
    """Remove an item from the customer's cart or clear the entire cart."""
    user_id = state.get("user_id")
    query = state.get("query", "")
    session_id = state.get("conversation_id", "")
    conn = _db()
    try:
        cart_items = conn.execute(
            "SELECT * FROM cart WHERE user_id = ? AND session_id = ?",
            (user_id, session_id),
        ).fetchall()

        if not cart_items:
            return {"success": False, "message": "Your cart is empty."}

        # Check if user wants to clear all
        q_lower = query.lower()
        if any(w in q_lower for w in ["clear", "empty", "remove all", "start over"]):
            conn.execute(
                "DELETE FROM cart WHERE user_id = ? AND session_id = ?",
                (user_id, session_id),
            )
            conn.commit()
            return {"success": True, "message": "Cart cleared. What would you like to add?", "cart_items": [], "cart_total": 0}

        # Use LLM to figure out which item to remove
        items_str = "\n".join(
            f"- CartID:{c['id']} | {c['service_name']} | qty:{c['quantity']}"
            for c in cart_items
        )
        match_prompt = f"""Customer says: "{query}"
Items in cart:
{items_str}
Which item should be removed? Return ONLY JSON: {{"cart_id": 1, "found": true}}
If unclear: {{"found": false, "message": "which item?"}}
No markdown, just JSON."""

        raw = llm_call(match_prompt, temperature=0.1, max_tokens=150)
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            match = json.loads(raw)
        except json.JSONDecodeError:
            match = {"found": False, "message": "Could not determine which item to remove."}

        if not match.get("found", False):
            return {"success": False, "message": match.get("message", "Which item would you like to remove?")}

        cart_id = match.get("cart_id")
        removed = conn.execute("SELECT service_name FROM cart WHERE id = ?", (cart_id,)).fetchone()
        conn.execute("DELETE FROM cart WHERE id = ?", (cart_id,))
        conn.commit()

        remaining = conn.execute(
            "SELECT * FROM cart WHERE user_id = ? AND session_id = ?",
            (user_id, session_id),
        ).fetchall()
        total = sum(item["quantity"] * item["unit_price"] for item in remaining)

        return {
            "success": True,
            "removed_item": removed["service_name"] if removed else "item",
            "cart_items": [
                {"name": c["service_name"], "qty": c["quantity"], "price": c["unit_price"], "modifiers": c["modifiers"]}
                for c in remaining
            ],
            "cart_total": round(total, 2),
            "message": f"Removed {removed['service_name'] if removed else 'item'} from your cart. Total: ${total:.2f}",
        }
    finally:
        conn.close()


def view_cart(state: dict, **kwargs) -> dict:
    """Show the customer's current cart contents."""
    user_id = state.get("user_id")
    session_id = state.get("conversation_id", "")
    conn = _db()
    try:
        cart_items = conn.execute(
            "SELECT * FROM cart WHERE user_id = ? AND session_id = ?",
            (user_id, session_id),
        ).fetchall()
        if not cart_items:
            return {"success": True, "message": "Your cart is empty.", "cart_items": [], "cart_total": 0}

        total = sum(item["quantity"] * item["unit_price"] for item in cart_items)
        return {
            "success": True,
            "cart_items": [
                {"name": c["service_name"], "qty": c["quantity"], "price": c["unit_price"], "modifiers": c["modifiers"]}
                for c in cart_items
            ],
            "cart_total": round(total, 2),
            "message": f"Your cart has {len(cart_items)} item(s) totaling ${total:.2f}.",
        }
    finally:
        conn.close()


def confirm_order(state: dict, **kwargs) -> dict:
    """Confirm and finalise the cart as a completed order."""
    user_id = state.get("user_id")
    session_id = state.get("conversation_id", "")
    conn = _db()
    try:
        # Check if cart exists for THIS session and user
        cart_items = conn.execute(
            "SELECT * FROM cart WHERE user_id = ? AND session_id = ?",
            (user_id, session_id),
        ).fetchall()
        
        if not cart_items:
            # Check if there are ANY items in cart for this user (could be different session)
            any_items = conn.execute("SELECT COUNT(*) FROM cart WHERE user_id = ?", (user_id,)).fetchone()[0]
            if any_items > 0:
                return {
                    "success": False, 
                    "message": "I found items in your cart from a different session, but your current session is empty. Please verify your order."
                }
            return {"success": False, "message": "Your cart is currently empty. What would you like to add?"}

        # FIX: Calculate total fresh by summing quantity * unit_price from the cart table
        total_row = conn.execute(
            "SELECT SUM(quantity * unit_price) as total FROM cart WHERE user_id = ? AND session_id = ?",
            (user_id, session_id),
        ).fetchone()
        total = total_row["total"] if total_row["total"] else 0.0

        items_json = json.dumps([
            {"name": c["service_name"], "qty": c["quantity"], "price": c["unit_price"], "modifiers": c["modifiers"]}
            for c in cart_items
        ])

        # Create the order
        cur = conn.execute(
            """INSERT INTO orders (user_id, service_id, status, order_type, scheduled_at, total_price, items_json, notes)
               VALUES (?, ?, 'confirmed', 'order', ?, ?, ?, ?)""",
            (
                user_id,
                cart_items[0]["service_id"],
                datetime.now(timezone.utc).isoformat(),
                round(total, 2),
                items_json,
                "Order confirmed via AI assistant",
            ),
        )
        order_id = cur.lastrowid

        # Clear cart
        conn.execute(
            "DELETE FROM cart WHERE user_id = ? AND session_id = ?",
            (user_id, session_id),
        )

        # Award loyalty points
        points_earned = int(total * 10)
        conn.execute(
            """UPDATE loyalty_points SET points = points + ?, updated_at = ?
               WHERE user_id = ?""",
            (points_earned, datetime.now(timezone.utc).isoformat(), user_id),
        )
        conn.commit()

        order_ref = _generate_ref("ORDER", order_id)

        return {
            "success": True,
            "order_id": order_id,
            "order_ref": order_ref,
            "items": json.loads(items_json),
            "total": round(total, 2),
            "points_earned": points_earned,
            "message": f"✅ Order confirmed! Your reference number is {order_ref}. Total: ${total:.2f}. You earned {points_earned} loyalty points.",
        }
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Booking / Scheduling Tools
# ─────────────────────────────────────────────

def book_appointment(state: dict, **kwargs) -> dict:
    """Book an appointment or reservation, parsing date/time from conversation."""
    user_id = state.get("user_id")
    query = state.get("query", "")
    history = state.get("history", [])
    conn = _db()
    try:
        services = conn.execute("SELECT * FROM services ORDER BY id").fetchall()
        providers = conn.execute(
            "SELECT * FROM service_providers WHERE available = 1 ORDER BY rating DESC"
        ).fetchall()

        if not services:
            return {"success": False, "message": "No services available for booking."}

        # Gather recent conversation context
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in (history or [])[-6:]
        )

        svc_list = "\n".join(f"ID:{s['id']}|{s['name']}|${s['price']}|{s['duration_min']}min" for s in services)
        prov_list = "\n".join(f"ID:{p['id']}|{p['name']}|{p['specialty']}" for p in providers) if providers else "No providers listed"

        parse_prompt = f"""Extract booking details from this conversation:

Conversation:
{history_text}
CUSTOMER: {query}

Available services:
{svc_list}

Available providers:
{prov_list}

Today's date is: {datetime.now().strftime('%Y-%m-%d %A')}

Extract ALL available info. Return ONLY JSON:
{{
  "service_id": 1,
  "service_name": "...",
  "provider_id": null,
  "provider_name": "first available",
  "date": "2025-03-01",
  "time": "14:00",
  "missing_fields": ["field1"],
  "parsed_ok": true
}}

If not enough info to book, set parsed_ok=false and list missing_fields.
Interpret relative dates (tomorrow, next Monday, etc.) based on today's date.
No markdown, just JSON."""

        raw = llm_call(parse_prompt, temperature=0.1, max_tokens=300)
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            booking = json.loads(raw)
        except json.JSONDecodeError:
            return {"success": False, "message": "I couldn't understand the booking details. Could you specify the service, date, and time?"}

        missing = booking.get("missing_fields", [])
        if not booking.get("parsed_ok", False) or missing:
            return {
                "success": False,
                "needs_info": True,
                "missing_fields": missing,
                "partial_booking": booking,
                "message": f"I need a bit more information to complete the booking. Please provide: {', '.join(missing)}.",
            }

        # Check for conflicts and provider availability
        date_str = booking.get("date", "")
        time_str = booking.get("time", "")
        if not date_str or not time_str:
             return {"success": False, "message": "I need a specific date and time to book."}
             
        try:
            req_dt = datetime.fromisoformat(f"{date_str}T{time_str}:00")
            day_key = req_dt.strftime("%A")[:3].lower()
        except ValueError:
             return {"success": False, "message": "The date or time format is invalid."}

        svc_id = booking.get("service_id")
        prov_id = booking.get("provider_id")
        
        # 1. Validate Provider Schedule
        assigned_provider = None
        if prov_id:
            p = conn.execute("SELECT * FROM service_providers WHERE id = ?", (prov_id,)).fetchone()
            if p:
                sched = json.loads(p["schedule"]) if p["schedule"] else {}
                if day_key not in sched:
                    return {
                        "success": False,
                        "message": f"I'm sorry, {p['name']} is not available on {req_dt.strftime('%A')}s. They work: {', '.join(sched.keys())}."
                    }
                assigned_provider = p
        else:
            # Find any provider available on this day
            for p in providers:
                sched = json.loads(p["schedule"]) if p["schedule"] else {}
                if day_key in sched:
                    assigned_provider = p
                    prov_id = p["id"]
                    break
            if not assigned_provider:
                return {"success": False, "message": f"We are closed or have no providers available on {req_dt.strftime('%A')}s."}

        # 2. Check existing bookings for conflicts
        conflicts = conn.execute(
            """SELECT COUNT(*) FROM orders
                WHERE provider_id = ? AND status IN ('pending','confirmed')
                AND scheduled_at = ?""",
            (prov_id, req_dt.isoformat()),
        ).fetchone()[0]
        if conflicts > 0:
            return {
                "success": False,
                "message": f"{assigned_provider['name']} already has a booking at {time_str} on {date_str}. Would you like a different time?",
            }

        svc_row = conn.execute("SELECT * FROM services WHERE id = ?", (svc_id,)).fetchone() if svc_id else None
        svc_name = svc_row["name"] if svc_row else booking.get("service_name", "Service")
        price = svc_row["price"] if svc_row else 0

        cur = conn.execute(
            """INSERT INTO orders (user_id, service_id, provider_id, status, order_type, scheduled_at, total_price, notes)
               VALUES (?, ?, ?, 'confirmed', 'appointment', ?, ?, ?)""",
            (user_id, svc_id, prov_id, scheduled_dt, price, "Booked via AI assistant"),
        )
        order_id = cur.lastrowid

        # Create calendar event
        duration = svc_row["duration_min"] if svc_row else 30
        try:
            start_dt = datetime.fromisoformat(scheduled_dt)
        except ValueError:
            start_dt = datetime.utcnow() + timedelta(days=1)
        end_dt = start_dt + timedelta(minutes=duration)

        prov_name = booking.get("provider_name", "")
        conn.execute(
            """INSERT INTO calendar_events (user_id, order_id, title, description, start_time, end_time, provider, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'confirmed')""",
            (
                user_id, order_id,
                f"Appointment: {svc_name}",
                f"Booking #{order_id} for {svc_name}",
                start_dt.isoformat(), end_dt.isoformat(),
                prov_name,
            ),
        )
        conn.commit()

        # Generate .ics file
        _generate_ics(order_id, svc_name, start_dt, end_dt, prov_name)

        booking_ref = _generate_ref("APPT", order_id)

        return {
            "success": True,
            "booking_id": order_id,
            "booking_ref": booking_ref,
            "service": svc_name,
            "provider": prov_name or "First available",
            "date": date_str,
            "time": time_str,
            "duration_min": duration,
            "price": price,
            "message": f"✅ Appointment confirmed: {svc_name} on {date_str} at {time_str} with {prov_name or 'first available'}. Reference: {booking_ref}. Duration: {duration} min. Price: ${price:.2f}.",
        }
    finally:
        conn.close()


def reschedule_booking(state: dict, **kwargs) -> dict:
    """Reschedule an existing booking, parsing the new date/time from conversation."""
    user_id = state.get("user_id")
    query = state.get("query", "")
    history = state.get("history", [])
    conn = _db()
    try:
        order = conn.execute(
            """SELECT o.*, s.name as service_name
               FROM orders o LEFT JOIN services s ON o.service_id = s.id
               WHERE o.user_id = ? AND o.status IN ('pending','confirmed')
               ORDER BY o.scheduled_at DESC LIMIT 1""",
            (user_id,),
        ).fetchone()
        if not order:
            return {"success": False, "message": "No active bookings found to reschedule."}

        history_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in (history or [])[-4:])

        parse_prompt = f"""Extract the new date and time from this reschedule request:

Conversation:
{history_text}
CUSTOMER: {query}

Current booking: {order['service_name']} on {order['scheduled_at']}
Today is: {datetime.now().strftime('%Y-%m-%d %A')}

Return ONLY JSON: {{"date": "2025-03-05", "time": "15:00", "parsed_ok": true}}
If not enough info: {{"parsed_ok": false, "message": "When would you like to reschedule to?"}}
Interpret relative dates. No markdown, just JSON."""

        raw = llm_call(parse_prompt, temperature=0.1, max_tokens=150)
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            return {"success": False, "message": "When would you like to reschedule to? Please provide a date and time."}

        if not result.get("parsed_ok", False):
            return {"success": False, "needs_info": True, "message": result.get("message", "When would you like to reschedule to?")}

        new_date = result.get("date", "")
        new_time = result.get("time", "")
        new_dt = f"{new_date}T{new_time}:00"

        conn.execute(
            "UPDATE orders SET scheduled_at = ?, notes = ? WHERE id = ?",
            (new_dt, "Rescheduled via AI assistant", order["id"]),
        )

        # Update calendar event
        try:
            start = datetime.fromisoformat(new_dt)
            end = start + timedelta(minutes=30)
            conn.execute(
                "UPDATE calendar_events SET start_time = ?, end_time = ?, status = 'rescheduled' WHERE order_id = ?",
                (start.isoformat(), end.isoformat(), order["id"]),
            )
        except ValueError:
            pass

        conn.commit()

        # Regenerate .ics
        try:
            start = datetime.fromisoformat(new_dt)
            end = start + timedelta(minutes=30)
            _generate_ics(order["id"], order["service_name"], start, end, "")
        except ValueError:
            pass

        booking_ref = _generate_ref("APPT", order["id"])

        return {
            "success": True,
            "booking_id": order["id"],
            "booking_ref": booking_ref,
            "service": order["service_name"],
            "new_date": new_date,
            "new_time": new_time,
            "message": f"✅ Appointment {booking_ref} for {order['service_name']} has been rescheduled to {new_date} at {new_time}.",
        }
    finally:
        conn.close()


def cancel_booking(state: dict, **kwargs) -> dict:
    """Cancel an existing booking by order ID or most recent pending booking."""
    user_id = state.get("user_id")
    conn = _db()
    try:
        order = conn.execute(
            """SELECT o.*, s.name as service_name
               FROM orders o LEFT JOIN services s ON o.service_id = s.id
               WHERE o.user_id = ? AND o.status IN ('pending', 'confirmed')
               ORDER BY o.scheduled_at DESC LIMIT 1""",
            (user_id,),
        ).fetchone()
        if not order:
            return {"success": False, "message": "No active bookings found to cancel."}
        conn.execute("UPDATE orders SET status = 'cancelled' WHERE id = ?", (order["id"],))
        conn.execute(
            "UPDATE calendar_events SET status = 'cancelled' WHERE order_id = ?",
            (order["id"],),
        )
        conn.commit()
        booking_ref = _generate_ref("APPT", order["id"])
        # Format date for readability
        try:
             dt = datetime.fromisoformat(order["scheduled_at"].replace("T", " "))
             dt_str = dt.strftime("%A, %B %d at %I:%M %p")
        except:
             dt_str = order["scheduled_at"]

        return {
            "success": True,
            "cancelled_booking_id": order["id"],
            "booking_ref": booking_ref,
            "service": order["service_name"],
            "scheduled_at": dt_str,
            "message": f"✅ Your appointment for {order['service_name']} on {dt_str} has been successfully cancelled. Reference: {booking_ref}.",
        }
    finally:
        conn.close()


def check_availability(state: dict, **kwargs) -> dict:
    """Check available time slots, considering existing bookings and provider schedules."""
    query = state.get("query", "")
    conn = _db()
    try:
        providers = conn.execute(
            "SELECT * FROM service_providers WHERE available = 1 ORDER BY rating DESC"
        ).fetchall()
        services = conn.execute("SELECT id, name, duration_min, price FROM services").fetchall()

        # Get existing bookings for next 7 days to find real availability
        now = datetime.now(timezone.utc)
        booked_slots = []
        for day_offset in range(1, 8):
            date = (now + timedelta(days=day_offset)).strftime("%Y-%m-%d")
            existing = conn.execute(
                "SELECT scheduled_at, provider_id FROM orders WHERE status IN ('pending','confirmed') AND scheduled_at LIKE ?",
                (f"{date}%",),
            ).fetchall()
            booked_slots.extend([(e["scheduled_at"], e["provider_id"]) for e in existing])

        # Generate available slots
        slots = []
        for day_offset in range(1, 8):
            dt = now + timedelta(days=day_offset)
            date_str = dt.strftime("%Y-%m-%d")
            day_name = dt.strftime("%A")
            day_key = day_name[:3].lower() # mon, tue, wed...
            
            # Find providers available on THIS day
            available_today = []
            for p in providers:
                sched = json.loads(p["schedule"]) if p["schedule"] else {}
                if day_key in sched:
                    available_today.append(p)
            
            if not available_today:
                continue

            # Check specific hours (simplified: 9-17)
            for hour in range(9, 17):
                slot_str = f"{date_str} {hour:02d}:00"
                # Check if slot is not fully booked across all available providers
                bookings_at_slot = sum(1 for s, pid in booked_slots if s and slot_str.replace(" ", "T") in s and any(ap["id"] == pid for ap in available_today))
                if bookings_at_slot < len(available_today):
                    slots.append({
                        "datetime": slot_str, 
                        "day": day_name, 
                        "available_providers_count": len(available_today) - bookings_at_slot,
                        "providers": [p["name"] for p in available_today]
                    })

        return {
            "success": True,
            "available_providers": [{"name": p["name"], "specialty": p["specialty"], "rating": p["rating"]} for p in providers],
            "available_slots": slots[:20],
            "services": [dict(s) for s in services],
            "message": f"{len(providers)} providers available. {len(slots)} open time slots in the next 7 days.",
        }
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Address Validation Tool
# ─────────────────────────────────────────────

def validate_address(state: dict, **kwargs) -> dict:
    """Extract, validate, and normalise delivery address from conversation."""
    query = state.get("query", "")
    history = state.get("history", [])

    history_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in (history or [])[-6:])

    validate_prompt = f"""Extract and validate the delivery address from this conversation:

Conversation:
{history_text}
CUSTOMER: {query}

Extract the following. Return ONLY JSON:
{{
  "found": true,
  "street": "34 Front Street",
  "unit": "",
  "city": "Toronto",
  "province": "ON",
  "postal_code": "M4K 6B2",
  "country": "Canada",
  "formatted": "34 Front Street, Toronto, ON M4K 6B2, Canada",
  "postal_code_valid": true,
  "issues": []
}}

Rules for Canadian postal codes: format A1A 1A1 (letter-digit-letter space digit-letter-digit).
CROSS-REFERENCE CITY/PROVINCE: Ensure the first letter of the postal code matches the province (e.g. M=Toronto, K/N=Ontario, T=Alberta).
If the address is valid but the postal code is slightly formatted differently (e.g. "M4K6B2" vs "M4K 6B2"), normalize it to "A1A 1A1" and set valid=true.
If no address found: {{"found": false, "message": "Please provide your delivery address."}}
No markdown, just JSON."""

    raw = llm_call(validate_prompt, temperature=0.1, max_tokens=300)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return {"success": False, "message": "Could you please provide your full delivery address including postal code?"}

    if not result.get("found", False):
        return {"success": False, "needs_info": True, "message": result.get("message", "Please provide your delivery address.")}

    address_str = result.get("formatted", "")
    postal_code = result.get("postal_code", "").upper().replace(" ", "")
    issues = result.get("issues", [])

    # Robust Regex Validation for Canadian Postal Codes
    is_canadian = result.get("country", "").lower() == "canada" or result.get("province", "") in ["ON", "QC", "BC", "AB", "MB", "SK", "NS", "NB", "PE", "NL", "YT", "NT", "NU"]
    if is_canadian and postal_code:
        if not re.match(r"^[A-Z]\d[A-Z]\d[A-Z]\d$", postal_code):
            issues.append("Invalid Canadian postal code format. Should be A1A 1A1.")
        else:
            # Re-format with space
            result["postal_code"] = f"{postal_code[:3]} {postal_code[3:]}"

    # Real existence check with Geopy
    if GEOPY_AVAILABLE and address_str and not issues:
        try:
            geolocator = Nominatim(user_agent="generic_ai_agent_business")
            location = geolocator.geocode(address_str, timeout=5)
            if not location:
                # Try without unit if present
                simplified = f"{result.get('street', '')}, {result.get('city', '')}, {result.get('province', '')}, Canada"
                location = geolocator.geocode(simplified, timeout=5)
            
            if not location:
                # If it's a very specific address, Nominatim sometimes fails. 
                # If the postal code is valid and city matches, we can be more lenient.
                pass 
            else:
                # Basic postal code confront (first 3 chars)
                pc_prefix = result.get("postal_code", "")[:3].upper()
                if pc_prefix and pc_prefix not in location.address.upper():
                    # Some Nominatim responses are weird, double check
                    if result.get("postal_code", "").replace(" ", "").upper() not in location.address.replace(" ", "").upper():
                         # Only flag if it's a major mismatch
                         if result.get("city", "").lower() not in location.address.lower():
                            issues.append(f"Postal code {result['postal_code']} might not match this location ({location.address[:50]}...)")
        except (GeocoderTimedOut, GeocoderServiceError):
            pass # Fallback to LLM validation if service is down

    if issues:
        return {
            "success": False,
            "address": result,
            "issues": issues,
            "message": f"I had a bit of trouble verifying that address: {'; '.join(issues)}. Could you please double-check the spelling or postal code?",
        }

    # Store address on the user profile
    user_id = state.get("user_id")
    if user_id:
        conn = _db()
        try:
            conn.execute(
                "UPDATE users SET address = ?, postal_code = ?, city = ? WHERE id = ?",
                (result.get("formatted", ""), result.get("postal_code", ""), result.get("city", ""), user_id),
            )
            conn.commit()
        finally:
            conn.close()

    return {
        "success": True,
        "address": result,
        "formatted_address": result.get("formatted", ""),
        "postal_code_valid": result.get("postal_code_valid", True),
        "message": f"Address confirmed: {result.get('formatted', '')}",
    }


# ─────────────────────────────────────────────
# Delivery Type Tool
# ─────────────────────────────────────────────

def set_delivery_type(state: dict, **kwargs) -> dict:
    """Set whether the order is for pickup or delivery."""
    query = state.get("query", "").lower()

    if any(w in query for w in ["deliver", "delivery", "bring", "send"]):
        return {
            "success": True,
            "delivery_type": "delivery",
            "needs_address": True,
            "message": "Got it — delivery! Please provide your delivery address.",
        }
    elif any(w in query for w in ["pickup", "pick up", "pick-up", "collect", "come get"]):
        return {
            "success": True,
            "delivery_type": "pickup",
            "needs_address": False,
            "message": "Got it — pickup! Your order will be ready for collection.",
        }
    else:
        return {
            "success": False,
            "message": "Would you like pickup or delivery?",
        }


# ─────────────────────────────────────────────
# Information & History Tools
# ─────────────────────────────────────────────

def get_pricing(state: dict, **kwargs) -> dict:
    """Get pricing information for all available services."""
    conn = _db()
    try:
        services = conn.execute(
            "SELECT name, description, price, duration_min, category, modifiers FROM services ORDER BY category, price"
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


def get_recommendations(state: dict, **kwargs) -> dict:
    """Provide personalised service recommendations based on customer history and popular items."""
    user_id = state.get("user_id")
    conn = _db()
    try:
        # What they've ordered before
        past_services = conn.execute(
            """SELECT s.name, COUNT(o.id) as cnt FROM orders o
               JOIN services s ON o.service_id = s.id
               WHERE o.user_id = ? GROUP BY s.name ORDER BY cnt DESC LIMIT 5""",
            (user_id,),
        ).fetchall()

        # Popular items globally
        popular = conn.execute(
            """SELECT s.name, COUNT(o.id) as cnt FROM orders o
               JOIN services s ON o.service_id = s.id
               GROUP BY s.name ORDER BY cnt DESC LIMIT 5"""
        ).fetchall()

        # New services they haven't tried
        used_ids = [r[0] for r in conn.execute(
            "SELECT DISTINCT service_id FROM orders WHERE user_id = ?", (user_id,)
        ).fetchall()]
        if used_ids:
            placeholders = ",".join("?" * len(used_ids))
            new_services = conn.execute(
                f"SELECT * FROM services WHERE id NOT IN ({placeholders}) ORDER BY RANDOM() LIMIT 3",
                used_ids,
            ).fetchall()
        else:
            new_services = conn.execute("SELECT * FROM services ORDER BY RANDOM() LIMIT 3").fetchall()

        loyalty = conn.execute("SELECT points, tier FROM loyalty_points WHERE user_id = ?", (user_id,)).fetchone()

        # Check family members for cross-sell
        user = conn.execute("SELECT family_members FROM users WHERE id = ?", (user_id,)).fetchone()
        family_info = []
        if user and user["family_members"]:
            try:
                family_info = json.loads(user["family_members"])
            except json.JSONDecodeError:
                pass

        return {
            "success": True,
            "past_favorites": [{"name": r["name"], "order_count": r["cnt"]} for r in past_services],
            "popular_items": [{"name": r["name"], "order_count": r["cnt"]} for r in popular],
            "new_to_try": [dict(s) for s in new_services],
            "loyalty_points": loyalty["points"] if loyalty else 0,
            "loyalty_tier": loyalty["tier"] if loyalty else "bronze",
            "family_members": family_info,
            "message": f"Based on your history, here are some personalised recommendations.",
        }
    finally:
        conn.close()


def get_order_history(state: dict, **kwargs) -> dict:
    """Retrieve the customer's past orders and bookings."""
    user_id = state.get("user_id")
    conn = _db()
    try:
        orders = conn.execute(
            """SELECT o.id, o.status, o.order_type, o.scheduled_at, o.total_price,
                      o.delivery_type, o.items_json,
                      s.name as service_name, sp.name as provider_name
               FROM orders o
               LEFT JOIN services s ON o.service_id = s.id
               LEFT JOIN service_providers sp ON o.provider_id = sp.id
               WHERE o.user_id = ?
               ORDER BY o.scheduled_at DESC LIMIT 10""",
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
    return {"success": True, "hours": hours, "message": f"Business hours:\n{hours}"}


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


def get_faqs(state: dict, **kwargs) -> dict:
    """Retrieve frequently asked questions from the knowledge base."""
    conn = _db()
    try:
        faqs = conn.execute(
            """SELECT topic, content FROM enriched_knowledge
               WHERE LOWER(topic) LIKE '%faq%' OR LOWER(topic) LIKE '%question%' OR LOWER(topic) LIKE '%common%'
               LIMIT 3"""
        ).fetchall()
        if not faqs:
            faqs = conn.execute("SELECT topic, content FROM enriched_knowledge LIMIT 3").fetchall()
        return {"success": True, "faqs": [dict(f) for f in faqs], "message": f"Retrieved {len(faqs)} FAQ entries."}
    finally:
        conn.close()


def get_promotions(state: dict, **kwargs) -> dict:
    """List current deals, promotions, and special offers."""
    conn = _db()
    try:
        promos = conn.execute(
            """SELECT topic, content FROM enriched_knowledge
               WHERE LOWER(topic) LIKE '%promo%' OR LOWER(topic) LIKE '%deal%'
                  OR LOWER(topic) LIKE '%offer%' OR LOWER(topic) LIKE '%discount%'
                  OR LOWER(topic) LIKE '%seasonal%'
               LIMIT 3"""
        ).fetchall()
        if not promos:
            promos = conn.execute("SELECT topic, content FROM enriched_knowledge LIMIT 2").fetchall()
        return {"success": True, "promotions": [dict(p) for p in promos], "message": f"Found {len(promos)} active promotions."}
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Loyalty Tools
# ─────────────────────────────────────────────

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
            (new_points, datetime.now(timezone.utc).isoformat(), user_id),
        )
        conn.commit()
        return {
            "success": True, "discount_applied": discount, "points_used": points_used,
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
            "success": True, "points": points, "tier": tier, "next_tier": next_tier,
            "points_to_next_tier": max(0, next_threshold - points),
            "dollar_value": (points // 100) * 5,
            "message": f"You have {points} loyalty points ({tier} tier). ${(points // 100) * 5:.2f} redeemable.",
        }
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Dispute / Complaint Tool
# ─────────────────────────────────────────────

def handle_dispute(state: dict, **kwargs) -> dict:
    """Handle order disputes, complaints, and discrepancies (e.g. wrong quantity delivered)."""
    user_id = state.get("user_id")
    query = state.get("query", "")
    conn = _db()
    try:
        recent_orders = conn.execute(
            """SELECT o.id, o.items_json, o.total_price, o.status, s.name as service_name
               FROM orders o LEFT JOIN services s ON o.service_id = s.id
               WHERE o.user_id = ? ORDER BY o.scheduled_at DESC LIMIT 5""",
            (user_id,),
        ).fetchall()

        orders_text = "\n".join(
            f"Order #{o['id']}: {o['service_name']} | ${o['total_price']} | Status: {o['status']} | Items: {o['items_json'] or 'N/A'}"
            for o in recent_orders
        )

        parse_prompt = f"""Customer complaint: "{query}"
Recent orders:
{orders_text}

Analyse the complaint and return JSON:
{{
  "order_id": 1,
  "complaint_type": "wrong_quantity|missing_item|wrong_item|quality|other",
  "description": "Customer ordered 3 but received 2",
  "suggested_resolution": "Refund for 1 item or redeliver",
  "understood": true
}}
No markdown, just JSON."""

        raw = llm_call(parse_prompt, temperature=0.1, max_tokens=250)
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            dispute = json.loads(raw)
        except json.JSONDecodeError:
            return {"success": False, "message": "I'm sorry, I couldn't process the details of your complaint. Could you please describe it again?"}

        # Log the complaint to the DB
        cur = conn.execute(
            """INSERT INTO complaints (user_id, order_id, conversation_id, complaint_type, description, suggested_resolution)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                dispute.get("order_id"),
                state.get("conversation_id"),
                dispute.get("complaint_type", "other"),
                dispute.get("description", query),
                dispute.get("suggested_resolution", "Review needed")
            )
        )
        complaint_id = cur.lastrowid
        conn.commit()
        
        ref_id = _generate_ref("REF", complaint_id)

        return {
            "success": True,
            "complaint_id": complaint_id,
            "complaint_ref": ref_id,
            "details": dispute,
            "message": f"I've logged your concern regarding Order #{dispute.get('order_id') or 'unknown'}. Your reference number is {ref_id}. Our team will review this and I'll make sure we address it right away.",
        }
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Profile & Other Tools
# ─────────────────────────────────────────────

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
        return {"success": True, "results": formatted, "query": search_query, "message": f"Found {len(formatted)} web results."}
    except Exception as e:
        return {"success": False, "message": f"Web search failed: {str(e)}", "results": []}


def escalate_to_human(state: dict, **kwargs) -> dict:
    """Flag the conversation for human agent handoff."""
    conv_id = state.get("conversation_id", "unknown")
    log_agent_event(conv_id, "human_escalation", {
        "user_id": state.get("user_id"), "query": state.get("query", ""),
        "reason": "Customer requested human agent or issue unresolved",
    })
    return {
        "success": True, "escalated": True,
        "message": "Your request has been flagged for a human agent. We will contact you within 24 hours via email.",
    }


def get_global_stats_tool(state: dict, **kwargs) -> dict:
    """Retrieve aggregated business statistics."""
    stats = get_global_stats()
    return {
        "success": True, "stats": stats,
        "message": f"Business has {stats.get('total_customers',0)} customers and ${stats.get('total_revenue',0):.2f} total revenue.",
    }


def update_customer_profile(state: dict, **kwargs) -> dict:
    """Update the customer's profile information (name, phone, address)."""
    user_id = state.get("user_id")
    query = state.get("query", "")
    conn = _db()
    try:
        user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if not user:
            return {"success": False, "message": "User not found."}

        # Use LLM to extract what they want to update
        parse_prompt = f"""Customer says: "{query}"
Current profile: name={user['full_name']}, phone={user['phone']}, email={user['email']}, address={user['address'] or 'not set'}

What field do they want to update and to what value?
Return JSON: {{"field": "phone", "new_value": "555-1234", "understood": true}}
If unclear: {{"understood": false, "message": "Which field?"}}
No markdown, just JSON."""

        raw = llm_call(parse_prompt, temperature=0.1, max_tokens=150)
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            update = json.loads(raw)
        except json.JSONDecodeError:
            update = {"understood": False}

        if update.get("understood", False):
            field = update.get("field", "")
            valid_fields = {"full_name": "full_name", "name": "full_name", "phone": "phone", "email": "email", "address": "address"}
            db_field = valid_fields.get(field.lower())
            if db_field:
                conn.execute(f"UPDATE users SET {db_field} = ? WHERE id = ?", (update["new_value"], user_id))
                conn.commit()
                return {
                    "success": True,
                    "updated_field": db_field,
                    "new_value": update["new_value"],
                    "message": f"Updated your {field} to: {update['new_value']}",
                }

        return {
            "success": True, "user_id": user_id,
            "current_profile": {"full_name": user["full_name"], "email": user["email"], "phone": user["phone"], "address": user["address"]},
            "message": "Your profile is on file. Please tell me which field (name, phone, email, address) you'd like to update and the new value.",
        }
    finally:
        conn.close()


def check_family_members(state: dict, **kwargs) -> dict:
    """Check customer's family members for cross-selling opportunities."""
    user_id = state.get("user_id")
    conn = _db()
    try:
        user = conn.execute("SELECT full_name, family_members FROM users WHERE id = ?", (user_id,)).fetchone()
        if not user or not user["family_members"]:
            return {"success": True, "family": [], "message": "No family members on file."}

        try:
            family = json.loads(user["family_members"])
        except json.JSONDecodeError:
            family = []

        # Check last visits for family members
        family_info = []
        for member in family:
            family_info.append({
                "name": member.get("name", ""),
                "relation": member.get("relation", ""),
            })

        return {
            "success": True,
            "customer_name": user["full_name"],
            "family": family_info,
            "message": f"Found {len(family_info)} family member(s) on file: {', '.join(m['name'] for m in family_info)}.",
        }
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Calendar / ICS Helper
# ─────────────────────────────────────────────

def _generate_ics(order_id: int, title: str, start: datetime, end: datetime, provider: str) -> str:
    """Generate a local .ics calendar file for an appointment."""
    calendar_dir = os.path.join(os.getcwd(), "calendar")
    os.makedirs(calendar_dir, exist_ok=True)
    filepath = os.path.join(calendar_dir, f"appointment_{order_id}.ics")

    ics_content = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//GenericAgent//EN
BEGIN:VEVENT
DTSTART:{start.strftime('%Y%m%dT%H%M%S')}
DTEND:{end.strftime('%Y%m%dT%H%M%S')}
SUMMARY:{title}
DESCRIPTION:Booking #{order_id} with {provider or 'provider'}
STATUS:CONFIRMED
UID:booking-{order_id}@genericagent
END:VEVENT
END:VCALENDAR"""

    with open(filepath, "w") as f:
        f.write(ics_content)
    return filepath


def get_calendar(state: dict, **kwargs) -> dict:
    """Get the customer's upcoming calendar events / appointments."""
    user_id = state.get("user_id")
    conn = _db()
    try:
        events = conn.execute(
            """SELECT * FROM calendar_events
               WHERE user_id = ? AND status != 'cancelled'
               ORDER BY start_time ASC LIMIT 10""",
            (user_id,),
        ).fetchall()
        return {
            "success": True,
            "events": [dict(e) for e in events],
            "total": len(events),
            "message": f"You have {len(events)} upcoming appointment(s).",
        }
    finally:
        conn.close()


# ─────────────────────────────────────────────
# TOOLS REGISTRY
# ─────────────────────────────────────────────

TOOLS: dict[str, tuple] = {
    # Cart / Order management
    "add_to_cart": (add_to_cart, "Activates when the customer wants to order, add, or get an item/product/service. Also when they say 'I want X' or 'give me X'."),
    "remove_from_cart": (remove_from_cart, "Activates when the customer wants to remove, delete, or change their mind about an item in their order/cart. Also 'I don't want X anymore'."),
    "view_cart": (view_cart, "Activates when the customer asks to see their cart, current order, what they've ordered so far."),
    "confirm_order": (confirm_order, "Activates when the customer says 'confirm', 'yes', 'place order', 'that's all', 'finalize', or agrees to a final order summary."),

    # Booking / Scheduling
    "book_appointment": (book_appointment, "Activates when the customer wants to book, schedule, reserve, or make an appointment. Also 'I need to see a doctor/dentist/stylist'."),
    "reschedule_booking": (reschedule_booking, "Activates when the customer wants to change, move, or reschedule the date or time of an existing booking."),
    "cancel_booking": (cancel_booking, "Activates when the customer wants to cancel, delete, or remove an existing booking or reservation."),
    "check_availability": (check_availability, "Activates when the customer asks about availability, open slots, time slots, when they can come, or 'when is X available'."),

    # Delivery / Address
    "set_delivery_type": (set_delivery_type, "Activates when the customer mentions pickup, delivery, or is asked about how they want to receive their order."),
    "validate_address": (validate_address, "Activates when the customer provides a delivery address, street, postal code, or when address validation is needed."),

    # Information
    "get_pricing": (get_pricing, "Activates when the customer asks about prices, costs, fees, rates, or how much something costs."),
    "get_recommendations": (get_recommendations, "Activates when the customer asks for recommendations, suggestions, what to try, or what is popular. Also for upselling."),
    "get_order_history": (get_order_history, "Activates when the customer asks about their past orders, previous bookings, or purchase history."),
    "get_business_hours": (get_business_hours, "Activates when the customer asks about opening hours, closing time, when the business is open, holiday hours."),
    "get_provider_info": (get_provider_info, "Activates when the customer asks about staff, therapists, doctors, providers, or who will serve them."),
    "get_faqs": (get_faqs, "Activates when the customer asks a general question, how something works, policies, or requests FAQs."),
    "get_promotions": (get_promotions, "Activates when the customer asks about deals, promotions, discounts, offers, or special prices."),

    # Loyalty
    "apply_loyalty_discount": (apply_loyalty_discount, "Activates when the customer wants to redeem, use, or apply loyalty points or get a discount via points."),
    "get_loyalty_balance": (get_loyalty_balance, "Activates when the customer asks about their loyalty points, rewards balance, tier, or membership status."),

    # Dispute
    "handle_dispute": (handle_dispute, "Activates when the customer has a complaint, dispute, wrong order, missing item, quality issue, or says something was wrong."),

    # Profile & Family
    "update_customer_profile": (update_customer_profile, "Activates when the customer wants to update their name, phone, email, address, or personal details."),
    "check_family_members": (check_family_members, "Activates when recommending services for family, asking about family members, or cross-selling for spouse/children."),

    # Calendar
    "get_calendar": (get_calendar, "Activates when the customer asks about their upcoming appointments, schedule, or calendar."),

    # Web search
    "search_web": (search_web, "Activates when the customer asks for current news, trends, external information, or anything not in the knowledge base."),

    # Escalation
    "escalate_to_human": (escalate_to_human, "Activates when the customer is frustrated, requests a human agent, or when the AI cannot resolve the issue."),

    # Stats
    "get_global_stats": (get_global_stats_tool, "Activates when asked about overall business performance, statistics, or revenue (usually staff/admin queries)."),
}
