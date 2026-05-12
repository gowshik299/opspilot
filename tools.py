# tools.py
import os
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

def safe_int(val) -> int:
    try:
        return int(val)
    except:
        return 0

def query_db(sql: str, params: dict = {}) -> list:
    """Execute SQL and return list of dicts"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql), params)
            columns = result.keys()
            return [dict(zip(columns, row)) for row in result]
    except Exception as e:
        return []

# ── Suppliers ─────────────────────────────────

def get_suppliers() -> str:
    try:
        rows = query_db("SELECT supplier_name, city, category FROM suppliers LIMIT 8")
        if not rows:
            return "No suppliers found."
        return "\n".join(
            f"• {r['supplier_name']} — {r['city']} — {r['category']}"
            for r in rows
        )
    except Exception as e:
        return f"Supplier error: {e}"

def top_suppliers() -> str:
    try:
        rows = query_db("""
            SELECT supplier, COUNT(*) as order_count 
            FROM procurement_history 
            GROUP BY supplier 
            ORDER BY order_count DESC 
            LIMIT 5
        """)
        if not rows:
            return "No procurement history found."
        return "\n".join(f"• {r['supplier']} — {r['order_count']} orders" for r in rows)
    except Exception as e:
        return f"Top supplier error: {e}"

def suppliers_by_city(city: str) -> str:
    try:
        rows = query_db(
            "SELECT supplier_name, category FROM suppliers WHERE LOWER(city) LIKE :city LIMIT 8",
            {"city": f"%{city.lower()}%"}
        )
        if not rows:
            return "No suppliers found."
        return "\n".join(f"• {r['supplier_name']} — {r['category']}" for r in rows)
    except Exception as e:
        return f"City supplier error: {e}"

# ── Procurement ───────────────────────────────

def get_procurement_history() -> str:
    try:
        rows = query_db("""
            SELECT supplier, category, total_price_inr 
            FROM procurement_history 
            ORDER BY id DESC LIMIT 6
        """)
        if not rows:
            return "No procurement history found."
        return "\n".join(
            f"• {r['supplier']} — {r['category']} — ₹{r['total_price_inr']}"
            for r in rows
        )
    except Exception as e:
        return f"History error: {e}"

def spend_summary() -> str:
    try:
        total_rows = query_db("SELECT SUM(total_price_inr) as total FROM procurement_history")
        total = safe_int(total_rows[0]['total']) if total_rows else 0

        cat_rows = query_db("""
            SELECT category, SUM(total_price_inr) as total
            FROM procurement_history
            GROUP BY category
            ORDER BY total DESC
            LIMIT 5
        """)
        lines = [f"Total Spend: ₹{total}"]
        lines += [f"• {r['category']} — ₹{safe_int(r['total'])}" for r in cat_rows]
        return "\n".join(lines)
    except Exception as e:
        return f"Spend error: {e}"

def highest_purchase() -> str:
    try:
        rows = query_db("""
            SELECT supplier, category, total_price_inr
            FROM procurement_history
            ORDER BY total_price_inr DESC
            LIMIT 1
        """)
        if not rows:
            return "No data found."
        r = rows[0]
        return f"• Supplier: {r['supplier']}\n• Category: {r['category']}\n• Amount: ₹{r['total_price_inr']}"
    except Exception as e:
        return f"Highest purchase error: {e}"

# ── Alerts ────────────────────────────────────

def check_alerts() -> str:
    try:
        rows = query_db("""
            SELECT item_name, status
            FROM pending_requirements
            WHERE LOWER(priority) = 'high'
            LIMIT 6
        """)
        if not rows:
            return "No urgent alerts."
        return "\n".join(f"• {r['item_name']} ({r['status']})" for r in rows)
    except Exception as e:
        return f"Alert error: {e}"

def pending_summary() -> str:
    try:
        rows = query_db("""
            SELECT req_id, item_name, category, priority, status
            FROM pending_requirements
            ORDER BY
                CASE priority
                    WHEN 'High' THEN 1
                    WHEN 'Medium' THEN 2
                    WHEN 'Low' THEN 3
                END
        """)
        if not rows:
            return "No pending requirements found."
        lines = [f"Total Pending: {len(rows)}"]
        for r in rows:
            lines.append(f"• {r['req_id']}: {r['item_name']} ({r['priority']} Priority) — {r['status']}")
        return "\n".join(lines)
    except Exception as e:
        return f"Pending error: {e}"

# ── Invoices ──────────────────────────────────

def get_invoice_summary() -> str:
    try:
        from memory import get_invoices
        rows = get_invoices()
        if not rows:
            return "No invoices found."
        lines = [f"Total Invoices: {len(rows)}"]
        lines += [f"• {r['vendor_name']} — ₹{r['amount']}" for r in rows[:5]]
        return "\n".join(lines)
    except Exception as e:
        return f"Invoice error: {e}"

def highest_invoice() -> str:
    try:
        from memory import get_invoices
        rows = get_invoices()
        if not rows:
            return "No invoices found."
        highest = max(rows, key=lambda x: x.get('amount', 0))
        return f"• Vendor: {highest['vendor_name']}\n• Amount: ₹{highest['amount']}"
    except Exception as e:
        return f"Invoice error: {e}"