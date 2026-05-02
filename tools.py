# tools.py
# Structured data tools — reads Excel + SQLite

import pandas as pd
import sqlite3
from config import EXCEL_FILE, DB_PATH


def read_sheet(sheet: str) -> pd.DataFrame:
    return pd.read_excel(EXCEL_FILE, sheet_name=sheet).fillna("")


def safe_int(x) -> int:
    try:
        return int(float(x))
    except Exception:
        return 0


# ── Suppliers ─────────────────────────────────

def get_suppliers() -> str:
    try:
        df = read_sheet("Suppliers").head(8)
        return "\n".join(
            f"• {r['Supplier_Name']} — {r['City']} — {r['Category']}"
            for _, r in df.iterrows()
        )
    except Exception as e:
        return f"Supplier error: {e}"


def top_suppliers() -> str:
    try:
        top = read_sheet("Procurement_History")["Supplier"].value_counts().head(5)
        return "\n".join(f"• {n} — {c} orders" for n, c in top.items())
    except Exception as e:
        return f"Top supplier error: {e}"


def suppliers_by_city(city: str) -> str:
    try:
        df = read_sheet("Suppliers")
        data = df[df["City"].astype(str).str.lower().str.contains(city.lower())].head(8)
        if data.empty:
            return "No suppliers found."
        return "\n".join(f"• {r['Supplier_Name']} — {r['Category']}" for _, r in data.iterrows())
    except Exception as e:
        return f"City supplier error: {e}"


# ── Procurement ───────────────────────────────

def get_procurement_history() -> str:
    try:
        df = read_sheet("Procurement_History").head(6)
        return "\n".join(
            f"• {r['Supplier']} — {r['Category']} — ₹{r['Total_Price_INR']}"
            for _, r in df.iterrows()
        )
    except Exception as e:
        return f"History error: {e}"


def spend_summary() -> str:
    try:
        df = read_sheet("Procurement_History")
        total = safe_int(df["Total_Price_INR"].sum())
        by_cat = (
            df.groupby("Category")["Total_Price_INR"]
            .sum().sort_values(ascending=False).head(5)
        )
        lines = [f"Total Spend: ₹{total}"]
        lines += [f"• {cat} — ₹{safe_int(val)}" for cat, val in by_cat.items()]
        return "\n".join(lines)
    except Exception as e:
        return f"Spend error: {e}"


def highest_purchase() -> str:
    try:
        df = read_sheet("Procurement_History")
        r = df.loc[df["Total_Price_INR"].idxmax()]
        return f"• Supplier: {r['Supplier']}\n• Category: {r['Category']}\n• Amount: ₹{r['Total_Price_INR']}"
    except Exception as e:
        return f"Highest purchase error: {e}"


# ── Alerts ────────────────────────────────────

def check_alerts() -> str:
    try:
        df = read_sheet("Pending_Requirements")
        high = df[df["Priority"].astype(str).str.lower().eq("high")].head(6)
        if high.empty:
            return "No urgent alerts."
        return "\n".join(f"• {r['Item_Name']} ({r['Status']})" for _, r in high.iterrows())
    except Exception as e:
        return f"Alert error: {e}"


def pending_summary() -> str:
    try:
        df = read_sheet("Pending_Requirements")
        open_items = len(df[df["Status"].astype(str).str.lower().eq("open")])
        return f"• Total Pending: {len(df)}\n• Open Items: {open_items}"
    except Exception as e:
        return f"Pending error: {e}"


# ── Invoices (SQLite) ─────────────────────────

def get_invoice_summary() -> str:
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM invoices")
        total = cur.fetchone()[0]
        cur.execute("SELECT vendor_name, amount FROM invoices ORDER BY id DESC LIMIT 5")
        rows = cur.fetchall()
        conn.close()
        lines = [f"Total Invoices: {total}"]
        lines += [f"• {r[0]} — ₹{r[1]}" for r in rows]
        return "\n".join(lines)
    except Exception as e:
        return f"Invoice error: {e}"


def highest_invoice() -> str:
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "SELECT vendor_name, amount FROM invoices ORDER BY CAST(amount AS REAL) DESC LIMIT 1"
        )
        row = cur.fetchone()
        conn.close()
        return f"• {row[0]} — ₹{row[1]}" if row else "No invoices found."
    except Exception as e:
        return f"Highest invoice error: {e}"


# ── Registry ──────────────────────────────────

TOOLS = {
    "get_suppliers":          get_suppliers,
    "top_suppliers":          top_suppliers,
    "suppliers_by_city":      suppliers_by_city,
    "get_procurement_history":get_procurement_history,
    "spend_summary":          spend_summary,
    "highest_purchase":       highest_purchase,
    "check_alerts":           check_alerts,
    "pending_summary":        pending_summary,
    "get_invoice_summary":    get_invoice_summary,
    "highest_invoice":        highest_invoice,
}