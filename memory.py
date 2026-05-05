# memory.py
# PostgreSQL via Supabase

import os
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Fix for SQLAlchemy compatibility
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)


def init_db():
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY,
                user_id TEXT,
                role TEXT,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS credentials (
                id SERIAL PRIMARY KEY,
                key TEXT UNIQUE,
                value TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS email_log (
                id SERIAL PRIMARY KEY,
                supplier_name TEXT,
                supplier_email TEXT,
                item_name TEXT,
                status TEXT,
                sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS invoices (
                id SERIAL PRIMARY KEY,
                filename TEXT,
                vendor_name TEXT,
                amount REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.commit()
    print("✅ PostgreSQL connected and tables ready!")


def get_db():
    return engine.connect()


# ── Chat history ──────────────────────────────

def save_message(user_id: str, role: str, content: str):
    with engine.connect() as conn:
        conn.execute(text(
            "INSERT INTO chat_history (user_id, role, content) VALUES (:u, :r, :c)"
        ), {"u": user_id, "r": role, "c": content})
        conn.commit()


def get_history(user_id: str, limit: int = 10) -> list:
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT role, content FROM chat_history WHERE user_id=:u ORDER BY id DESC LIMIT :l"
        ), {"u": user_id, "l": limit}).fetchall()
    rows = list(reversed(rows))
    return [{"role": r[0], "content": r[1]} for r in rows]


# ── Credentials ───────────────────────────────

def save_credential(key: str, value: str):
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO credentials (key, value) VALUES (:k, :v)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
        """), {"k": key, "v": value})
        conn.commit()


def get_credential(key: str):
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT value FROM credentials WHERE key=:k"
        ), {"k": key}).fetchone()
    return row[0] if row else None


# ── Invoices ──────────────────────────────────

def save_invoice(filename: str, vendor_name: str, amount: float):
    with engine.connect() as conn:
        conn.execute(text(
            "INSERT INTO invoices (filename, vendor_name, amount) VALUES (:f, :v, :a)"
        ), {"f": filename, "v": vendor_name, "a": amount})
        conn.commit()


def get_invoices() -> list:
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT filename, vendor_name, amount, created_at FROM invoices ORDER BY created_at DESC"
        )).fetchall()
    return [{"filename": r[0], "vendor_name": r[1], "amount": r[2], "created_at": str(r[3])} for r in rows]


def get_spend_summary() -> dict:
    with engine.connect() as conn:
        total = conn.execute(text("SELECT SUM(amount) FROM invoices")).fetchone()[0] or 0
        count = conn.execute(text("SELECT COUNT(*) FROM invoices")).fetchone()[0]
        row = conn.execute(text(
            "SELECT vendor_name, SUM(amount) as t FROM invoices GROUP BY vendor_name ORDER BY t DESC LIMIT 1"
        )).fetchone()
    return {
        "total_spend": total,
        "total_orders": count,
        "top_supplier": row[0] if row else "—"
    }


# Run at import
init_db()