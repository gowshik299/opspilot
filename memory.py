# memory.py
# Single SQLite layer for all persistence

import sqlite3

DB = "opspilot.db"


def connect():
    return sqlite3.connect(DB)


def get_db():
    """Alias used by gmail.py"""
    return connect()


def init_db():
    conn = connect()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    TEXT,
            role       TEXT,
            content    TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS invoices (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            filename    TEXT,
            vendor_name TEXT,
            amount      REAL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS credentials (
            id    INTEGER PRIMARY KEY AUTOINCREMENT,
            key   TEXT UNIQUE,
            value TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS email_log (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            supplier_name  TEXT,
            supplier_email TEXT,
            item_name      TEXT,
            status         TEXT,
            sent_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


# ── Chat history ──────────────────────────────

def save_message(user_id: str, role: str, content: str):
    conn = connect()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO chat_history (user_id, role, content) VALUES (?, ?, ?)",
        (user_id, role, content)
    )
    conn.commit()
    conn.close()


def get_history(user_id: str, limit: int = 10) -> list:
    conn = connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT role, content FROM chat_history WHERE user_id=? ORDER BY id DESC LIMIT ?",
        (user_id, limit),
    )
    rows = cur.fetchall()
    conn.close()
    rows.reverse()
    return [{"role": r[0], "content": r[1]} for r in rows]


# ── Credentials ───────────────────────────────

def save_credential(key: str, value: str):
    conn = connect()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO credentials (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value)
    )
    conn.commit()
    conn.close()


def get_credential(key: str):
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT value FROM credentials WHERE key=?", (key,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


# ── Invoices ──────────────────────────────────

def save_invoice(filename: str, vendor_name: str, amount: float):
    conn = connect()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO invoices (filename, vendor_name, amount) VALUES (?, ?, ?)",
        (filename, vendor_name, amount)
    )
    conn.commit()
    conn.close()


def get_invoices() -> list:
    conn = connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT filename, vendor_name, amount, created_at FROM invoices ORDER BY created_at DESC"
    )
    rows = cur.fetchall()
    conn.close()
    return [
        {"filename": r[0], "vendor_name": r[1], "amount": r[2], "created_at": r[3]}
        for r in rows
    ]


def get_spend_summary() -> dict:
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT SUM(amount) FROM invoices")
    total = cur.fetchone()[0] or 0
    cur.execute("SELECT COUNT(*) FROM invoices")
    count = cur.fetchone()[0]
    cur.execute(
        "SELECT vendor_name, SUM(amount) as t FROM invoices GROUP BY vendor_name ORDER BY t DESC LIMIT 1"
    )
    row = cur.fetchone()
    conn.close()
    return {
        "total_spend": total,
        "total_orders": count,
        "top_supplier": row[0] if row else "—",
    }


# Run at import
init_db()