# retrieval.py
# HYBRID V3 - Structured + Smart Retrieval (No embeddings)

import pandas as pd
import sqlite3
import difflib

from config import DB_PATH, EXCEL_FILE

# Only Excel chunks are cached (static file).
# SQLite chunks are fetched fresh every call so new invoices always appear.
EXCEL_CACHE = None


def invalidate_cache():
    global EXCEL_CACHE
    EXCEL_CACHE = None


# ---------------------------------------------------
# BUILD STRUCTURED CHUNKS
# ---------------------------------------------------

def build_excel_chunks():
    global EXCEL_CACHE

    if EXCEL_CACHE is not None:
        return EXCEL_CACHE

    chunks = []

    try:
        sheets = pd.read_excel(EXCEL_FILE, sheet_name=None)

        for sheet, df in sheets.items():
            df = df.fillna("")
            cols = df.columns.tolist()

            for _, row in df.iterrows():
                parts = []
                for col in cols:
                    val = str(row[col]).strip()
                    if val:
                        parts.append(f"{col}: {val}")

                text = " | ".join(parts)

                chunks.append({
                    "source": sheet,
                    "text": text,
                    "columns": cols
                })

    except Exception as e:
        print(f"Excel load error: {e}")

    EXCEL_CACHE = chunks
    return chunks


def build_sqlite_chunks():
    chunks = []

    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()

        cur.execute("""
            SELECT vendor_name, amount, filename
            FROM invoices
        """)

        rows = cur.fetchall()
        conn.close()

        for r in rows:
            text = f"Vendor: {r[0]} | Amount: {r[1]} | File: {r[2]}"
            chunks.append({
                "source": "Invoices",
                "text": text,
                "columns": ["vendor", "amount", "file"]
            })

    except Exception as e:
        print(f"SQLite load error: {e}")

    return chunks


def get_all_chunks():
    # Excel is cached, SQLite is always fresh
    return build_excel_chunks() + build_sqlite_chunks()


# ---------------------------------------------------
# QUERY UNDERSTANDING
# ---------------------------------------------------

def extract_keywords(query):
    q = query.lower()
    keywords = []

    mapping = {
        "pending": ["pending", "open", "requirement"],
        "supplier": ["supplier", "vendor"],
        "invoice": ["invoice", "bill"],
        "amount": ["amount", "cost", "price"],
        "urgent": ["urgent", "high"],
        "city": ["city", "location"],
        "category": ["category", "type"]
    }

    for key, words in mapping.items():
        for w in words:
            if w in q:
                keywords.append(key)

    return keywords


# ---------------------------------------------------
# SCORING
# ---------------------------------------------------

def score_chunk(query, chunk):
    q = query.lower()
    text = chunk["text"].lower()
    source = chunk["source"]

    score = 0

    # Keyword match
    for w in q.split():
        if w in text:
            score += 2

    # Fuzzy similarity
    score += difflib.SequenceMatcher(None, q, text[:200]).ratio() * 5

    # Semantic keyword groups
    keys = extract_keywords(query)

    if "pending" in keys and source == "Pending_Requirements":
        score += 6
    if "supplier" in keys and source == "Suppliers":
        score += 6
    if "invoice" in keys and source == "Invoices":
        score += 6
    if "category" in keys and "category" in text:
        score += 3
    if "amount" in keys and "amount" in text:
        score += 3
    if "urgent" in keys and ("high" in text or "urgent" in text):
        score += 4

    return score


# ---------------------------------------------------
# MAIN RETRIEVE
# ---------------------------------------------------

def retrieve(query, top_k=8):
    all_chunks = get_all_chunks()

    ranked = []

    for c in all_chunks:
        s = score_chunk(query, c)
        if s > 1:
            ranked.append((s, c))

    ranked.sort(reverse=True, key=lambda x: x[0])

    results = []
    seen = set()

    for score, item in ranked:
        key = item["text"][:120]
        if key not in seen:
            results.append(item)
            seen.add(key)
        if len(results) >= top_k:
            break

    return results
