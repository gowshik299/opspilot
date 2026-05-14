# rag.py
# RAG V9 — BM25 + pgvector Hybrid

import os
import re
import math
import pickle
import logging
from collections import defaultdict

import pdfplumber
import numpy as np
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text

from langsmith import traceable

from config import DOCUMENTS_DIR, RAG_STORE, PDF_FILES

load_dotenv()
logger = logging.getLogger(__name__)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.0,
    max_tokens=512,
)
rerank_llm = ChatGroq(
    model="llama3-8b-8192",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    max_tokens=20,
)

# Load model once
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Embedding model loaded!")

SYNONYMS = {
    "ppe":         ["personal protective equipment", "helmet", "gloves", "goggles", "vest"],
    "safety":      ["hazard", "protection", "precautions", "permit", "lockout", "tagout"],
    "transformer": ["HV", "LV", "substation", "winding", "oil", "bushing"],
    "outage":      ["shutdown", "restoration", "breaker", "fault", "emergency", "trip"],
    "maintenance": ["inspection", "repair", "checklist", "service", "overhaul"],
    "fire":        ["extinguisher", "evacuation", "smoke", "sprinkler"],
    "cable":       ["conductor", "wire", "insulation", "termination"],
    "switchgear":  ["panel", "breaker", "relay", "protection", "interlock"],
}

HEADING_RE = re.compile(r"^(?:\d+[\.\d]*\s+[A-Z]|[A-Z][A-Z\s]{4,}$)", re.MULTILINE)
TOPIC_BOOSTS = {
    "safety": ["safety_manual.pdf"],
    "ppe": ["safety_manual.pdf"],
    "outage": ["outage_procedures.pdf"],
    "shutdown": ["outage_procedures.pdf"],
    "maintenance": ["equipment_maintenance.pdf"],
    "transformer": ["equipment_maintenance.pdf"],
}


# ── PDF reader ────────────────────────────────

def read_pdf(path: str) -> list:
    pages = []
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                t = page.extract_text()
                if t and len(t.strip()) > 40:
                    pages.append((i + 1, t.strip()))
    except Exception as e:
        logger.warning(f"PDF read error {path}: {e}")
    return pages


# ── Chunking ──────────────────────────────────

def section_chunks(text: str, source: str, page: int, max_words=160, overlap=30) -> list:
    chunks = []
    splits = HEADING_RE.split(text)
    headings = HEADING_RE.findall(text)
    sections = [(headings[i].strip() if i < len(headings) else "", p.strip())
                for i, p in enumerate(splits[1:])] or [("", text.strip())]

    for heading, body in sections:
        words = body.split()
        i = 0
        while i < len(words):
            chunk_text = " ".join(words[i: i + max_words])
            if len(chunk_text.strip()) > 60:
                chunks.append({
                    "source": source, "page": page, "section": heading,
                    "text": (f"[{heading}] " if heading else "") + chunk_text,
                })
            i += max_words - overlap
    return chunks


# ── BM25 ──────────────────────────────────────

class BM25:
    def __init__(self, corpus: list, k1=1.5, b=0.75):
        self.k1 = k1; self.b = b; self.N = len(corpus)
        self.avgdl = sum(len(d.split()) for d in corpus) / max(self.N, 1)
        self.df: dict = defaultdict(int)
        self.tf: list = []
        for doc in corpus:
            freq: dict = defaultdict(int)
            for t in self._tok(doc): freq[t] += 1
            self.tf.append(freq)
            for t in set(freq): self.df[t] += 1

    def _tok(self, text): return re.findall(r"[a-z0-9]+", text.lower())

    def score(self, query: str, idx: int) -> float:
        freq = self.tf[idx]; dl = sum(freq.values()); s = 0.0
        for t in self._tok(query):
            if t not in freq: continue
            idf = math.log((self.N - self.df[t] + 0.5) / (self.df[t] + 0.5) + 1)
            tf_n = (freq[t] * (self.k1 + 1)) / (freq[t] + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            s += idf * tf_n
        return s

    def scores(self, query: str) -> list:
        return [self.score(query, i) for i in range(self.N)]


# ── pgvector functions ────────────────────────

def get_db_engine():
    return create_engine(os.getenv("DATABASE_URL"))


def store_chunks_pgvector(chunks: list, embeddings):
    """Store chunks and embeddings in PostgreSQL"""
    engine = get_db_engine()
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM document_chunks"))
        for i, chunk in enumerate(chunks):
            embedding_list = embeddings[i].tolist()
            conn.execute(text("""
                INSERT INTO document_chunks (source, chunk_text, embedding)
                VALUES (:source, :text, :embedding)
            """), {
                "source": chunk["source"],
                "text": chunk["text"],
                "embedding": str(embedding_list)
            })
        conn.commit()
    print(f"✅ Stored {len(chunks)} chunks in pgvector")


def search_pgvector(query_embedding: list, top_k: int = 10) -> list:
    """Search similar chunks using pgvector cosine similarity"""
    engine = get_db_engine()
    embedding_str = str(query_embedding)

    with engine.connect() as conn:
        results = conn.execute(text("""
            SELECT source, chunk_text,
                   1 - (embedding <=> CAST(:query_vec AS vector)) as similarity
            FROM document_chunks
            ORDER BY embedding <=> CAST(:query_vec AS vector)
            LIMIT :top_k
        """), {
            "query_vec": embedding_str,
            "top_k": top_k
        })
        return [
            {"source": r[0], "text": r[1], "score": float(r[2])}
            for r in results
        ]


# ── Build index ───────────────────────────────

def build_index():
    logger.info("Building RAG index...")
    all_chunks = []

    for fname in PDF_FILES:
        path = os.path.join(DOCUMENTS_DIR, fname)
        if not os.path.exists(path):
            logger.warning(f"PDF not found: {path}")
            continue
        for page_num, page_text in read_pdf(path):
            all_chunks.extend(section_chunks(page_text, fname, page_num))

    if not all_chunks:
        logger.error("No chunks found")
        return

    texts = [c["text"] for c in all_chunks]

    # BM25 uses pickle (keyword search)
    bm25 = BM25(texts)
    with open(RAG_STORE, "wb") as f:
        pickle.dump((bm25, all_chunks), f)

    # Embeddings go to pgvector
    print(f"Embedding {len(texts)} chunks...")
    embeddings = embedder.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings)
    store_chunks_pgvector(all_chunks, embeddings)

    print(f"✅ Index built: {len(all_chunks)} chunks")


def load_index():
    if not os.path.exists(RAG_STORE):
        build_index()
    if not os.path.exists(RAG_STORE):
        return BM25([]), []
    with open(RAG_STORE, "rb") as f:
        return pickle.load(f)


# ── Retrieval ─────────────────────────────────

def _norm(scores: list) -> list:
    arr = np.array(scores)
    mx = arr.max()
    return (arr / mx if mx > 0 else arr).tolist()


@traceable(name="retrieve_candidates")
def retrieve_candidates(query: str, top_k: int = 10) -> list:
    """Hybrid search: BM25 (40%) + pgvector (60%)"""
    bm25, all_chunks = load_index()

    # Expand query with synonyms
    expanded = query + " " + " ".join(
        t for k, ts in SYNONYMS.items() if k in query.lower() for t in ts
    )

    # BM25 keyword search
    bm25_scores = bm25.scores(expanded)
    bm25_norm = _norm(bm25_scores)

    # pgvector semantic search
    query_embedding = embedder.encode([query])[0].tolist()
    pg_results = search_pgvector(query_embedding, top_k)

    # Combine results
    seen = set()
    combined = []

    # pgvector results (60% weight)
    for r in pg_results:
        key = r["text"][:50]
        if key not in seen:
            seen.add(key)
            combined.append({
                "text": r["text"],
                "source": r["source"],
                "score": r["score"] * 0.6
            })

    # BM25 results (40% weight)
    bm25_top = np.argsort(bm25_scores)[::-1][:top_k]
    for idx in bm25_top:
        if idx < len(all_chunks):
            chunk = all_chunks[idx]
            key = chunk["text"][:50]
            if key not in seen:
                seen.add(key)
                combined.append({
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "score": float(bm25_norm[idx]) * 0.4
                })

    return sorted(combined, key=lambda x: x["score"], reverse=True)[:top_k]


# ── MMR ───────────────────────────────────────

def mmr_select(query: str, chunks: list, k: int = 5, lam: float = 0.6) -> list:
    if len(chunks) <= k:
        return chunks

    texts = [query] + [c["text"] for c in chunks]
    embs = embedder.encode(texts)

    rel = cosine_similarity([embs[0]], embs[1:])[0]
    sim = cosine_similarity(embs[1:])

    sel, rem = [], list(range(len(chunks)))
    for _ in range(k):
        if not rem: break
        best = (max(rem, key=lambda i: rel[i]) if not sel
                else max(rem, key=lambda i: lam * rel[i] - (1 - lam) * max(sim[i][j] for j in sel)))
        sel.append(best)
        rem.remove(best)

    return [chunks[i] for i in sel]


# ── Rerank ────────────────────────────────────

def rerank_chunks(query: str, chunks: list) -> list:
    if len(chunks) <= 3:
        return chunks
    try:
        res = rerank_llm.invoke([
            {"role": "user", "content":
                f"Query: {query}\n\nChunks:\n" +
                "\n".join(f"[{i+1}] {c['text'][:300]}" for i, c in enumerate(chunks)) +
                "\n\nReturn only 3 numbers comma-separated."}
        ])
        nums = re.findall(r"\d+", res.content)
        chosen = [chunks[int(n)-1] for n in nums[:3] if 0 < int(n) <= len(chunks)]
        return chosen or chunks[:3]
    except Exception:
        return chunks[:3]


# ── Answer ────────────────────────────────────

@traceable(name="grounded_answer")
def grounded_answer(query: str, chunks: list) -> str:
    context = "\n".join(
        f"[{i+1}] {c['source']} p{c.get('page','')}\n{c['text'][:1200]}"
        for i, c in enumerate(chunks)
    )
    res = llm.invoke([
        {
            "role": "system",
            "content": """You are a helpful assistant for a power utility company.
STRICT RULES:
- Answer ONLY using the context provided
- Copy exact values, numbers, specifications from context
- Never make up information
- If not in context say "Not found in the available manuals"
- Be direct and specific"""
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        }
    ])
    return res.content


# ── Main ──────────────────────────────────────

def search_documents(query: str) -> str:
    try:
        candidates = retrieve_candidates(query, top_k=10)
        if not candidates:
            return "No relevant information found in the manuals."
        selected = mmr_select(query, candidates, k=6)
        reranked = rerank_chunks(query, selected)
        return grounded_answer(query, reranked)
    except Exception as e:
        logger.error(f"RAG error: {e}")
        return f"Search error: {e}"


def rebuild_index():
    if os.path.exists(RAG_STORE):
        os.remove(RAG_STORE)
    build_index()