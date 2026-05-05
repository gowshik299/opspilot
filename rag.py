# rag.py
# RAG V8 — BM25 + Sentence Transformers Hybrid

import os
import re
import math
import pickle
import logging
from collections import defaultdict

import pdfplumber
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from config import DATA_DIR, RAG_STORE, PDF_FILES

load_dotenv()
logger = logging.getLogger(__name__)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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


# ── Build index ───────────────────────────────

def build_index():
    logger.info("Building RAG index...")
    all_chunks = []

    for fname in PDF_FILES:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            logger.warning(f"PDF not found: {path}")
            continue
        for page_num, page_text in read_pdf(path):
            all_chunks.extend(section_chunks(page_text, fname, page_num))

    if not all_chunks:
        logger.error("No chunks found")
        return

    texts = [c["text"] for c in all_chunks]

    # BM25
    bm25 = BM25(texts)

    # Sentence Transformers embeddings
    print(f"Embedding {len(texts)} chunks...")
    embeddings = embedder.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings)

    with open(RAG_STORE, "wb") as f:
        pickle.dump((bm25, embeddings, all_chunks), f)

    print(f"✅ Index built: {len(all_chunks)} chunks")


def load_index():
    if not os.path.exists(RAG_STORE):
        build_index()
    with open(RAG_STORE, "rb") as f:
        return pickle.load(f)


# ── Retrieval ─────────────────────────────────

def _norm(scores: list) -> list:
    arr = np.array(scores)
    mx = arr.max()
    return (arr / mx if mx > 0 else arr).tolist()


def retrieve_candidates(query: str, top_k: int = 10) -> list:
    bm25, embeddings, meta = load_index()

    expanded = query + " " + " ".join(
        t for k, ts in SYNONYMS.items() if k in query.lower() for t in ts
    )

    bm25_norm = _norm(bm25.scores(expanded))

    query_emb = embedder.encode([query])
    semantic_scores = cosine_similarity(query_emb, embeddings)[0]
    semantic_norm = _norm(semantic_scores.tolist())

    ql = query.lower()
    boosted = {s for k, srcs in TOPIC_BOOSTS.items() if k in ql for s in srcs}
    qwords = set(re.findall(r"[a-z0-9]+", ql))

    scored = []
    for i, chunk in enumerate(meta):
        h = 0.40 * bm25_norm[i] + 0.60 * semantic_norm[i]
        h += len(qwords & set(re.findall(r"[a-z0-9]+", chunk["text"].lower()))) * 0.01
        if chunk["source"] in boosted:
            h += 0.10
        scored.append((h, i))

    scored.sort(reverse=True)
    return [dict(**meta[i], score=round(s, 4)) for s, i in scored[:top_k] if s > 0.01]


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
        res = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content":
                f"Query: {query}\n\nChunks:\n" +
                "\n".join(f"[{i+1}] {c['text'][:300]}" for i, c in enumerate(chunks)) +
                "\n\nReturn only 3 numbers comma-separated."}],
            temperature=0,
            max_tokens=20,
        )
        nums = re.findall(r"\d+", res.choices[0].message.content)
        chosen = [chunks[int(n)-1] for n in nums[:3] if 0 < int(n) <= len(chunks)]
        return chosen or chunks[:3]
    except Exception:
        return chunks[:3]


# ── Answer ────────────────────────────────────

def grounded_answer(query: str, chunks: list) -> str:
    context = "\n".join(
        f"[{i+1}] {c['source']} p{c.get('page','')}\n{c['text'][:1200]}"
        for i, c in enumerate(chunks)
    )
    res = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
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
        ],
        temperature=0.0,
        max_tokens=512
    )
    return res.choices[0].message.content


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