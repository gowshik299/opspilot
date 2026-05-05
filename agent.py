# agent.py
# Semantic routing with Sentence Transformers

import os
import re
import json
import pickle
import logging
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from retrieval import retrieve
from rag import search_documents, embedder
from web_tools import search_web
from gmail import send_email, scan_inbox
from memory import save_message, get_history

load_dotenv()
logger = logging.getLogger(__name__)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.3-70b-versatile"
EMBEDDINGS_CACHE = "data/tool_embeddings.pkl"

TOOLS = {
    "rag": "safety procedures PPE protective equipment high voltage transformer maintenance outage procedures equipment manual technical specifications",
    "procurement": "pending requirements suppliers purchase history spending budget invoices procurement orders vendor comparison",
    "web_search": "current market price latest news search online real time information today",
    "email_supplier": "send email contact supplier notify vendor email price quote request",
    "scan_email": "check inbox new emails supplier replies any updates messages received",
    "alerts": "alerts warnings overdue urgent high priority deadlines due soon",
    "general": "general question answer help information"
}

TOOL_EMBEDDINGS = {}


# ── Cached tool embeddings ────────────────────

def get_tool_embeddings():
    global TOOL_EMBEDDINGS
    if TOOL_EMBEDDINGS:
        return TOOL_EMBEDDINGS

    if os.path.exists(EMBEDDINGS_CACHE):
        with open(EMBEDDINGS_CACHE, "rb") as f:
            TOOL_EMBEDDINGS = pickle.load(f)
        return TOOL_EMBEDDINGS

    os.makedirs("data", exist_ok=True)
    descriptions = list(TOOLS.values())
    embs = embedder.encode(descriptions)
    TOOL_EMBEDDINGS = {tool: embs[i] for i, tool in enumerate(TOOLS.keys())}

    with open(EMBEDDINGS_CACHE, "wb") as f:
        pickle.dump(TOOL_EMBEDDINGS, f)

    return TOOL_EMBEDDINGS


# ── Semantic router ───────────────────────────

def route_query(query: str) -> str:
    embeddings = get_tool_embeddings()
    query_emb = embedder.encode([query])

    scores = {}
    for tool, tool_emb in embeddings.items():
        score = cosine_similarity(query_emb, [tool_emb])[0][0]
        scores[tool] = score

    best = max(scores, key=scores.get)
    logger.info(f"Route: '{query[:40]}' → {best} ({scores[best]:.3f})")
    return best


# ── Query rewriter ────────────────────────────

def rewrite_query(query: str, history: list) -> str:
    if not history:
        return query
    convo = "\n".join(f"{h['role']}: {h['content']}" for h in history[-6:])
    try:
        res = groq_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content":
                f"Rewrite the latest request as standalone query resolving pronouns.\n\nHistory:\n{convo}\n\nRequest: {query}\n\nReturn only rewritten query."}],
            temperature=0,
            max_tokens=100
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Rewrite failed: {e}")
        return query


# ── Email parser ──────────────────────────────

def parse_email_intent(query: str, chunks: list) -> dict:
    context = "\n".join(c["text"] for c in chunks[:5])
    try:
        res = groq_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content":
                f"""Extract email details. Return ONLY valid JSON: to_name, to_email, subject, body.
Request: {query}
Context: {context}"""}],
            temperature=0.2,
            max_tokens=800
        )
        raw = re.sub(r"```json|```", "", res.choices[0].message.content).strip()
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Email parse failed: {e}")
        return {}


# ── Web summarizer ────────────────────────────

def summarise_web(query: str, results) -> str:
    if isinstance(results, str):
        return results
    snippets = "\n".join(
        f"Source: {r.get('url','')}\n{r.get('content','')[:600]}"
        for r in results.get("results", [])[:4]
    )
    if not snippets:
        return "No web results found."
    try:
        res = groq_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content":
                f"Summarise for procurement manager at power utility.\n\nQuery: {query}\n\n{snippets}"}],
            temperature=0.2,
            max_tokens=512
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"Web summary error: {e}"


# ── LLM answer ────────────────────────────────

def llm_answer(query: str, chunks: list, route: str) -> str:
    context = "\n".join(
        f"[{i+1}] {c['source']}\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    try:
        res = groq_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are OpsPilot AI for a power utility procurement team. Use only provided data. Be concise and specific."},
                {"role": "user", "content": f"Data:\n{context}\n\nQuestion: {query}"}
            ],
            temperature=0.1,
            max_tokens=1024
        )
        return res.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM failed: {e}")
        return "\n".join(f"• {c['text'][:200]}" for c in chunks[:5])


# ── Main agent ────────────────────────────────

async def run_agent(user_name: str, message: str) -> str:
    save_message(user_name, "user", message)
    history = get_history(user_name)

    search_query = rewrite_query(message, history)
    route = route_query(search_query)

    if route == "rag":
        result = search_documents(search_query)

    elif route == "email_supplier":
        chunks = retrieve(search_query)
        email_data = parse_email_intent(search_query, chunks)
        if not email_data or email_data.get("to_email") == "unknown":
            result = "⚠️ Couldn't find supplier email. Please specify supplier name."
        else:
            status = send_email(email_data["to_email"], email_data["to_name"], email_data["subject"], email_data["body"])
            result = f"{status}\n\nTo: {email_data['to_name']} ({email_data['to_email']})\nSubject: {email_data['subject']}\n\n{email_data['body'][:400]}…"

    elif route == "scan_email":
        result = scan_inbox(last_n=10)

    elif route == "web_search":
        result = summarise_web(search_query, search_web(search_query))

    elif route == "alerts":
        from tools import check_alerts
        result = check_alerts()

    else:
        chunks = retrieve(search_query)
        if chunks:
            result = llm_answer(search_query, chunks, route)
        else:
            result = "No relevant data found. Please rephrase your question."

    save_message(user_name, "assistant", result)
    return result