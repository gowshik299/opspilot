# agent.py
import os
import re
import json
import pickle
import logging
from dotenv import load_dotenv
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity

from rag import embedder
from memory import save_message, get_history
from mcp_server import TOOL_REGISTRY, QUERY_TOOLS
from cache import get_cached, set_cached

load_dotenv()
logger = logging.getLogger(__name__)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL             = "llama-3.3-70b-versatile"
EMBEDDINGS_CACHE  = "data/agent_embeddings.pkl"

TOOL_EMBEDDINGS: dict = {}


# ── Routing embeddings ────────────────────────────────────────────────────────

def get_tool_embeddings() -> dict:
    global TOOL_EMBEDDINGS
    if TOOL_EMBEDDINGS:
        return TOOL_EMBEDDINGS

    if os.path.exists(EMBEDDINGS_CACHE):
        with open(EMBEDDINGS_CACHE, "rb") as f:
            cached = pickle.load(f)
        if set(cached.keys()) == set(TOOL_REGISTRY.keys()):
            TOOL_EMBEDDINGS = cached
            return TOOL_EMBEDDINGS

    os.makedirs("data", exist_ok=True)
    descs = [desc for desc, _ in TOOL_REGISTRY.values()]
    embs  = embedder.encode(descs)
    TOOL_EMBEDDINGS = {tool: embs[i] for i, tool in enumerate(TOOL_REGISTRY)}
    with open(EMBEDDINGS_CACHE, "wb") as f:
        pickle.dump(TOOL_EMBEDDINGS, f)
    return TOOL_EMBEDDINGS


def route_query(query: str) -> str:
    embeddings = get_tool_embeddings()
    query_emb  = embedder.encode([query])
    scores     = {
        tool: float(cosine_similarity(query_emb, [emb])[0][0])
        for tool, emb in embeddings.items()
    }
    best = max(scores, key=scores.get)
    logger.info(f"Route '{query[:50]}' → {best} ({scores[best]:.3f})")
    return best


# ── Query rewriter ────────────────────────────────────────────────────────────

def rewrite_query(query: str, history: list) -> str:
    if not history:
        return query
    convo = "\n".join(f"{h['role']}: {h['content']}" for h in history[-6:])
    try:
        res = groq_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content":
                f"Rewrite as a standalone query resolving pronouns.\n\nHistory:\n{convo}\n\nRequest: {query}\n\nReturn only the rewritten query."}],
            temperature=0,
            max_tokens=100,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Rewrite failed: {e}")
        return query


# ── Email intent parser ───────────────────────────────────────────────────────

def parse_email_intent(query: str, context: str) -> dict:
    try:
        res = groq_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content":
                f"""Extract email details. Return ONLY valid JSON with keys: to_name, to_email, subject, body.
Request: {query}
Context: {context}"""}],
            temperature=0.2,
            max_tokens=800,
        )
        raw = re.sub(r"```json|```", "", res.choices[0].message.content).strip()
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Email parse failed: {e}")
        return {}


# ── Web search summariser ─────────────────────────────────────────────────────

def _strip_urls(text: str) -> str:
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def summarise_web(query: str, results) -> str:
    if isinstance(results, str):
        return results

    snippets = "\n".join(
        _strip_urls(r.get('content', '')[:500])
        for r in results.get("results", [])[:3]
    )

    if not snippets:
        return "No web results found."

    try:
        res = groq_client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Summarise in 3 bullet points. Only facts and prices. No URLs. No markdown links. Plain text only.",
                },
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nData:\n{snippets}\n\nSummary:",
                },
            ],
            temperature=0.0,
            max_tokens=200,
        )
        return _strip_urls(res.choices[0].message.content)
    except Exception as e:
        return f"Web search error: {e}"


# ── LLM answer over retrieval chunks ─────────────────────────────────────────

def llm_answer(query: str, chunks: list) -> str:
    context = "\n".join(
        f"[{i+1}] {c['source']}\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    try:
        res = groq_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content":
                    "You are OpsPilot AI for a power utility procurement team. "
                    "Use only the provided data. Be concise and specific."},
                {"role": "user", "content": f"Data:\n{context}\n\nQuestion: {query}"},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        return res.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return "\n".join(f"• {c['text'][:200]}" for c in chunks[:5])


# ── Main agent ────────────────────────────────────────────────────────────────

async def run_agent(user_name: str, message: str) -> str:
    cached = get_cached(message)
    if cached:
        return cached

    save_message(user_name, "user", message)
    history      = get_history(user_name)
    search_query = rewrite_query(message, history)
    route        = route_query(search_query)

    _, direct_fn = TOOL_REGISTRY.get(route, (None, None))

    # Direct single-function tools (no extra logic needed)
    if direct_fn is not None:
        result = direct_fn(search_query) if route in QUERY_TOOLS else direct_fn()

    # Web search: fetch then summarise with LLM
    elif route == "web_search":
        from web_tools import search_web
        result = summarise_web(search_query, search_web(search_query))

    # Procurement: retrieve from Excel/DB then answer with LLM
    elif route in ("procurement", "general"):
        from retrieval import retrieve
        chunks = retrieve(search_query)
        result = llm_answer(search_query, chunks) if chunks else "No relevant data found. Please rephrase your question."

    # Email: parse intent then send
    elif route == "email_supplier":
        from retrieval import retrieve
        chunks  = retrieve(search_query)
        context = "\n".join(c["text"] for c in chunks[:5])
        data    = parse_email_intent(search_query, context)
        to_email = data.get("to_email", "")
        if not data or not to_email or to_email == "unknown":
            result = "Couldn't find supplier email. Please specify the supplier name."
        else:
            from mcp_server import send_supplier_email
            status = send_supplier_email(
                to_email,
                data.get("to_name", ""),
                data.get("subject", ""),
                data.get("body", ""),
            )
            result = (
                f"{status}\n\n"
                f"To: {data.get('to_name')} ({to_email})\n"
                f"Subject: {data.get('subject')}\n\n"
                f"{data.get('body', '')[:400]}…"
            )

    else:
        result = "I couldn't understand that request. Please try rephrasing."

    set_cached(message, result)
    save_message(user_name, "assistant", result)
    return result
