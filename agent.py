# agent.py
# Orchestrator — routes intent to correct handler

import os
import re
import json
import logging
from dotenv import load_dotenv
from groq import Groq

from reterival import retrieve
from rag import search_documents
from web_tools import search_web
from gmail import send_email
from memory import save_message, get_history

load_dotenv()
logger = logging.getLogger(__name__)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL_FAST = "llama3-8b-8192"
MODEL_FULL = "llama3-70b-8192"


# ── Intent ────────────────────────────────────

def detect_intent(q: str) -> str:
    ql = q.lower()
    if any(x in ql for x in ["send email", "mail to", "email to", "notify supplier", "contact supplier", "send a mail"]):
        return "email"
    if any(x in ql for x in ["search web", "latest", "current price", "market price", "news", "search online"]):
        return "web_search"
    if any(x in ql for x in ["safety", "ppe", "procedure", "manual", "outage", "maintenance", "equipment", "transformer", "switchgear", "cable"]):
        return "rag"
    if any(x in ql for x in ["how many", "count", "total"]):
        return "summary"
    if any(x in ql for x in ["show", "list", "what are"]):
        return "list"
    if any(x in ql for x in ["best", "top", "highest", "largest", "compare"]):
        return "compare"
    if any(x in ql for x in ["create", "generate", "draft", "prepare", "recommend"]):
        return "action"
    if any(x in ql for x in ["why", "how", "explain"]):
        return "explain"
    return "general"


# ── Query rewrite ─────────────────────────────

def rewrite_query(query: str, history: list) -> str:
    if not history:
        return query
    convo = "\n".join(f"{h['role']}: {h['content']}" for h in history[-6:])
    try:
        res = client.chat.completions.create(
            model=MODEL_FAST,
            messages=[{"role": "user", "content":
                f"Rewrite the latest request as a standalone query, resolving pronouns from history.\n\nHistory:\n{convo}\n\nRequest: {query}\n\nReturn only the rewritten query."}],
            temperature=0,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Rewrite failed: {e}")
        return query


# ── Email ─────────────────────────────────────

def parse_email_intent(query: str, chunks: list) -> dict:
    context = "\n".join(c["text"] for c in chunks[:5])
    try:
        res = client.chat.completions.create(
            model=MODEL_FULL,
            messages=[{"role": "user", "content":
                f"""Extract email details from the request. Return ONLY valid JSON with keys: to_name, to_email, subject, body.
Use supplier data from context. If email unknown, set to_email to "unknown".
Write a professional procurement email body.

Request: {query}
Context: {context}"""}],
            temperature=0.2,
        )
        raw = re.sub(r"```json|```", "", res.choices[0].message.content).strip()
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Email parse failed: {e}")
        return {}


# ── Web search ────────────────────────────────

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
        res = client.chat.completions.create(
            model=MODEL_FULL,
            messages=[{"role": "user", "content":
                f"Summarise these web results for a procurement manager at a power utility.\n\nQuery: {query}\n\n{snippets}"}],
            temperature=0.2,
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"Web summary error: {e}"


# ── Structured data answer ────────────────────

def answer(query: str, chunks: list, intent: str) -> str:
    context = "\n".join(f"[{i+1}] {c['source']}\n{c['text']}" for i, c in enumerate(chunks))
    res = client.chat.completions.create(
        model=MODEL_FULL,
        messages=[{"role": "user", "content":
            f"""You are OpsPilot AI for a power utility procurement team.
Question: {query}
Intent: {intent}
Data:\n{context}

Use only provided data. Bullets for lists, totals for summaries, best-first for comparisons. Be concise."""}],
        temperature=0.2,
    )
    return res.choices[0].message.content


def fallback_answer(chunks: list, intent: str) -> str:
    if not chunks:
        return "No relevant data found."
    if intent == "summary":
        return f"Found {len(chunks)} relevant records."
    return "\n".join(f"• {c['text'][:200]}" for c in chunks[:6])


def needs_llm(query: str, intent: str) -> bool:
    if intent in ["email", "web_search", "rag", "action", "compare", "explain"]:
        return True
    if intent in ["list", "summary"] and any(x in query.lower() for x in ["show", "list", "what are", "how many", "count"]):
        return False
    if any(x in query.lower() for x in ["pending", "supplier", "invoice", "amount", "city"]):
        return False
    return True


# ── Main agent ────────────────────────────────

async def run_agent(user_name: str, message: str) -> str:
    save_message(user_name, "user", message)
    history      = get_history(user_name)
    intent       = detect_intent(message)
    search_query = rewrite_query(message, history)

    # EMAIL
    if intent == "email":
        chunks     = retrieve(search_query)
        email_data = parse_email_intent(search_query, chunks)
        if not email_data or email_data.get("to_email") == "unknown":
            result = "⚠️ Couldn't find a supplier email. Please specify the supplier name or their email directly."
        else:
            status = send_email(email_data["to_email"], email_data["to_name"], email_data["subject"], email_data["body"])
            result = f"{status}\n\nTo: {email_data['to_name']} ({email_data['to_email']})\nSubject: {email_data['subject']}\n\n{email_data['body'][:400]}…"

    # WEB SEARCH
    elif intent == "web_search":
        result = summarise_web(search_query, search_web(search_query))

    # PDF RAG
    elif intent == "rag":
        result = search_documents(search_query)

    # SIMPLE (no LLM)
    elif not needs_llm(message, intent):
        result = fallback_answer(retrieve(search_query), intent)

    # LLM + structured data
    else:
        chunks = retrieve(search_query)
        try:
            result = answer(search_query, chunks, intent)
        except Exception as e:
            logger.error(f"LLM failed: {e}")
            result = fallback_answer(chunks, intent)

    save_message(user_name, "assistant", result)
    return result