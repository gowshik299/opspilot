# mcp_server.py
# ─────────────────────────────────────────────────────────────────────────────
# Single source of truth for ALL agent tools.
#   ADD a tool  → write a new @mcp.tool() function below
#   REMOVE one  → delete its function (and its TOOL_REGISTRY entry at the bottom)
# ─────────────────────────────────────────────────────────────────────────────

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("opspilot")


# ── Suppliers ────────────────────────────────────────────────────────────────

@mcp.tool()
def get_suppliers() -> str:
    """List approved suppliers: name, city, category."""
    from tools import get_suppliers as fn
    return fn()

@mcp.tool()
def top_suppliers() -> str:
    """Top suppliers ranked by order volume."""
    from tools import top_suppliers as fn
    return fn()

@mcp.tool()
def suppliers_in_city(city: str) -> str:
    """Suppliers available in a specific city."""
    from tools import suppliers_by_city as fn
    return fn(city)


# ── Procurement ──────────────────────────────────────────────────────────────

@mcp.tool()
def procurement_history() -> str:
    """Recent procurement order history."""
    from tools import get_procurement_history as fn
    return fn()

@mcp.tool()
def spend_summary() -> str:
    """Total spend and breakdown by category."""
    from tools import spend_summary as fn
    return fn()

@mcp.tool()
def highest_purchase() -> str:
    """Largest single purchase on record."""
    from tools import highest_purchase as fn
    return fn()


# ── Alerts / Pending ─────────────────────────────────────────────────────────

@mcp.tool()
def check_alerts() -> str:
    """High-priority pending requirement alerts."""
    from tools import check_alerts as fn
    return fn()

@mcp.tool()
def pending_summary() -> str:
    """Count of open and pending procurement items."""
    from tools import pending_summary as fn
    return fn()


# ── Invoices ─────────────────────────────────────────────────────────────────

@mcp.tool()
def invoice_summary() -> str:
    """Recent invoices and total billed amount."""
    from tools import get_invoice_summary as fn
    return fn()

@mcp.tool()
def highest_invoice() -> str:
    """Highest-value invoice received."""
    from tools import highest_invoice as fn
    return fn()


# ── Documents / RAG ──────────────────────────────────────────────────────────

@mcp.tool()
def search_documents(query: str) -> str:
    """Search safety manuals, outage procedures, and maintenance guides."""
    from rag import search_documents as fn
    return fn(query)


# ── Web ───────────────────────────────────────────────────────────────────────

@mcp.tool()
def web_search(query: str) -> str:
    """Search the web for live prices, news, or current information."""
    from web_tools import search_web
    result = search_web(query)
    if isinstance(result, str):
        return result
    snippets = "\n".join(
        f"[{r.get('url', '')}]\n{r.get('content', '')[:500]}"
        for r in result.get("results", [])[:4]
    )
    return snippets or "No results found."


# ── Email ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def send_supplier_email(to_email: str, to_name: str, subject: str, body: str) -> str:
    """Send an email to a supplier."""
    from gmail import send_email as fn
    return fn(to_email, to_name, subject, body)

@mcp.tool()
def scan_inbox() -> str:
    """Check Gmail inbox for recent supplier replies."""
    from gmail import scan_inbox as fn
    return fn(last_n=10)


# ── TOOL_REGISTRY ─────────────────────────────────────────────────────────────
# agent.py reads this to build semantic routing embeddings.
# Each entry: route_key → (routing_description, callable_or_None)
# callable=None means agent.py handles dispatch with extra logic (see run_agent).
# QUERY_TOOLS below lists which callables receive the query string as argument.

TOOL_REGISTRY: dict = {
    "search_documents": (
        "safety procedures PPE protective equipment high voltage transformer maintenance outage procedures equipment manual technical specifications",
        search_documents,
    ),
    "suppliers": (
        "show list suppliers vendor approved supplier contact details name city category",
        get_suppliers,
    ),
    "procurement": (
        "pending requirements purchase history spending budget invoices orders vendor comparison top spend summary",
        None,  # agent dispatches to retrieval + LLM
    ),
    "alerts": (
        "alerts warnings overdue urgent high priority deadlines due soon",
        check_alerts,
    ),
    "web_search": (
        "current market price latest news search online real time information today live",
        web_search,
    ),
    "email_supplier": (
        "send email contact supplier notify vendor price quote request draft",
        None,  # agent parses intent first
    ),
    "scan_email": (
        "check inbox new emails supplier replies any updates messages received",
        scan_inbox,
    ),
    "general": (
        "general question answer help information",
        None,
    ),
}

# Tools in this set receive (query_string) as their sole argument.
QUERY_TOOLS = {"search_documents", "web_search"}


if __name__ == "__main__":
    mcp.run()
