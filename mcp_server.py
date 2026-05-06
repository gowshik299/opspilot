# mcp_server.py — Production HTTP MCP Server
import os
from mcp.server.fastmcp import FastMCP
from tools import (
    get_suppliers, top_suppliers, suppliers_by_city,
    get_procurement_history, spend_summary, highest_purchase,
    check_alerts, pending_summary, get_invoice_summary, highest_invoice,
)
from rag import search_documents
from gmail import send_email, scan_inbox as _scan_inbox
from web_tools import search_web

mcp = FastMCP("opspilot-procurement")


# ── Suppliers ────────────────────────────────────────────────────────────────

@mcp.tool()
def suppliers(category: str = "All") -> str:
    """Get approved suppliers list filtered by category."""
    return get_suppliers()

@mcp.tool()
def best_suppliers() -> str:
    """Get top suppliers by order count."""
    return top_suppliers()

@mcp.tool()
def suppliers_in_city(city: str) -> str:
    """Get suppliers in a specific city."""
    return suppliers_by_city(city)


# ── Procurement ──────────────────────────────────────────────────────────────

@mcp.tool()
def procurement_history() -> str:
    """Get recent procurement history with prices."""
    return get_procurement_history()

@mcp.tool()
def spending_summary() -> str:
    """Get total spend breakdown by category."""
    return spend_summary()

@mcp.tool()
def highest_spend() -> str:
    """Get the single largest purchase on record."""
    return highest_purchase()


# ── Alerts / Pending ─────────────────────────────────────────────────────────

@mcp.tool()
def alerts() -> str:
    """Get high priority pending requirement alerts."""
    return check_alerts()

@mcp.tool()
def pending() -> str:
    """Get pending requirements summary."""
    return pending_summary()


# ── Invoices ─────────────────────────────────────────────────────────────────

@mcp.tool()
def invoices() -> str:
    """Get invoice summary."""
    return get_invoice_summary()

@mcp.tool()
def top_invoice() -> str:
    """Get highest value invoice."""
    return highest_invoice()


# ── Documents / RAG ──────────────────────────────────────────────────────────

@mcp.tool()
def search_manuals(query: str) -> str:
    """Search safety manual, outage procedures and equipment maintenance guides."""
    return search_documents(query)


# ── Web ───────────────────────────────────────────────────────────────────────

@mcp.tool()
def web_search(query: str) -> str:
    """Search the web for live prices, news, or current information."""
    result = search_web(query)
    if isinstance(result, str):
        return result
    return "\n".join(
        r.get("content", "")[:400]
        for r in result.get("results", [])[:3]
    )


# ── Email ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def send_supplier_email(to_email: str, to_name: str, subject: str, body: str) -> str:
    """Send an email to a supplier."""
    return send_email(to_email, to_name, subject, body)

@mcp.tool()
def scan_inbox() -> str:
    """Check Gmail inbox for recent supplier replies."""
    return _scan_inbox(last_n=10)


# ── TOOL_REGISTRY ─────────────────────────────────────────────────────────────
# Used by agent.py for semantic routing.
# Each entry: route_key → (routing_description, callable_or_None)

TOOL_REGISTRY: dict = {
    "search_documents": (
        "safety procedures PPE protective equipment high voltage transformer maintenance outage procedures equipment manual technical specifications insulation voltage lineman work permit lockout",
        search_manuals,
    ),
    "suppliers": (
        "show list suppliers vendor approved supplier contact details name city category",
        suppliers,
    ),
    "procurement": (
        "pending requirements suppliers purchase history spending budget invoices procurement orders vendor comparison priority open status items needed buy order",
        None,
    ),
    "alerts": (
        "alerts warnings overdue urgent high priority deadlines due soon critical",
        alerts,
    ),
    "web_search": (
        "current market price latest news search online real time information today cost per unit INR rupees",
        None,
    ),
    "email_supplier": (
        "send email contact supplier notify vendor email price quote request write mail",
        None,
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

# Tools that receive the query string as their sole argument.
QUERY_TOOLS = {"search_documents"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=port,
    )
