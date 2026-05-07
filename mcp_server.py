# mcp_server.py - Standalone MCP Server
import os
from fastmcp import FastMCP
from contextlib import asynccontextmanager
from fastapi import FastAPI

mcp = FastMCP("opspilot-procurement")


@mcp.tool()
def get_suppliers() -> str:
    """Get approved suppliers list with ratings and contact details"""
    from tools import get_suppliers as fn
    return fn()

@mcp.tool()
def best_suppliers() -> str:
    """Get top suppliers ranked by order count and performance"""
    from tools import top_suppliers as fn
    return fn()

@mcp.tool()
def suppliers_in_city(city: str) -> str:
    """Get suppliers in a specific city"""
    from tools import suppliers_by_city as fn
    return fn(city)

@mcp.tool()
def procurement_history() -> str:
    """Get recent procurement history with prices and vendors"""
    from tools import get_procurement_history as fn
    return fn()

@mcp.tool()
def spending_summary() -> str:
    """Get total spend breakdown by category"""
    from tools import spend_summary as fn
    return fn()

@mcp.tool()
def check_alerts() -> str:
    """Get high priority pending requirement alerts"""
    from tools import check_alerts as fn
    return fn()

@mcp.tool()
def pending_requirements() -> str:
    """Get all pending procurement requirements"""
    from tools import pending_summary as fn
    return fn()

@mcp.tool()
def invoice_summary() -> str:
    """Get invoice summary and payment status"""
    from tools import get_invoice_summary as fn
    return fn()

@mcp.tool()
def search_manuals(query: str) -> str:
    """Search safety manuals, outage procedures and equipment maintenance guides"""
    from rag import search_documents as fn
    return fn(query)

@mcp.tool()
def search_web(query: str) -> str:
    """Search web for current market prices and supplier information"""
    from web_tools import search_web as fn
    import asyncio
    return asyncio.run(fn(query))


# ── FastAPI with MCP lifespan ─────────────────────────────────────────────────

mcp_app = mcp.http_app(path="/")

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with mcp_app.router.lifespan_context(app):
        yield

app = FastAPI(title="OpsPilot MCP Server", lifespan=lifespan)
app.mount("/mcp", mcp_app)

@app.get("/health")
def health():
    return {"status": "running", "service": "opspilot-mcp"}


# ── TOOL_REGISTRY ─────────────────────────────────────────────────────────────
# Used by agent.py for semantic routing.

TOOL_REGISTRY: dict = {
    "search_documents": (
        "safety procedures PPE protective equipment high voltage transformer maintenance outage procedures equipment manual technical specifications insulation voltage lineman work permit lockout",
        search_manuals,
    ),
    "suppliers": (
        "show list suppliers vendor approved supplier contact details name city category",
        get_suppliers,
    ),
    "procurement": (
        "pending requirements suppliers purchase history spending budget invoices procurement orders vendor comparison priority open status items needed buy order",
        None,
    ),
    "alerts": (
        "alerts warnings overdue urgent high priority deadlines due soon critical",
        check_alerts,
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
        None,
    ),
    "general": (
        "general question answer help information",
        None,
    ),
}

QUERY_TOOLS = {"search_documents", "search_web"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
