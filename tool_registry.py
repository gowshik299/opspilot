# tool_registry.py
from tools import (
    get_suppliers, top_suppliers, check_alerts,
    pending_summary, get_invoice_summary, get_procurement_history
)
from rag import search_documents

TOOL_REGISTRY = {
    "search_documents": (
        "safety procedures PPE protective equipment high voltage transformer maintenance outage procedures equipment manual technical specifications insulation voltage lineman work permit lockout",
        search_documents,
    ),
    "suppliers": (
        "suppliers vendor list approved supplier contact details name city category top suppliers best supplier who are suppliers show suppliers",
        get_suppliers,
    ),
    "procurement": (
        "pending requirements purchase history spending budget invoices procurement orders vendor comparison priority open status items needed buy order total spend highest purchase",
        None,
    ),
    "alerts": (
        "alerts warnings overdue urgent high priority deadlines due soon critical pending high priority items",
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

QUERY_TOOLS = {"search_documents", "web_search"}
