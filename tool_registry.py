# tool_registry.py
from tools import (
    get_suppliers, top_suppliers, check_alerts,
    pending_summary, get_invoice_summary, get_procurement_history
)
from rag import search_documents

TOOL_REGISTRY = {
    "search_documents": (
        "safety procedures PPE protective equipment high voltage transformer maintenance outage procedures equipment manual technical specifications insulation voltage lineman work permit lockout fire extinguisher checklist inspection",
        search_documents,
    ),
    "suppliers": (
        "suppliers vendor list approved supplier contact details name city category top suppliers best supplier who are suppliers show suppliers",
        get_suppliers,
    ),
    "procurement": (
        "purchase history spending budget procurement orders vendor comparison total spend highest purchase order history buying records",
        None,
    ),
    "alerts": (
        "alerts warnings overdue urgent high priority deadlines due soon critical",
        check_alerts,
    ),
    "pending": (
        "pending requirements show pending open items requisitions what needs to be ordered pending list requirements status",
        pending_summary,
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
        "check gmail inbox scan inbox supplier email replies received messages unread emails check mail inbox",
        None,
    ),
    "general": (
        "general question answer help information",
        None,
    ),
}

QUERY_TOOLS = {"search_documents", "web_search"}
