# mcp_procurement.py
import os as _os, sys as _sys
_log = r"C:\Users\gowshik.t\proc-agent\mcp_debug.log"
with open(_log, "a") as _f:
    _f.write(f"=== STARTED pid={_os.getpid()} exe={_sys.executable} cwd={_os.getcwd()} ===\n")
    _f.flush()

import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

with open(_log, "a") as _f:
    _f.write("mcp imports ok\n"); _f.flush()

from tools import (
    get_suppliers, top_suppliers, suppliers_by_city,
    get_procurement_history, spend_summary,
    check_alerts, pending_summary, get_invoice_summary,
)

with open(_log, "a") as _f:
    _f.write("tools imports ok\n"); _f.flush()

server = Server("procurement")

TOOLS = [
    types.Tool(
        name="suppliers",
        description="Get approved suppliers list",
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="best_suppliers",
        description="Get top suppliers by order count",
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="suppliers_in_city",
        description="Get suppliers in a specific city",
        inputSchema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    ),
    types.Tool(
        name="procurement_history",
        description="Get recent procurement history",
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="spending_summary",
        description="Get total spend breakdown by category",
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="alerts",
        description="Get high priority pending requirement alerts",
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="pending",
        description="Get pending requirements summary",
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="invoices",
        description="Get invoice summary",
        inputSchema={"type": "object", "properties": {}},
    ),
]


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.Content]:
    if name == "suppliers":
        result = get_suppliers()
    elif name == "best_suppliers":
        result = top_suppliers()
    elif name == "suppliers_in_city":
        result = suppliers_by_city(arguments.get("city", ""))
    elif name == "procurement_history":
        result = get_procurement_history()
    elif name == "spending_summary":
        result = spend_summary()
    elif name == "alerts":
        result = check_alerts()
    elif name == "pending":
        result = pending_summary()
    elif name == "invoices":
        result = get_invoice_summary()
    else:
        result = f"Unknown tool: {name}"

    return [types.TextContent(type="text", text=result)]


async def main():
    with open(_log, "a") as _f:
        _f.write("main() called — entering stdio_server\n"); _f.flush()
    async with stdio_server() as (read_stream, write_stream):
        with open(_log, "a") as _f:
            _f.write("stdio_server ready — running MCP\n"); _f.flush()
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )
    with open(_log, "a") as _f:
        _f.write("server.run() exited\n"); _f.flush()


if __name__ == "__main__":
    import traceback
    try:
        asyncio.run(main())
    except Exception as e:
        with open(_log, "a") as _f:
            _f.write(f"CRASH: {e}\n{traceback.format_exc()}\n")
        _sys.exit(1)
