# ⚡ OpsPilot — AI Procurement Intelligence Agent

A production-grade AI agent for power utility procurement teams. Built with FastAPI, MCP, RAG, Redis, and PostgreSQL — deployed on Google Cloud Run.

## 🌐 Live Demo
**Website:** https://opspilot-162649919209.asia-south2.run.app/app  
**MCP Server:** https://opspilot-mcp-162649919209.asia-south2.run.app/health

## 🏗️ Architecture
User → OpsPilot (FastAPI)
↓
agent.py (Semantic Routing)
↓
MCP Client → opspilot-mcp (MCP Server)
↓
Tools (PostgreSQL) + RAG (pgvector) + Web Search
↓
Redis Cache → Response



✨ Features

- **AI Chat Agent** — Answers procurement questions in natural language
- **Semantic Routing** — Automatically routes queries to the right tool
- **Hybrid RAG** — BM25 + pgvector semantic search over safety manuals and equipment guides
- **MCP Server** — Production HTTP MCP server for tool orchestration
- **Email Integration** — Send and scan Gmail directly from the agent
- **Web Search** — Real-time market prices via Tavily API
- **Redis Caching** — Sub-100ms responses for repeated queries
- **PostgreSQL** — Persistent storage for suppliers, procurement history, chat memory

opspilot/
├── main.py          # FastAPI routes + MCP mount
├── agent.py         # AI agent with semantic routing
├── rag.py           # Hybrid RAG pipeline (BM25 + pgvector)
├── tools.py         # PostgreSQL tool functions
├── tool_registry.py # Semantic routing registry
├── mcp_server.py    # Standalone HTTP MCP server
├── cache.py         # Redis caching layer
├── memory.py        # PostgreSQL chat memory
├── gmail.py         # Email send/scan
├── web_tools.py     # Tavily web search
├── config.py        # Configuration
├── Dockerfile       # Main app container
├── Dockerfile.mcp   # MCP server container
└── static/
└── index.html   # Frontend UI
## 🚀 Deployment

Two separate Cloud Run services:
- **opspilot** — Main website + FastAPI
- **opspilot-mcp** — MCP tool server

Auto-deploys on `git push` via Cloud Build triggers.

## 🔧 Environment Variables

```env
GROQ_API_KEY=
DATABASE_URL=
REDIS_URL=
TAVILY_API_KEY=
GMAIL=
APP_PASSWORD=
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=
LANGCHAIN_TRACING_V2=
```

## 💬 Example Queries

- "Who are our top suppliers?"
- "Check urgent alerts"
- "What are the PPE requirements for high voltage work?"
- "Search web for transformer oil price India"
- "Send email to Bharat Electricals about price quote"
- "Show pending requirements"
- "What is our total procurement spend?"

## 🏭 Built For

Power utility company procurement teams. Helps procurement officers:
- Find supplier information instantly
- Check safety procedures from manuals
- Monitor pending requirements and alerts
- Send supplier emails with AI-drafted content
- Get real-time market prices

---
Built by Gowshik — AI Engineer in training 🚀