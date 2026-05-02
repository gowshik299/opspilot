# web_tools.py

import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

_client = None

def get_client():
    global _client
    if _client is None:
        _client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return _client


def search_web(query: str) -> dict | str:
    try:
        return get_client().search(query=query, search_depth="advanced", max_results=5)
    except Exception as e:
        return f"Web search unavailable: {e}"