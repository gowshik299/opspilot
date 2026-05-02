# test_agent.py

import asyncio
from agent import run_agent

result = asyncio.run(
    run_agent("Test User", "write a professional email to supplier")
)

print(result)