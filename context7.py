from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import Connection  
import os


CONTEXT7_API_KEY = os.getenv("CONTEXT7_API_KEY")

if not CONTEXT7_API_KEY:
    raise ValueError("Missing CONTEXT7_API_KEY environment variable.")

# 定义服务器配置，使用正确的类型注解
servers: dict[str, Connection] = {
    "context7": {
        "transport": "stdio",  # Local subprocess communication
        "command": "npx",
        "args": ["-y", "@upstash/context7-mcp", "--api-key", CONTEXT7_API_KEY],
    },
}

context7_mcp_client = MultiServerMCPClient(servers)

__all__ = ["context7_mcp_client"]