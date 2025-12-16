import os
import asyncio
import operator

from typing import TypedDict,Annotated
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain.messages import AnyMessage
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI


_ = load_dotenv()


from tools.shell_tools import shell_tool
from mcp_clients.context7 import context7_mcp_client

# 从环境变量读取配置
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")  # 默认值


if not BASE_URL:
    raise ValueError("缺少 BASE_URL 配置，请在 .env 文件中设置")

def get_api_key() -> str:
    if API_KEY:
        return API_KEY
    else:
        raise ValueError("API_KEY 未找到，请在 .env 文件中设置")


@dataclass
class CommitDiff:
    lines: str

    

async def main():
    
    llm = ChatOpenAI(
        base_url=BASE_URL,
        api_key=get_api_key,
        model=MODEL_NAME,
        temperature=0.7,
        timeout=10,
    )

    # 使用 await 调用异步的 get_tools() 方法
    context7_tools = await context7_mcp_client.get_tools()

    agent = create_agent(
        model=llm,
        tools=[*context7_tools, shell_tool],
    )

    response = await agent.ainvoke(
        input= {"messages": [{"role": "user", "content": "请你执行pwd，并告诉我结果"}]},
    )

    print(response)

if __name__ == "__main__":
    asyncio.run(main())