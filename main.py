import asyncio
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from cr_agent.config import load_openai_config
from cr_agent.rate_limiter import AsyncRateLimiter, RateLimitedLLM

from tools.shell_tools import shell_tool
from mcp_clients.context7 import context7_mcp_client

async def main():
    model_config = load_openai_config(timeout=10)
    limiter = None
    llm_base = ChatOpenAI(
        base_url=model_config.base_url,
        api_key=model_config.api_key,
        model=model_config.model_name,
        temperature=model_config.temperature,
        timeout=model_config.timeout,
    )
    llm = RateLimitedLLM(llm_base, limiter) if limiter else llm_base

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
