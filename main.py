import os
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIChatModel

# 加载 .env 文件中的环境变量
_ = load_dotenv()

# 从环境变量读取配置
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")  # 默认值

# 验证必需的配置
if not API_KEY:
    raise ValueError("缺少 API_KEY 配置，请在 .env 文件中设置")
if not BASE_URL:
    raise ValueError("缺少 BASE_URL 配置，请在 .env 文件中设置")

provider = OpenAIProvider(
    api_key=API_KEY,
    base_url=BASE_URL,
)

model = OpenAIChatModel(
    MODEL_NAME,
    provider=provider,
)

agent = Agent(model=model)

if __name__ == "__main__":
    agent.to_cli_sync()