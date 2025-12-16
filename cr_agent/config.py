from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

_ = load_dotenv()

DEFAULT_MODEL_NAME = "gpt-4o-mini"


@dataclass(frozen=True)
class OpenAIConfig:
    base_url: str
    api_key: str
    model_name: str = DEFAULT_MODEL_NAME
    temperature: float = 0.7
    timeout: int = 60


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"缺少 {name} 配置，请在 .env 文件中设置")
    return value


def load_openai_config(
    *,
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    timeout: int = 60,
) -> OpenAIConfig:
    """Load model/runtime configuration from environment variables."""
    base_url = _require_env("BASE_URL")
    api_key = _require_env("API_KEY")
    resolved_model = model_name or os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
    return OpenAIConfig(
        base_url=base_url,
        api_key=api_key,
        model_name=resolved_model,
        temperature=temperature,
        timeout=timeout,
    )
