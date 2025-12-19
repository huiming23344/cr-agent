from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence

from cr_agent.rate_limiter import RateLimiterProtocol


class PromptBuilder(Protocol):
    """Builds the system prompt for an agent."""

    def build(self) -> str:
        ...


@dataclass
class StaticPromptBuilder(PromptBuilder):
    """Simple builder that always returns a static prompt string."""

    prompt: str

    def build(self) -> str:
        return self.prompt


@dataclass
class AgentRuntimeConfig:
    """Common runtime settings for a domain agent."""

    llm: Any
    prompt_builder: PromptBuilder
    tools: Sequence[Any]
    response_format: Optional[Any] = None
    name: Optional[str] = None


class BaseDomainAgent:
    """Base class for domain agents; subclasses choose the execution strategy."""

    def __init__(
        self,
        *,
        llm: Any,
        prompt_builder: PromptBuilder,
        tools: Optional[Sequence[Any]] = None,
        response_format: Optional[Any] = None,
        name: Optional[str] = None,
        rate_limiter: Optional[RateLimiterProtocol] = None,
    ):
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.tools = list(tools or [])
        self.response_format = response_format
        self.name = name or self.__class__.__name__
        self.rate_limiter = rate_limiter
        self.runtime = self._create_runtime()

    def _create_runtime(self):
        raise NotImplementedError("Subclasses must implement _create_runtime()")

    def build_prompt(self) -> str:
        return self.prompt_builder.build()

    async def ainvoke(self, *args, **kwargs):
        return await self.runtime.ainvoke(*args, **kwargs)

    def invoke(self, *args, **kwargs):
        return self.runtime.invoke(*args, **kwargs)
