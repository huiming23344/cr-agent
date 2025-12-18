"""Agent abstractions used across domains.

This module exposes a thin base class plus a React-based implementation so
callers can swap runtime/graph strategies while reusing prompt construction.
"""

from .base import AgentRuntimeConfig, BaseDomainAgent, PromptBuilder, StaticPromptBuilder
from .react import ReactDomainAgent

__all__ = [
    "AgentRuntimeConfig",
    "BaseDomainAgent",
    "PromptBuilder",
    "StaticPromptBuilder",
    "ReactDomainAgent",
]
