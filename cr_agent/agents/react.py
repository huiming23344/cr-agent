from __future__ import annotations

from langgraph.prebuilt import create_react_agent

from .base import BaseDomainAgent


class ReactDomainAgent(BaseDomainAgent):
    """React-style agent that uses LangGraph's built-in create_react_agent."""

    def _create_runtime(self):
        prompt = self.build_prompt()
        return create_react_agent(
            self.llm,
            tools=self.tools,
            prompt=prompt,
            response_format=self.response_format,
        )
