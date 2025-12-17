from __future__ import annotations

import asyncio
from pathlib import Path

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from cr_agent.config import load_openai_config, load_repo_path
from cr_agent.file_review import FileReviewEngine
from cr_agent.models import AgentState
from tools.git_tools import get_last_commit_diff

model_config = load_openai_config(timeout=600)
llm = ChatOpenAI(
    base_url=model_config.base_url,
    api_key=model_config.api_key,
    model=model_config.model_name,
    temperature=model_config.temperature,
    timeout=model_config.timeout,
)

file_reviewer = FileReviewEngine(llm, max_qps=1.0)


async def review_all_files(state: AgentState):
    """对 commit diff 中的所有文件并行审查（LLM 总 QPS 受 file_reviewer 控制）。"""
    commit_diff = state["commit_diff"]
    tasks = [file_reviewer.review_file(fd) for fd in commit_diff.files]
    return {"file_cr_result": await asyncio.gather(*tasks)} if tasks else {"file_cr_result": []}


def print_resule(state: AgentState):
    print(state["file_cr_result"])


def build_review_agent():
    agent_builder = StateGraph(AgentState)
    agent_builder.add_node("get_last_commit_diff", get_last_commit_diff)
    agent_builder.add_node("review_all_files", review_all_files)
    agent_builder.add_node("print_resule", print_resule)

    agent_builder.add_edge(START, "get_last_commit_diff")
    agent_builder.add_edge("get_last_commit_diff", "review_all_files")
    agent_builder.add_edge("review_all_files", "print_resule")
    agent_builder.add_edge("print_resule", END)
    return agent_builder.compile()


review_agent = build_review_agent()


def run_default_repo():
    default_repo_root = Path(__file__).resolve().parent
    repo_root = str(load_repo_path(default=str(default_repo_root)))
    return asyncio.run(review_agent.ainvoke({"repo_path": repo_root, "file_cr_result": []}))


if __name__ == "__main__":
    run_default_repo()
