from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from cr_agent.config import load_openai_config
from cr_agent.file_review import AsyncRateLimiter, FileReviewEngine
from cr_agent.models import AgentState
from cr_agent.rate_limiter import RateLimitedLLM
from cr_agent.reporting import render_markdown_report, summarize_to_cli, write_markdown_report
from cr_agent.profile import ProfileConfig, RepoProfile, load_profile
from tools.git_tools import get_last_commit_diff


def _load_env(env_file: Optional[str]) -> None:
    """Load .env or specified env file without overriding existing env vars."""
    if env_file:
        load_dotenv(env_file, override=False)
    else:
        default_env = Path(".env")
        if default_env.exists():
            load_dotenv(default_env, override=False)


def _resolve_repo_path(arg_repo: Optional[str]) -> str:
    if arg_repo:
        return str(Path(arg_repo).expanduser().resolve())
    env_repo = os.getenv("CR_REPO_PATH")
    if env_repo:
        return str(Path(env_repo).expanduser().resolve())
    return str(Path(__file__).resolve().parent)


def _select_profile(profile_cfg: ProfileConfig, repo_path: str) -> RepoProfile:
    return profile_cfg.match_repo(repo_path)


def _build_review_agent(file_reviewer: FileReviewEngine):
    async def review_all_files(state: AgentState):
        commit_diff = state["commit_diff"]
        tasks = [file_reviewer.review_file(fd) for fd in commit_diff.files]
        return {"file_cr_result": await asyncio.gather(*tasks)} if tasks else {"file_cr_result": []}

    def render_report(state: AgentState):
        return {
            "report_markdown": render_markdown_report(
                repo_path=state["repo_path"],
                commit_diff=state["commit_diff"],
                file_results=state.get("file_cr_result", []),
            )
        }

    agent_builder = StateGraph(AgentState)
    agent_builder.add_node("get_last_commit_diff", get_last_commit_diff)
    agent_builder.add_node("review_all_files", review_all_files)
    agent_builder.add_node("render_report", render_report)

    agent_builder.add_edge(START, "get_last_commit_diff")
    agent_builder.add_edge("get_last_commit_diff", "review_all_files")
    agent_builder.add_edge("review_all_files", "render_report")
    agent_builder.add_edge("render_report", END)
    return agent_builder.compile()


def main():
    parser = argparse.ArgumentParser(description="Run code review agent.")
    parser.add_argument("--repo", help="Repository root path (arg > env > .env).")
    parser.add_argument("--profile", help="Profile YAML for repo/domain/skip rules (arg > env > none).")
    parser.add_argument("--env-file", dest="env_file", help="Custom .env file to load (no override).")
    args = parser.parse_args()

    _load_env(args.env_file)
    repo_path = _resolve_repo_path(args.repo)

    profile_cfg: Optional[ProfileConfig] = None
    selected_repo: Optional[RepoProfile] = None
    profile_path = args.profile or os.getenv("CR_PROFILE_PATH")
    if profile_path:
        profile_cfg = load_profile(Path(profile_path))
        selected_repo = _select_profile(profile_cfg, repo_path)

    allowed_tags = selected_repo.domains if selected_repo else None
    blacklist_patterns = selected_repo.skip_regex if selected_repo else None
    blacklist_basenames = selected_repo.skip_basenames if selected_repo else None

    max_qps_value = os.getenv("CR_MAX_QPS")
    rate_limiter = None
    if max_qps_value:
        try:
            max_qps = float(max_qps_value)
            if max_qps > 0:
                rate_limiter = AsyncRateLimiter(max_qps)
        except ValueError as exc:
            raise ValueError(f"CR_MAX_QPS must be a number, got {max_qps_value}") from exc

    model_config = load_openai_config(timeout=600)
    llm_base = ChatOpenAI(
        base_url=model_config.base_url,
        api_key=model_config.api_key,
        model=model_config.model_name,
        temperature=model_config.temperature,
        timeout=model_config.timeout,
    )
    llm = RateLimitedLLM(llm_base, rate_limiter) if rate_limiter else llm_base

    file_reviewer = FileReviewEngine(
        llm,
        rate_limiter=rate_limiter,
        allowed_tags=allowed_tags,
        blacklist_patterns=blacklist_patterns,
        blacklist_basenames=blacklist_basenames,
    )
    review_agent = _build_review_agent(file_reviewer)

    result = asyncio.run(review_agent.ainvoke({"repo_path": repo_path, "file_cr_result": []}))

    file_results = result.get("file_cr_result", [])
    commit_diff = result.get("commit_diff")
    report_text = result.get("report_markdown") or render_markdown_report(
        repo_path=repo_path,
        commit_diff=commit_diff,
        file_results=file_results,
    )
    report_dir_override = os.getenv("CR_REPORT_DIR")
    report_path = write_markdown_report(
        repo_path=repo_path,
        commit_diff=commit_diff,
        file_results=file_results,
        custom_dir=report_dir_override,
        report_text=report_text,
    )
    summarize_to_cli(commit_diff=commit_diff, file_results=file_results, report_path=report_path)
    return result


if __name__ == "__main__":
    main()
