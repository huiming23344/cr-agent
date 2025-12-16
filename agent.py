import asyncio
import json
import time
from pathlib import Path
from typing import Iterable, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from cr_agent.config import load_openai_config
from cr_agent.models import AgentState, FileCRResult
from tools.git_tools import get_last_commit_diff

model_config = load_openai_config(timeout=600)
llm = ChatOpenAI(
    base_url=model_config.base_url,
    api_key=model_config.api_key,
    model=model_config.model_name,
    temperature=model_config.temperature,
    timeout=model_config.timeout,
)



PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一名资深代码审查专家，请严格按照 FileCRResult 的结构化模式返回结果，并确保所有文字说明使用中文。"),
    ("human",
     "请审查以下文件 diff。\n"
     "规则：\n"
     "- 如果是二进制文件或补丁为空：needs_human_review=true，并说明原因。\n"
     "- 行号尽量使用新版本的编号；如果不确定可填 null。\n"
     "- overall_severity 应等于最严重问题的级别。\n\n"
     "输入(JSON)：\n{payload_json}")
])

MAX_PATCH_CHARS = 12000
MAX_QPS = 1.0  # 每秒最多 1 次模型调用

def _payload(file_diff) -> dict:
    patch = file_diff.patch or ""
    if len(patch) > MAX_PATCH_CHARS:
        patch = patch[:MAX_PATCH_CHARS] + "\n\n...<PATCH TRUNCATED>..."

    return {
        "file_path": file_diff.b_path or file_diff.a_path or "<unknown>",
        "change_type": file_diff.change_type,
        "is_binary": file_diff.is_binary,
        "added_lines": file_diff.added_lines,
        "deleted_lines": file_diff.deleted_lines,
        "patch": patch,
        # 其余字段随你保留
    }


prepare = RunnableLambda(lambda fd: {"payload_json": json.dumps(_payload(fd), ensure_ascii=False)})

# 关键：结构化输出
llm_structured = llm.with_structured_output(FileCRResult)

# 关键：重试（结构化解析/校验失败时）
review_chain = (prepare | PROMPT | llm_structured).with_retry(
    stop_after_attempt=3,
    retry_if_exception_type=(ValidationError, ValueError),
)

class AsyncRateLimiter:
    """简单的异步限速器，确保请求间隔 >= 1 / MAX_QPS 秒。"""

    def __init__(self, qps: float):
        self.interval = 1.0 / max(qps, 1e-6)
        self._lock = asyncio.Lock()
        self._next_time = 0.0

    async def __aenter__(self):
        async with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_time - now)
            if wait:
                await asyncio.sleep(wait)
            self._next_time = time.monotonic() + self.interval

    async def __aexit__(self, exc_type, exc, tb):
        return False


rate_limiter = AsyncRateLimiter(MAX_QPS)




def review_file_diff(file_diff) -> FileCRResult:
    if getattr(file_diff, "is_binary", True):
        return FileCRResult(
            file_path=file_diff.b_path or file_diff.a_path or "<unknown>",
            change_type=file_diff.change_type,
            summary="二进制文件，跳过自动代码审查，请人工确认。",
            overall_severity="info",
            approved=False,
            issues=[],
            needs_human_review=True,
            meta={"reason": "binary_file"},
        )
    return review_chain.invoke(file_diff)


async def review_file_diff_async(file_diff) -> FileCRResult:
    if getattr(file_diff, "is_binary", True):
        return FileCRResult(
            file_path=file_diff.b_path or file_diff.a_path or "<unknown>",
            change_type=file_diff.change_type,
            summary="二进制文件，跳过自动代码审查，请人工确认。",
            overall_severity="info",
            approved=False,
            issues=[],
            needs_human_review=True,
            meta={"reason": "binary_file"},
        )
    async with rate_limiter:
        return await review_chain.ainvoke(file_diff)


async def _gather_reviews(files: Iterable) -> List[FileCRResult]:
    tasks = [review_file_diff_async(fd) for fd in files]
    return await asyncio.gather(*tasks)


async def review_all_files(state: AgentState):
    """并行审查所有文件，并受限于 MAX_QPS 速率。"""
    commit_diff = state["commit_diff"]
    results = await _gather_reviews(commit_diff.files)
    return {"file_cr_result": list(results)}


def print_resule(state: AgentState):
    print(state["file_cr_result"])

def build_review_agent():
    """Construct and compile the LangGraph review agent."""
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
    #repo_root = "/Users/luo/baidu/bcc/nova-go"
    repo_root = str(Path(__file__).resolve().parent)
    return asyncio.run(review_agent.ainvoke({"repo_path": repo_root}))


if __name__ == "__main__":
    run_default_repo()


#  todo 处理截断问题
