from dataclasses import asdict
import json
import operator
import os
from typing import Any, List, Literal, Optional, Dict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from pydantic import ValidationError
from typing_extensions import Annotated, TypedDict

from tools.git_tools import CommitDiff, FileDiff, get_last_commit_diff
_ = load_dotenv()

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


llm = ChatOpenAI(
        base_url=BASE_URL,
        api_key=get_api_key,
        model=MODEL_NAME,
        temperature=0.7,
        timeout=60,
)



Severity = Literal["info", "minor", "major", "critical"]
Category = Literal[
    "bug", "security", "performance", "concurrency", "reliability",
    "api", "style", "test", "documentation", "build", "other"
]

class CRIssue(BaseModel):
    severity: Severity
    category: Category
    message: str = Field(..., min_length=3)
    file_path: Optional[str] = None
    line_start: Optional[int] = Field(default=None, ge=1)
    line_end: Optional[int] = Field(default=None, ge=1)
    suggestion: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

class FileCRResult(BaseModel):
    file_path: str
    change_type: str
    summary: str = Field(..., min_length=3)
    overall_severity: Severity
    approved: bool
    issues: List[CRIssue] = Field(default_factory=list)
    needs_human_review: bool = False
    meta: Dict[str, Any] = Field(default_factory=dict)



PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior code reviewer. "
     "Return a structured review following the schema exactly."),
    ("human",
     "Review this file diff.\n"
     "Rules:\n"
     "- If binary or patch is empty: needs_human_review=true and explain.\n"
     "- Prefer line numbers for the NEW version; if unsure, use null.\n"
     "- overall_severity reflects the most severe issue.\n\n"
     "INPUT(JSON):\n{payload_json}")
])

MAX_PATCH_CHARS = 12000

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

def review_file_diff(file_diff) -> FileCRResult:
    return review_chain.invoke(file_diff)


def review_commit_diff(state: AgentState):
    commit_diff = state["commit_diff"]
    result = review_file_diff(commit_diff.files[-2])
    return {
        "file_cr_result": [result],
    }
    
    
def print_commit_diff(state: AgentState) -> None:
    print(state["commit_diff"].files[-2])

def print_resule(state: AgentState):
    print(state["file_cr_result"])


class AgentState(TypedDict):
    repo_path: str
    commit_diff: CommitDiff
    file_cr_result: List[FileCRResult]

# Build workflow
agent_builder = StateGraph(AgentState)  

# Add nodes
agent_builder.add_node("get_last_commit_diff", get_last_commit_diff)
agent_builder.add_node("print_commit_diff", print_commit_diff) 
agent_builder.add_node("review_commit_diff", review_commit_diff)
agent_builder.add_node("print_resule", print_resule)

# Add edges
agent_builder.add_edge(START, "get_last_commit_diff")
agent_builder.add_edge("get_last_commit_diff", "print_commit_diff")
agent_builder.add_edge("print_commit_diff", "review_commit_diff")
agent_builder.add_edge("review_commit_diff", "print_resule")
agent_builder.add_edge("print_resule", END)

agent = agent_builder.compile()

messages = agent.invoke({"repo_path": "/Users/luo/baidu/personal-code/nova-cr-agent"})


