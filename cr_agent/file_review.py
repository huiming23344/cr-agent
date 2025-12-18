from __future__ import annotations

import asyncio
import json
import operator
import re
import time
from pathlib import Path
from typing import Annotated, Iterable, List, Optional, TypedDict, cast

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import ValidationError

from cr_agent.models import (
    FileCRResult,
    FileDiff,
    FileTaggingLLMResult,
    FileTaggingResult,
    Tag,
    TagCRLLMResult,
    TagCRResult,
)
from cr_agent.rules import RULE_DOMAINS

__all__ = ["FileReviewEngine"]

BLACKLIST_PATTERNS: tuple[re.Pattern, ...] = tuple()

TAG_DESCRIPTIONS: dict[Tag, str] = {
    "STYLE": "风格/可读性（命名、结构、注释、可维护性）",
    "ERROR": "错误处理（边界、异常、返回值、降级、日志）",
    "API": "接口设计（对外契约、兼容性、参数与返回、文档）",
    "CONC": "并发（线程安全、竞态、锁、异步、资源释放）",
    "PERF": "性能（复杂度、IO、缓存、分配、热点）",
    "SEC": "安全（鉴权、输入校验、注入、敏感信息、权限）",
    "TEST": "测试（覆盖率、用例质量、回归、稳定性）",
    "CONFIG": "配置/依赖（配置项、环境变量、依赖版本、部署影响）",
}


@tool
def style_guideline_lookup(query: str) -> str:
    """查询风格/可读性规范或示例。"""
    return f"[STYLE指引] 暂未实现真实查找：请根据上下文自查。输入：{query}"


@tool
def error_case_library(query: str) -> str:
    """检索常见错误处理策略示例。"""
    return f"[ERROR案例] 暂未实现真实查找，请结合日志和代码确认。输入：{query}"


@tool
def api_contract_checker(query: str) -> str:
    """检查接口设计/兼容性注意事项。"""
    return f"[API契约] 暂未实现真实校验：请人工核对接口文档。输入：{query}"


@tool
def concurrency_pattern_helper(query: str) -> str:
    """提供并发/同步模式的参考建议。"""
    return f"[CONC建议] 暂未实现真实分析：请审核锁、goroutine、任务调度。输入：{query}"


@tool
def performance_budget_tool(query: str) -> str:
    """估算热点/复杂度/资源占用等性能风险。"""
    return f"[PERF预算] 暂未实现真实 profiling：请关注复杂度与缓存策略。输入：{query}"


@tool
def security_threat_scanner(query: str) -> str:
    """提示潜在安全威胁（鉴权、注入、敏感信息）。"""
    return f"[SEC扫描] 暂未实现真实检测：请人工检查敏感路径。输入：{query}"


@tool
def test_coverage_inspector(query: str) -> str:
    """审查测试覆盖、用例质量与回归风险。"""
    return f"[TEST覆盖] 暂未实现真实统计：请检查新增/受影响场景。输入：{query}"


@tool
def config_dependency_auditor(query: str) -> str:
    """检查配置/依赖/部署影响。"""
    return f"[CONFIG审计] 暂未实现真实审计：请确认依赖版本与环境变量。输入：{query}"


TAG_TOOLS: dict[Tag, List] = {
    "STYLE": [style_guideline_lookup],
    "ERROR": [error_case_library],
    "API": [api_contract_checker],
    "CONC": [concurrency_pattern_helper],
    "PERF": [performance_budget_tool],
    "SEC": [security_threat_scanner],
    "TEST": [test_coverage_inspector],
    "CONFIG": [config_dependency_auditor],
}

class _AsyncLimiterBase:
    async def __aenter__(self):
        ...

    async def __aexit__(self, exc_type, exc, tb):
        return False


class AsyncRateLimiter(_AsyncLimiterBase):
    """简单限速器：确保任意两次调用间隔 >= 1 / qps 秒。"""

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


class NoopRateLimiter(_AsyncLimiterBase):
    """No-op context manager used when no rate limit is configured."""


class FileReviewState(TypedDict):
    file_diff: FileDiff
    tags: List[Tag]
    tagging_reasoning: Optional[str]
    tag_results: Annotated[List[TagCRResult], operator.add]
    file_cr_result: Optional[FileCRResult]
    skip: bool


class FileReviewEngine:
    """负责单个文件的打标、标签路由和结果合并。"""

    def __init__(
        self,
        llm,
        *,
        max_patch_chars: int = 12_000,
        rate_limiter: Optional[_AsyncLimiterBase] = None,
        allowed_tags: Optional[tuple[Tag, ...]] = None,
        blacklist_patterns: Optional[tuple[re.Pattern, ...]] = None,
        blacklist_basenames: Optional[Iterable[str]] = None,
    ):
        self.llm = llm
        self.max_patch_chars = max_patch_chars
        self.rate_limiter = rate_limiter or NoopRateLimiter()
        self.enabled_tags: tuple[Tag, ...] = allowed_tags or cast(tuple[Tag, ...], RULE_DOMAINS)
        self.blacklist_patterns: tuple[re.Pattern, ...] = blacklist_patterns or ()
        self.blacklist_basenames = {name.strip() for name in (blacklist_basenames or []) if name and name.strip()}
        self.prepare = RunnableLambda(lambda fd, engine=self: {"payload_json": json.dumps(engine._prepare_payload(fd), ensure_ascii=False)})
        self.tagger_prompt = self._build_tagger_prompt()
        self.tagger_chain = self._build_tagger_chain()
        self.tag_agents = self._build_tag_agents()
        self.file_graph = self._build_file_review_graph()

    async def review_file(self, file_diff: FileDiff) -> FileCRResult:
        state = await self.file_graph.ainvoke(
            {
                "file_diff": file_diff,
                "tags": [],
                "tagging_reasoning": None,
                "tag_results": [],
                "file_cr_result": None,
                "skip": False,
            }
        )
        result = state.get("file_cr_result")
        if result is None:
            return self._skip_file_result(
                file_diff,
                reason="internal_error",
                summary="未能生成该文件的审查结果，请人工确认。",
            )
        return result

    # ------------------------------------------------------------------ #
    # 构建链路
    # ------------------------------------------------------------------ #

    def _build_tagger_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一名代码变更标签分类器。\n"
                    "请根据输入的单个文件 diff，为其分配 0~多个标签（可多选）：\n"
                    "- STYLE  风格/可读性\n"
                    "- ERROR  错误处理\n"
                    "- API    接口设计\n"
                    "- CONC   并发\n"
                    "- PERF   性能\n"
                    "- SEC    安全\n"
                    "- TEST   测试\n"
                    "- CONFIG 配置/依赖\n\n"
                    "要求：\n"
                    "- 只从上述标签中选择，不要输出其它标签。\n"
                    "- 如果 diff 体现了某个方面的变更，就打上对应标签；一个文件可以多个标签。\n"
                    "- 如果无法明确判断，也请尽量给出最相关的标签；一般代码变更至少包含 STYLE。\n"
                    "- 输出必须严格符合 FileTaggingLLMResult 结构化模式；所有说明使用中文。",
                ),
                (
                    "human",
                    "请对以下文件 diff 打标签。\n"
                    "输入(JSON)：\n{payload_json}",
                ),
            ]
        )

    def _build_tag_agent_prompt(self, tag: Tag) -> str:
        desc = TAG_DESCRIPTIONS[tag]
        tools = TAG_TOOLS.get(tag, [])
        tool_lines = "\n".join(
            f"- {tool.name}: {getattr(tool, 'description', '').strip() or '专项辅助工具'}" for tool in tools
        ) or "- （无可用工具）"
        return (
            f"你是一名资深代码审查专家，专注于 [{tag}] 方向：{desc}。\n"
            "使用 ReAct 策略（思考 -> 如需工具则调用 -> 根据工具结果总结）。\n"
            "可用工具：\n"
            f"{tool_lines}\n\n"
            "输出要求：\n"
            "- 只聚焦本标签相关的问题；无关内容忽略。\n"
            "- 最终输出必须符合 TagCRLLMResult 结构化模式，所有文字使用中文。\n"
            "- 在给出结构化输出前，如有必要可多次调用工具以收集信息。"
        )

    def _build_tagger_chain(self):
        return (self.prepare | self.tagger_prompt | self.llm.with_structured_output(FileTaggingLLMResult)).with_retry(
            stop_after_attempt=3,
            retry_if_exception_type=(ValidationError, ValueError),
        )

    def _build_tag_agents(self) -> dict[Tag, object]:
        agents: dict[Tag, object] = {}
        for tag in self.enabled_tags:
            prompt = self._build_tag_agent_prompt(tag)
            tools = TAG_TOOLS.get(tag, [])
            agents[tag] = create_react_agent(
                self.llm,
                tools=tools,
                prompt=prompt,
                response_format=TagCRLLMResult,
            )
        return agents

    def _build_file_review_graph(self):

        g = StateGraph(FileReviewState)
        g.add_node("guard_file", self._guard_file)
        g.add_node("tag_file", self._tag_file_node)
        g.add_node("maybe_finalize", self._maybe_finalize)

        for tag in self.enabled_tags:
            node_name = f"review_{tag}"
            g.add_node(node_name, self._make_tag_reviewer_node(tag))
            g.add_edge(node_name, "maybe_finalize")

        g.add_edge(START, "guard_file")
        g.add_conditional_edges("guard_file", self._route_after_guard, {"skip": END, "continue": "tag_file"})
        g.add_conditional_edges("tag_file", self._route_by_tags, {tag: f"review_{tag}" for tag in self.enabled_tags})
        g.add_edge("tag_file", "maybe_finalize")
        g.add_edge("maybe_finalize", END)
        return g.compile()

    # ------------------------------------------------------------------ #
    # LangGraph 节点
    # ------------------------------------------------------------------ #

    def _guard_file(self, state: FileReviewState):
        fd = state["file_diff"]
        if self._matches_blacklist(fd):
            return {
                "skip": True,
                "tags": [],
                "file_cr_result": self._skip_file_result(
                    fd,
                    reason="name_blacklist",
                    summary="文件名匹配黑名单，跳过自动代码审查，请人工确认。",
                ),
            }
        if getattr(fd, "is_binary", False):
            return {
                "skip": True,
                "tags": [],
                "file_cr_result": self._skip_file_result(
                    fd,
                    reason="binary_file",
                    summary="二进制文件，跳过自动审查，请人工确认。",
                ),
            }
        if not (fd.patch or "").strip():
            return {
                "skip": True,
                "tags": [],
                "file_cr_result": self._skip_file_result(
                    fd,
                    reason="empty_patch",
                    summary="补丁为空或不可解析，跳过自动审查，请人工确认。",
                ),
            }
        return {"skip": False}

    async def _tag_file_node(self, state: FileReviewState):
        tagging = await self._tag_file_diff(state["file_diff"])
        return {"tags": tagging.tags, "tagging_reasoning": tagging.reasoning}

    def _route_after_guard(self, state: FileReviewState):
        return "skip" if state.get("skip") else "continue"

    def _route_by_tags(self, state: FileReviewState):
        return [t for t in state.get("tags", []) if t in self.enabled_tags]

    def _make_tag_reviewer_node(self, tag: Tag):
        async def _node(state: FileReviewState):
            result = await self._review_tag(state["file_diff"], tag)
            return {"tag_results": [result]}

        return _node

    def _maybe_finalize(self, state: FileReviewState):
        if state.get("file_cr_result") is not None:
            return {}

        tags = state.get("tags", [])
        expected = len(tags)
        received = len(state.get("tag_results", []))
        if expected and received < expected:
            return {}

        return {
            "file_cr_result": self._merge_file_results(
                file_diff=state["file_diff"],
                tags=tags,
                tag_results=state.get("tag_results", []),
                tagging_reasoning=state.get("tagging_reasoning"),
            )
        }

    # ------------------------------------------------------------------ #
    # 具体动作
    # ------------------------------------------------------------------ #

    def _prepare_payload(self, file_diff: FileDiff) -> dict:
        patch = file_diff.patch or ""
        if len(patch) > self.max_patch_chars:
            patch = patch[: self.max_patch_chars] + "\n\n...<PATCH TRUNCATED>..."

        return {
            "file_path": self._file_path(file_diff),
            "change_type": file_diff.change_type,
            "is_binary": file_diff.is_binary,
            "added_lines": file_diff.added_lines,
            "deleted_lines": file_diff.deleted_lines,
            "patch": patch,
            "rename_from": file_diff.rename_from,
            "rename_to": file_diff.rename_to,
        }

    async def _tag_file_diff(self, file_diff: FileDiff) -> FileTaggingResult:
        async with self.rate_limiter:
            llm_result: FileTaggingLLMResult = await self.tagger_chain.ainvoke(file_diff)

        tags = self._normalize_tags(llm_result.tags)
        if (file_diff.patch or "").strip() and not tags:
            tags = ["STYLE"]
        tags = self._filter_enabled_tags(tags)

        return FileTaggingResult(file_path=self._file_path(file_diff), tags=tags, reasoning=llm_result.reasoning)

    async def _review_tag(self, file_diff: FileDiff, tag: Tag) -> TagCRResult:
        payload_json = json.dumps(self._prepare_payload(file_diff), ensure_ascii=False)
        user_message = (
            "请针对下列文件 diff 执行专项代码审查，并仅关注本标签相关的问题。\n"
            "需要时可以调用可用工具。\n"
            "输入(JSON)：\n"
            f"{payload_json}"
        )
        agent = self.tag_agents[tag]
        async with self.rate_limiter:
            agent_state = await agent.ainvoke({"messages": [{"role": "user", "content": user_message}]})

        structured = agent_state.get("structured_response")
        if structured is None:
            raise ValueError(f"Tag agent for {tag} 未返回结构化结果")
        return TagCRResult(file_path=self._file_path(file_diff), tag=tag, **structured.model_dump())

    def _merge_file_results(
        self,
        *,
        file_diff: FileDiff,
        tags: List[Tag],
        tag_results: List[TagCRResult],
        tagging_reasoning: Optional[str],
    ) -> FileCRResult:
        issues = []
        severities: List[str] = []
        needs_human_review = False
        approved = True
        per_tag_summary: List[str] = []

        for tr in tag_results:
            per_tag_summary.append(f"[{tr.tag}] {tr.summary}")
            issues.extend(tr.issues)
            severities.append(tr.overall_severity)
            needs_human_review = needs_human_review or bool(tr.needs_human_review)
            approved = approved and bool(tr.approved)

        overall_severity = self._max_severity(severities) if severities else "info"
        approved = approved and (not needs_human_review)
        tags_str = ", ".join(tags) if tags else "无"

        summary = "；".join(per_tag_summary) if per_tag_summary else f"标签={tags_str}：未发现需要专项审查的问题。"
        meta = {
            "tags": tags,
            "tagging_reasoning": tagging_reasoning,
            "per_tag": [tr.model_dump() for tr in tag_results],
        }

        return FileCRResult(
            file_path=self._file_path(file_diff),
            change_type=file_diff.change_type,
            summary=summary,
            overall_severity=overall_severity,  # type: ignore[arg-type]
            approved=approved,
            issues=issues,
            needs_human_review=needs_human_review,
            meta=meta,
        )

    # ------------------------------------------------------------------ #
    # 工具方法
    # ------------------------------------------------------------------ #

    @staticmethod
    def _file_path(file_diff: FileDiff) -> str:
        return file_diff.b_path or file_diff.a_path or "<unknown>"

    def _matches_blacklist(self, file_diff: FileDiff) -> bool:
        path = FileReviewEngine._file_path(file_diff)
        basename = Path(path).name
        if basename in self.blacklist_basenames:
            return True
        return any(pattern.search(path) for pattern in self.blacklist_patterns)

    @staticmethod
    def _normalize_tags(tags: Iterable[Tag]) -> List[Tag]:
        seen: set[Tag] = set()
        out: List[Tag] = []
        for tag in tags:
            if tag in seen:
                continue
            seen.add(tag)
            out.append(tag)
        return out

    def _filter_enabled_tags(self, tags: Iterable[Tag]) -> List[Tag]:
        return [tag for tag in tags if tag in self.enabled_tags]

    @staticmethod
    def _severity_rank(severity: str) -> int:
        order = {"info": 0, "minor": 1, "major": 2, "critical": 3}
        return order.get(str(severity), 0)

    def _max_severity(self, severities: Iterable[str]) -> str:
        best = "info"
        best_rank = -1
        for sev in severities:
            r = self._severity_rank(sev)
            if r > best_rank:
                best_rank = r
                best = str(sev)
        return best

    def _skip_file_result(self, file_diff: FileDiff, *, reason: str, summary: str) -> FileCRResult:
        return FileCRResult(
            file_path=self._file_path(file_diff),
            change_type=file_diff.change_type,
            summary=summary,
            overall_severity="info",
            approved=False,
            issues=[],
            needs_human_review=True,
            meta={"reason": reason},
        )
