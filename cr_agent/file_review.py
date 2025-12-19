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
from pydantic import ValidationError
from cr_agent.agents import ReactDomainAgent, StaticPromptBuilder
from cr_agent.rate_limiter import AsyncRateLimiter, NoopRateLimiter, RateLimiterProtocol

from cr_agent.models import (
    FileCRResult,
    FileDiff,
    FileTaggingLLMResult,
    FileTaggingResult,
    Tag,
    TagCRLLMResult,
    TagCRResult,
)
from cr_agent.rules import RULE_DOMAINS, RuleMeta, get_rules_catalog
from tools.standard_tools import code_standard_doc

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


TAG_TOOLS: dict[Tag, List] = {
    "STYLE": [],
    "ERROR": [],
    "API": [],
    "CONC": [],
    "PERF": [],
    "SEC": [],
    "TEST": [],
    "CONFIG": [],
}


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
        rate_limiter: Optional[RateLimiterProtocol] = None,
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
        tools = self._tools_for_tag(tag)
        tool_lines = "\n".join(
            f"- {tool.name}: {getattr(tool, 'description', '').strip() or '专项辅助工具'}" for tool in tools
        ) or "- （无可用工具）"
        return (
            f"你是一名资深代码审查专家（专精领域：[{tag}]：{desc}），你的目标是**高检出率（Recall 优先）**地发现与该领域相关、且违反团队既有规范的改动。\n"
            "你的工作模式：**Rule-first** —— 以输入的 standards（规则清单与摘要）为唯一主要依据；仅在必要时通过工具读取对应 Markdown 规则原文来核实细节。\n"
            "\n"
            "审查输入：\n"
            "- 一个文件的 diff/代码上下文\n"
            "- 一组与该标签相关的 standards（含 rule_id、规则摘要、可能包含 doc_path）\n"
            "\n"
            "可用工具（仅用于查证规则原文或补充必要上下文）：\n"
            f"{tool_lines}\n"
            "\n"
            "关键原则（务必遵守）：\n"
            "1) **只审查与[{tag}]相关的问题**；与本领域无关的内容一律忽略。\n"
            "2) **以规范为准**：\n"
            "   - 你提出的每一条 issue，必须能对应到 standards 中的某条规则；如规则摘要不足以支撑结论，先调用 code_standard_doc(rule_id) 阅读原文再下结论。\n"
            "   - 若确实找不到对应规则，则默认不输出该问题。\n"
            "3) **对“建议性意见”强约束**：\n"
            "   - 对于**未在规范中明确要求**的改进点（如个人偏好、风格争议、可选重构），不要提出。\n"
            "   - 仅当满足以下任一条件时，才允许输出“非规范但应提示”的问题，并在 issue 中明确标注为 \"advisory\"：\n"
            "     a) 可能导致严重故障/数据丢失/安全漏洞/权限绕过/并发死锁等高风险；或\n"
            "     b) 属于工程领域普遍共识的硬性问题（例如明显的注入、明文凭据、竞态导致的崩溃、panic/exception 未处理导致服务不可用）。\n"
            "4) **Recall 优先策略**：\n"
            "   - 对任何“疑似违反规则”的点，先倾向于收集证据（必要时读规则原文），不要因为不确定就跳过。\n"
            "   - 但不要编造规则或臆测需求；不确定且无规则支撑时，不输出。\n"
            "5) **证据驱动**：每条 issue 必须包含可定位的证据（具体代码片段/行号范围/函数名/变更段），并解释为何违反规则。\n"
            "\n"
            "执行步骤（建议遵循，但不必输出你的思考过程）：\n"
            "A. 快速浏览 diff，列出所有可能与[{tag}]相关的风险点（宁可多列候选）。\n"
            "B. 将候选点逐一映射到 standards 的 rule_id；若摘要不足以判断，调用 code_standard_doc(rule_id) 查证。\n"
            "C. 对每个确认问题输出一条结构化 issue：\n"
            "   - type: \"violation\"（规范违规）或 \"advisory\"（仅限严重/共识问题）\n"
            "   - rule_ids: [对应规则]；若为 advisory 且无规则支撑，rule_ids 必须是 []\n"
            "   - severity: 按规范或你的风险判断（advisory 必须说明风险）\n"
            "   - evidence: 可定位的代码证据\n"
            "   - explanation: 简洁说明违反点与影响\n"
            "   - fix: 给出最小化、可执行的修复建议（不要大重构）\n"
            "\n"
            "输出要求（必须严格满足）：\n"
            "- 最终只输出一次，且必须符合 TagCRLLMResult 结构化模式。\n"
            "- 所有文字使用中文。\n"
            "- issue.rule_ids 必须准确列出对应 rule_id；若未引用任何规范则输出 []。\n"
            "- 在给出结构化输出前，如需要可多次调用工具收集规则原文信息。\n"
        )

    def _build_tagger_chain(self):
        return (self.prepare | self.tagger_prompt | self.llm.with_structured_output(FileTaggingLLMResult)).with_retry(
            stop_after_attempt=3,
            retry_if_exception_type=(ValidationError, ValueError),
        )

    def _build_tag_agents(self) -> dict[Tag, ReactDomainAgent]:
        agents: dict[Tag, ReactDomainAgent] = {}
        for tag in self.enabled_tags:
            prompt_builder = StaticPromptBuilder(self._build_tag_agent_prompt(tag))
            tools = self._tools_for_tag(tag)
            agents[tag] = ReactDomainAgent(
                llm=self.llm,
                prompt_builder=prompt_builder,
                tools=tools,
                response_format=TagCRLLMResult,
                name=f"tag-{tag}",
                rate_limiter=self.rate_limiter,
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
        language = self._infer_language(file_diff)
        standards = self._get_rules_for(tag=tag, language=language)
        standards_text = self._format_rules_for_prompt(standards)

        payload_json = json.dumps(self._prepare_payload(file_diff), ensure_ascii=False)
        user_message = (
            "请针对下列文件 diff 执行专项代码审查，并仅关注本标签相关的问题。\n"
            f"适用代码规范（language={language or 'unknown'}, domain={tag}）：\n{standards_text}\n"
            "需要时可以调用可用工具（若规范提供文档，可用 code_standard_doc(rule_id) 查看细节）。\n"
            "输入(JSON)：\n"
            f"{payload_json}"
        )
        agent = self.tag_agents[tag]
        agent_state = await agent.ainvoke({"messages": [{"role": "user", "content": user_message}]})

        structured = agent_state.get("structured_response")
        if structured is None:
            raise ValueError(f"Tag agent for {tag} 未返回结构化结果")
        rule_ids = sorted(
            {
                rid
                for issue in structured.issues
                for rid in getattr(issue, "rule_ids", []) or []
                if rid
            }
        )
        return TagCRResult(
            file_path=self._file_path(file_diff),
            tag=tag,
            summary=structured.summary,
            overall_severity=structured.overall_severity,
            approved=structured.approved,
            issues=structured.issues,
            needs_human_review=structured.needs_human_review,
            rule_ids=rule_ids,
            meta=structured.meta,
        )

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
        rule_ids_set: set[str] = set()

        for tr in tag_results:
            per_tag_summary.append(f"[{tr.tag}] {tr.summary}")
            issues.extend(tr.issues)
            severities.append(tr.overall_severity)
            needs_human_review = needs_human_review or bool(tr.needs_human_review)
            approved = approved and bool(tr.approved)
            for issue in tr.issues:
                for rid in issue.rule_ids:
                    if rid:
                        rule_ids_set.add(rid)

        overall_severity = self._max_severity(severities) if severities else "info"
        approved = approved and (not needs_human_review)
        tags_str = ", ".join(tags) if tags else "无"
        file_rule_ids = sorted(rule_ids_set)

        summary = "；".join(per_tag_summary) if per_tag_summary else f"标签={tags_str}：未发现需要专项审查的问题。"
        meta = {
            "tags": tags,
            "tagging_reasoning": tagging_reasoning,
            "per_tag": [tr.model_dump() for tr in tag_results],
            "rule_ids": file_rule_ids,
        }

        return FileCRResult(
            file_path=self._file_path(file_diff),
            change_type=file_diff.change_type,
            summary=summary,
            overall_severity=overall_severity,  # type: ignore[arg-type]
            approved=approved,
            issues=issues,
            rule_ids=file_rule_ids,
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

    def _tools_for_tag(self, tag: Tag) -> List:
        """Return tools for a tag, always including code_standard_doc."""
        base = list(TAG_TOOLS.get(tag, []))
        if code_standard_doc not in base:
            base.append(code_standard_doc)
        return base

    def _infer_language(self, file_diff: FileDiff) -> Optional[str]:
        path = self._file_path(file_diff).lower()
        suffix = Path(path).suffix
        if suffix == ".go":
            return "go"
        if suffix == ".py":
            return "python"
        return None

    def _get_rules_for(self, *, tag: Tag, language: Optional[str]) -> List[RuleMeta]:
        try:
            catalog = get_rules_catalog()
        except Exception:
            return []

        if language:
            by_lang_domain = catalog.by_language_domain.get(language, {})
            rules = by_lang_domain.get(tag, [])
            if rules:
                return list(rules)

        # Fallback to cross-language domain list
        return list(catalog.by_domain.get(tag, []))

    @staticmethod
    def _format_rules_for_prompt(rules: List[RuleMeta]) -> str:
        if not rules:
            return "- 无匹配规范（按通用审查逻辑处理）"

        lines: List[str] = []
        for meta in rules:
            details: List[str] = []
            if meta.severity:
                details.append(f"severity={meta.severity}")
            if meta.prompt_hint:
                details.append(meta.prompt_hint)
            if meta.doc_path:
                details.append("可用 code_standard_doc(rule_id) 查看文档")

            detail_str = "；".join(details)
            title = meta.title or "未命名规则"
            if detail_str:
                lines.append(f"- {meta.rule_id}: {title}｜{detail_str}")
            else:
                lines.append(f"- {meta.rule_id}: {title}")

        return "\n".join(lines)

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
