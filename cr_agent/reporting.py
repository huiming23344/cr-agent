from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from cr_agent.models import CRIssue, CommitDiff, FileCRResult
from cr_agent.rules import get_rules_catalog


def render_markdown_report(
    *,
    repo_path: str,
    commit_diff: CommitDiff,
    file_results: Iterable[FileCRResult],
) -> str:
    renderer = _MarkdownReportRenderer(repo_path=Path(repo_path), commit_diff=commit_diff)
    return renderer.render(list(file_results))


def write_markdown_report(
    *,
    repo_path: str,
    commit_diff: CommitDiff,
    file_results: Iterable[FileCRResult],
    custom_dir: Optional[str] = None,
    report_text: Optional[str] = None,
) -> Path:
    report_md = report_text or render_markdown_report(repo_path=repo_path, commit_diff=commit_diff, file_results=file_results)
    target_dir = Path(custom_dir).expanduser().resolve() if custom_dir else Path(repo_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    report_path = target_dir / "code_review_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    return report_path


def summarize_to_cli(*, commit_diff: CommitDiff, file_results: Iterable[FileCRResult], report_path: Optional[Path] = None) -> None:
    files_count = len(list(file_results))
    title = (commit_diff.message or "").strip().splitlines()[0] if commit_diff else ""
    print(f"[CR] {title or '变更'} | 文件 {files_count} | 报告: {report_path or '未写入'}")


@dataclass
class _MarkdownReportRenderer:
    repo_path: Path
    commit_diff: CommitDiff
    _file_cache: Dict[str, Optional[List[str]]] = field(default_factory=dict)

    def render(self, results: List[FileCRResult]) -> str:
        overview = self._render_overview(results)
        rule_issues_md, general_issues_md = self._render_issues(results)

        parts = [
            "# Code Review Report",
            "## 概述",
            overview,
            "## Rule Issues",
            rule_issues_md or "_无带 rule_id 的问题。_",
            "## General Issues",
            general_issues_md or "_无未关联 rule 的问题。_",
        ]
        return "\n\n".join(part for part in parts if part is not None)

    def _render_overview(self, results: List[FileCRResult]) -> str:
        files_count = len(results)
        approvals = sum(1 for r in results if r.approved)
        needs_review = sum(1 for r in results if r.needs_human_review)
        commit_title = (self.commit_diff.message or "").strip().splitlines()[0] if self.commit_diff else ""

        lines = [
            f"- 变更摘要：{commit_title or '（无提交信息）'}",
            f"- 文件数：{files_count}（通过 {approvals}，需人工 {needs_review}）",
        ]
        return "\n".join(lines)

    def _render_issues(self, results: List[FileCRResult]) -> Tuple[str, str]:
        with_rule: List[str] = []
        general: List[str] = []

        for fr in results:
            for issue in fr.issues:
                rule_id = self._extract_rule_id(issue, fr)
                block = self._render_issue_block(issue, fr, rule_id=rule_id)
                if rule_id:
                    with_rule.append(block)
                else:
                    general.append(block)

        return "\n\n".join(with_rule), "\n\n".join(general)

    def _extract_rule_id(self, issue: CRIssue, file_result: FileCRResult) -> Optional[str]:
        extras = getattr(issue, "model_extra", {}) or {}
        if "rule_id" in extras:
            value = extras["rule_id"]
            if value:
                return str(value)

        per_tag = file_result.meta.get("per_tag") or []
        for entry in per_tag:
            if isinstance(entry, dict) and "meta" in entry and isinstance(entry["meta"], dict):
                rid = entry["meta"].get("rule_id")
                if rid:
                    return str(rid)
        return None

    def _render_issue_block(self, issue: CRIssue, fr: FileCRResult, *, rule_id: Optional[str]) -> str:
        path_line = self._format_path(issue, fr)
        rule_title = self._lookup_rule_title(rule_id)
        rule_hint = self._lookup_rule_hint(rule_id)
        header = f"### [{rule_id}] {rule_title}" if rule_id else f"### {path_line}"

        lines = [header]
        lines.append(f"- 说明：{issue.message}")
        if issue.suggestion:
            lines.append(f"- 建议：{issue.suggestion}")
        if rule_id:
            lines.append(f"- 规则说明：{rule_hint}")
        if path_line:
            lines.append(f"- 位置：{path_line}")
        lines.append(f"- 严重级别：{issue.severity}")

        code_block = self._render_code_context(issue, fr)
        if code_block:
            lines.append("```")
            lines.append(code_block)
            lines.append("```")

        return "\n".join(lines)

    def _format_path(self, issue: CRIssue, fr: FileCRResult) -> str:
        path = issue.file_path or fr.file_path or "<unknown>"
        if issue.line_start and issue.line_end:
            return f"{path}:{issue.line_start}-{issue.line_end}"
        if issue.line_start:
            return f"{path}:{issue.line_start}"
        return path

    def _render_code_context(self, issue: CRIssue, fr: FileCRResult) -> Optional[str]:
        path = issue.file_path or fr.file_path
        if not path:
            return None

        line_start = issue.line_start or 1
        line_end = issue.line_end or line_start

        lines = self._load_file_lines(path)
        if lines:
            start = max(1, line_start - 3)
            end = min(len(lines), line_end + 3)
            snippet = lines[start - 1 : end]
            numbered = [f"{start + idx:>4} {text.rstrip()}" for idx, text in enumerate(snippet)]
            return "\n".join(numbered)

        return f"{path}:{line_start}-{line_end}"

    def _load_file_lines(self, path: str) -> Optional[List[str]]:
        if path in self._file_cache:
            return self._file_cache[path]
        full_path = self.repo_path / path
        try:
            content = full_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            content = None
        self._file_cache[path] = content
        return content

    @staticmethod
    def _lookup_rule_title(rule_id: Optional[str]) -> str:
        if not rule_id:
            return "未关联规则"
        try:
            meta = get_rules_catalog().by_id.get(rule_id)
        except Exception:
            meta = None
        return meta.title if meta and meta.title else rule_id

    @staticmethod
    def _lookup_rule_hint(rule_id: Optional[str]) -> str:
        if not rule_id:
            return "无规则说明"
        try:
            meta = get_rules_catalog().by_id.get(rule_id)
        except Exception:
            meta = None
        if meta:
            parts = [meta.prompt_hint or "", meta.raw.get("description", "")]
            text = "；".join(p.strip() for p in parts if p and str(p).strip())
            return text or "无规则说明"
        return "无规则说明"
