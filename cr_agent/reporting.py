from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from cr_agent.models import CommitDiff, CRIssue, FileCRResult

REPORTS_DIR_NAME = "cr_reports"


def ensure_report_directory(repo_path: str | Path, custom_dir: Optional[str] = None) -> Path:
    """Return the directory for persisted reports, creating it if needed."""
    base = Path(custom_dir).expanduser() if custom_dir else Path(repo_path)
    report_dir = base if custom_dir else base / REPORTS_DIR_NAME
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir


def _format_commit_headline(commit_diff: Optional[CommitDiff]) -> str:
    if not commit_diff:
        return "unknown"
    short_sha = commit_diff.commit_sha[:7] if commit_diff.commit_sha else "unknown"
    subject = (commit_diff.message or "").splitlines()[0].strip()
    return f"{short_sha} - {subject}" if subject else short_sha


def _collect_rule_issues(file_results: Sequence[FileCRResult]) -> List[Tuple[FileCRResult, CRIssue]]:
    collected: List[Tuple[FileCRResult, CRIssue]] = []
    for file_result in file_results:
        for issue in file_result.issues:
            if any(rule_id for rule_id in issue.rule_ids):
                collected.append((file_result, issue))
    return collected


def summarize_to_cli(*, commit_diff: Optional[CommitDiff], file_results: Sequence[FileCRResult], report_path: Path) -> None:
    """Print a concise CLI overview of the current review run."""
    total_files = len(file_results)
    total_issues = sum(len(fr.issues) for fr in file_results)
    rule_issues = sum(1 for fr in file_results for issue in fr.issues if any(issue.rule_ids))
    changed_files = len(commit_diff.files) if commit_diff else total_files
    commit_line = _format_commit_headline(commit_diff)
    message = (commit_diff.message or "").strip() if commit_diff else ""
    lead = message.splitlines()[0] if message else "No commit message."

    interesting_files = [fr.file_path for fr in file_results if fr.issues][:3]
    focus_hint = ", ".join(interesting_files) if interesting_files else "无突出问题文件"

    print("=== 代码变更 CR 概览 ===")
    print(f"提交：{commit_line}")
    print(f"主要内容：{lead}")
    print(f"涉及文件：{changed_files} 个（审查 {total_files} 个文件）")
    print(f"发现问题：{total_issues} 条，其中带 rule_id 的 {rule_issues} 条")
    print(f"重点关注文件：{focus_hint}")
    print(f"详细报告：{report_path}")


def _format_file_issues(file_result: FileCRResult) -> List[str]:
    lines: List[str] = []
    if not file_result.issues:
        lines.append("无问题。")
        return lines

    header = "| Severity | Category | Message | Rule IDs | Location | Suggestion |"
    separator = "| --- | --- | --- | --- | --- | --- |"
    lines.extend([header, separator])

    for issue in file_result.issues:
        rule_text = ", ".join(issue.rule_ids) if issue.rule_ids else "-"
        location = "-"
        if issue.line_start:
            end = f"-{issue.line_end}" if issue.line_end and issue.line_end != issue.line_start else ""
            location = f"L{issue.line_start}{end}"
        suggestion = issue.suggestion.replace("\n", " ") if issue.suggestion else "-"
        message = issue.message.replace("\n", " ")
        lines.append(f"| {issue.severity} | {issue.category} | {message} | {rule_text} | {location} | {suggestion} |")
    return lines


def _format_rule_issue_section(rule_issues: Iterable[Tuple[FileCRResult, CRIssue]]) -> List[str]:
    lines: List[str] = []
    issues = list(rule_issues)
    if not issues:
        lines.append("无带 rule_id 的问题。")
        return lines

    header = "| File | Severity | Rule IDs | Message | Lines |"
    separator = "| --- | --- | --- | --- | --- |"
    lines.extend([header, separator])

    for file_result, issue in issues:
        rule_text = ", ".join(issue.rule_ids)
        message = issue.message.replace("\n", " ")
        if issue.line_start:
            end = f"-{issue.line_end}" if issue.line_end and issue.line_end != issue.line_start else ""
            lines_str = f"L{issue.line_start}{end}"
        else:
            lines_str = "-"
        lines.append(f"| `{file_result.file_path}` | {issue.severity} | {rule_text} | {message} | {lines_str} |")
    return lines


def build_markdown_report(
    *,
    repo_path: str,
    commit_diff: Optional[CommitDiff],
    file_results: Sequence[FileCRResult],
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_label = _format_commit_headline(commit_diff)
    total_files = len(file_results)
    total_issues = sum(len(fr.issues) for fr in file_results)
    rule_issue_entries = _collect_rule_issues(file_results)
    rule_issue_count = len(rule_issue_entries)
    changed_files = len(commit_diff.files) if commit_diff else total_files
    summary_line = (commit_diff.message or "").strip().splitlines()[0] if commit_diff and commit_diff.message else "No commit message."

    lines: List[str] = [
        f"# Code Review Report",
        "",
        f"- 生成时间：{now}",
        f"- 仓库：`{repo_path}`",
        f"- 提交：{commit_label}",
        f"- 主要内容：{summary_line}",
        f"- 涉及文件：{changed_files} 个（审查 {total_files} 个文件）",
        f"- 问题总数：{total_issues}（其中带 rule_id 的 {rule_issue_count}）",
        "",
        "## 文件级结论",
    ]

    for file_result in file_results:
        tags = ", ".join(file_result.meta.get("tags", [])) if isinstance(file_result.meta, dict) else ""
        tag_text = f"标签：{tags}" if tags else "标签：无"
        rule_text = ", ".join(file_result.rule_ids) if file_result.rule_ids else "无"
        lines.extend(
            [
                f"### `{file_result.file_path}`",
                f"- change_type: {file_result.change_type}",
                f"- overall_severity: {file_result.overall_severity}",
                f"- approved: {file_result.approved}",
                f"- rule_ids: {rule_text}",
                f"- {tag_text}",
                f"- summary: {file_result.summary}",
                "",
            ]
        )
        lines.extend(_format_file_issues(file_result))
        lines.append("")

    lines.extend(
        [
            "## 带 rule_id 的问题汇总",
            "",
        ]
    )
    lines.extend(_format_rule_issue_section(rule_issue_entries))
    lines.append("")
    return "\n".join(lines)


def write_markdown_report(
    *,
    repo_path: str,
    commit_diff: Optional[CommitDiff],
    file_results: Sequence[FileCRResult],
    custom_dir: Optional[str] = None,
) -> Path:
    report_dir = ensure_report_directory(repo_path, custom_dir=custom_dir)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    commit_part = commit_diff.commit_sha[:7] if commit_diff and commit_diff.commit_sha else "no-commit"
    filename = f"cr-report-{timestamp}-{commit_part}.md"
    content = build_markdown_report(repo_path=repo_path, commit_diff=commit_diff, file_results=file_results)
    report_path = report_dir / filename
    report_path.write_text(content, encoding="utf-8")
    return report_path
