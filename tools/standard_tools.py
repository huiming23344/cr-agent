from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from cr_agent.rules import get_rules_catalog


@tool
def code_standard_doc(rule_id: str) -> str:
    """读取代码规范的 Markdown 文档，便于按规则审查。"""
    try:
        catalog = get_rules_catalog()
    except Exception as exc:
        return f"无法加载规则索引：{exc}"

    meta = catalog.by_id.get(rule_id) if catalog else None
    if not meta:
        return f"未找到规则 {rule_id}，请确认 rule_id 是否正确。"
    if not meta.doc_path:
        return f"规则 {rule_id} 未提供文档路径。"

    try:
        content = Path(meta.doc_path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"规则 {rule_id} 的文档不存在：{meta.doc_path}"
    except Exception as exc:
        return f"读取规则 {rule_id} 文档失败：{exc}"

    max_chars = 4000
    if len(content) > max_chars:
        return content[:max_chars] + "\n...<内容截断>..."
    return content
