from __future__ import annotations

import fnmatch
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


class RulesConfigError(ValueError):
    pass

RULE_DOMAINS: Tuple[str, ...] = ("STYLE", "ERROR", "API", "CONC", "PERF", "SEC", "TEST", "CONFIG")
SUPPORTED_LANGUAGES: Tuple[str, ...] = ("go", "python")


@dataclass(frozen=True)
class RuleMeta:
    rule_id: str
    title: str = ""
    language: str = ""
    severity: Optional[str] = None
    domains: Tuple[str, ...] = tuple()
    prompt_hint: Optional[str] = None
    deprecated: bool = False
    doc_path: Optional[Path] = None
    raw: Dict[str, Any] = field(default_factory=dict)


RuleIndex = Dict[str, RuleMeta]


@dataclass(frozen=True)
class RulesCatalog:
    by_id: RuleIndex
    by_language: Dict[str, List[RuleMeta]]
    by_domain: Dict[str, List[RuleMeta]]
    by_language_domain: Dict[str, Dict[str, List[RuleMeta]]]


def load_rules_catalog(*, registry_path: Path) -> RulesCatalog:
    """Load rule metadata from registry.yaml without profile filtering."""
    registry_path = Path(registry_path).expanduser().resolve()
    registry = _load_yaml(registry_path)

    standards_dir = registry_path.parent
    repo_root = standards_dir.parent

    rules_index = _parse_registry_rules(registry, standards_dir=standards_dir, repo_root=repo_root)

    by_language = _aggregate_by_language(rules_index)
    by_domain = _aggregate_by_domain(rules_index)
    by_language_domain = _aggregate_by_language_domain(rules_index)
    return RulesCatalog(
        by_id=rules_index,
        by_language=by_language,
        by_domain=by_domain,
        by_language_domain=by_language_domain,
    )


def load_rules_index(*, registry_path: Path) -> RuleIndex:
    """Backwards compatible helper returning only the id->RuleMeta mapping."""
    return load_rules_catalog(registry_path=registry_path).by_id


def _aggregate_by_language(rules_index: RuleIndex) -> Dict[str, List[RuleMeta]]:
    grouped: Dict[str, List[RuleMeta]] = defaultdict(list)
    for meta in sorted(rules_index.values(), key=lambda m: (m.language or "", m.rule_id)):
        if meta.deprecated:
            continue
        if not meta.language:
            continue
        grouped[meta.language].append(meta)
    return {language: grouped[language] for language in sorted(grouped)}


def _aggregate_by_domain(rules_index: RuleIndex) -> Dict[str, List[RuleMeta]]:
    grouped: Dict[str, List[RuleMeta]] = defaultdict(list)
    for meta in sorted(rules_index.values(), key=lambda m: m.rule_id):
        if meta.deprecated:
            continue
        for domain in meta.domains:
            grouped[domain].append(meta)
    return {domain: grouped[domain] for domain in sorted(grouped)}


def _aggregate_by_language_domain(rules_index: RuleIndex) -> Dict[str, Dict[str, List[RuleMeta]]]:
    grouped: Dict[str, Dict[str, List[RuleMeta]]] = defaultdict(lambda: defaultdict(list))
    for meta in sorted(rules_index.values(), key=lambda m: (m.language or "", m.rule_id)):
        if meta.deprecated:
            continue
        if not meta.language:
            continue
        for domain in meta.domains:
            grouped[meta.language][domain].append(meta)
    return {
        language: {domain: grouped[language][domain] for domain in sorted(grouped[language])}
        for language in sorted(grouped)
    }


def _parse_registry_rules(
    registry: Any,
    *,
    standards_dir: Path,
    repo_root: Path,
) -> Dict[str, RuleMeta]:
    if not isinstance(registry, dict):
        raise RulesConfigError("registry.yaml 必须是 YAML mapping（dict）")

    rules_node = registry.get("rules")
    if rules_node is None:
        raise RulesConfigError("registry.yaml 缺少 rules")

    rules: List[Dict[str, Any]] = []
    if isinstance(rules_node, list):
        rules = [r for r in rules_node if isinstance(r, dict)]
    elif isinstance(rules_node, dict):
        for language, items in rules_node.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict):
                    merged = dict(item)
                    merged.setdefault("language", str(language))
                    rules.append(merged)

    index: Dict[str, RuleMeta] = {}
    for item in rules:
        rule_id = item.get("id") or item.get("rule_id")
        if not rule_id:
            continue
        rule_id = str(rule_id)

        language = str(item.get("language") or _infer_language(rule_id) or "")
        if language and language not in SUPPORTED_LANGUAGES:
            raise RulesConfigError(f"{rule_id}: 不支持的 language='{language}'，允许 {SUPPORTED_LANGUAGES}")
        title = str(item.get("title") or "")
        severity = str(item.get("severity")) if item.get("severity") is not None else None
        domains = _normalize_domains(item.get("domains"), fallback=item.get("domain"))
        prompt_hint = str(item.get("prompt_hint")) if item.get("prompt_hint") is not None else None
        deprecated = bool(item.get("deprecated", False))

        doc_path = None
        path_value = item.get("path") or item.get("doc")
        if path_value:
            doc_path = _resolve_rule_path(str(path_value), standards_dir=standards_dir, repo_root=repo_root)

        if rule_id in index:
            raise RulesConfigError(f"registry.yaml 存在重复规则 id: {rule_id}")

        index[rule_id] = RuleMeta(
            rule_id=rule_id,
            title=title,
            language=language,
            severity=severity,
            domains=domains,
            prompt_hint=prompt_hint,
            deprecated=deprecated,
            doc_path=doc_path,
            raw=dict(item),
        )

    return index


def _normalize_domains(domains_value: Any, fallback: Any = None) -> Tuple[str, ...]:
    values: List[str] = []
    raw_values: List[Any] = []

    if isinstance(domains_value, (list, tuple, set)):
        raw_values.extend(domains_value)
    elif domains_value is not None:
        raw_values.append(domains_value)
    elif fallback is not None:
        raw_values.append(fallback)

    for value in raw_values:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        if text not in RULE_DOMAINS:
            raise RulesConfigError(f"domain '{text}' 不受支持，必须属于 {RULE_DOMAINS}")
        if text not in values:
            values.append(text)
    return tuple(values)


def _infer_language(rule_id: str) -> Optional[str]:
    if rule_id.startswith("GO-"):
        return "go"
    if rule_id.startswith("PY-"):
        return "python"
    return None


def _resolve_rule_path(path_value: str, *, standards_dir: Path, repo_root: Path) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p

    if path_value.replace("\\", "/").startswith("coding-standards/"):
        candidate = (repo_root / p).resolve()
        return candidate

    candidate = (standards_dir / p).resolve()
    return candidate


def _load_yaml(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except ModuleNotFoundError:
        return _load_yaml_minimal(text, source=str(path))


def _load_yaml_minimal(text: str, *, source: str = "<string>") -> Any:
    """A minimal YAML loader (no external deps) for the coding-standards files.

    Supported subset:
    - mappings, lists, nested blocks via indentation
    - scalars: strings, ints, floats, booleans, null
    - inline JSON-like lists: ["a", "b"]
    - block scalars: `>` and `|`
    """
    lines = text.splitlines()
    tokens: List[Tuple[int, str]] = []
    for raw in lines:
        stripped = raw.rstrip("\n")
        leading = stripped[: len(stripped) - len(stripped.lstrip())]
        if "\t" in leading:
            raise RulesConfigError(f"{source}: YAML 缩进不支持 tab")
        indent = len(leading)
        if not stripped.strip():
            tokens.append((indent, ""))  # preserve blank lines for block scalars
            continue
        content = _strip_comment(stripped[indent:])
        if not content.strip():
            tokens.append((indent, ""))
            continue
        tokens.append((indent, content.rstrip()))

    value, next_i = _parse_block(tokens, 0, 0, source=source)
    # ignore remaining blanks/comments
    for j in range(next_i, len(tokens)):
        indent, content = tokens[j]
        if content.strip():
            raise RulesConfigError(f"{source}: 无法解析的内容（行缩进={indent}）：{content}")
    return value


def _parse_block(tokens: List[Tuple[int, str]], i: int, indent: int, *, source: str) -> Tuple[Any, int]:
    i = _skip_empty(tokens, i)
    if i >= len(tokens):
        return {}, i
    cur_indent, content = tokens[i]
    if cur_indent < indent:
        return {}, i
    if content.lstrip().startswith("- "):
        return _parse_list(tokens, i, indent, source=source)
    return _parse_mapping(tokens, i, indent, source=source)


def _parse_list(tokens: List[Tuple[int, str]], i: int, indent: int, *, source: str) -> Tuple[List[Any], int]:
    out: List[Any] = []
    while i < len(tokens):
        i = _skip_empty(tokens, i)
        if i >= len(tokens):
            break
        cur_indent, content = tokens[i]
        if cur_indent < indent:
            break
        if cur_indent != indent or not content.startswith("- "):
            raise RulesConfigError(f"{source}: 期望 list item（- ...），但得到：{content}")

        rest = content[2:].strip()
        i += 1
        if not rest:
            nested, i = _parse_block(tokens, i, indent + 2, source=source)
            out.append(nested)
            continue

        if ":" in rest and not rest.startswith(("\"", "'")):
            key, val = _split_key_value(rest, source=source)
            item: Dict[str, Any] = {}
            if val in (">", "|"):
                block_text, i = _parse_block_scalar(tokens, i, indent, style=val)
                item[key] = block_text
            elif val == "":
                nested, i = _parse_block(tokens, i, indent + 2, source=source)
                item[key] = nested
            else:
                item[key] = _parse_scalar(val)

            # merge additional mapping entries for this list item
            more, i = _parse_mapping(tokens, i, indent + 2, source=source, allow_empty=True)
            if more:
                item.update(more)
            out.append(item)
            continue

        out.append(_parse_scalar(rest))

    return out, i


def _parse_mapping(
    tokens: List[Tuple[int, str]],
    i: int,
    indent: int,
    *,
    source: str,
    allow_empty: bool = False,
) -> Tuple[Dict[str, Any], int]:
    out: Dict[str, Any] = {}
    while i < len(tokens):
        i = _skip_empty(tokens, i)
        if i >= len(tokens):
            break
        cur_indent, content = tokens[i]
        if cur_indent < indent:
            break
        if cur_indent != indent:
            if allow_empty:
                break
            raise RulesConfigError(f"{source}: 缩进不一致：{content}")
        if content.startswith("- "):
            if allow_empty:
                break
            raise RulesConfigError(f"{source}: mapping 中不应出现 list item：{content}")

        key, val = _split_key_value(content, source=source)
        i += 1

        if val in (">", "|"):
            block_text, i = _parse_block_scalar(tokens, i, indent, style=val)
            out[key] = block_text
            continue

        if val == "":
            nested, i = _parse_block(tokens, i, indent + 2, source=source)
            out[key] = nested
            continue

        out[key] = _parse_scalar(val)

    return out, i


def _parse_block_scalar(
    tokens: List[Tuple[int, str]],
    i: int,
    indent: int,
    *,
    style: str,
) -> Tuple[str, int]:
    lines: List[str] = []
    base_indent: Optional[int] = None
    while i < len(tokens):
        cur_indent, content = tokens[i]
        if content == "":
            next_i = _skip_empty(tokens, i + 1)
            if next_i >= len(tokens):
                break
            next_indent, next_content = tokens[next_i]
            if next_content != "" and next_indent <= indent:
                break
            lines.append("")
            i += 1
            continue
        if cur_indent <= indent:
            break
        if base_indent is None:
            base_indent = cur_indent
        lines.append(content if cur_indent == base_indent else (" " * (cur_indent - base_indent) + content))
        i += 1

    if style == ">":
        # Fold newlines (approximation): keep blank lines as paragraph breaks.
        out: List[str] = []
        paragraph: List[str] = []
        for line in lines:
            if line == "":
                if paragraph:
                    out.append(" ".join(s.strip() for s in paragraph).strip())
                    paragraph = []
                out.append("")
            else:
                paragraph.append(line)
        if paragraph:
            out.append(" ".join(s.strip() for s in paragraph).strip())
        return "\n".join(out).strip(), i

    # Literal
    return "\n".join(lines).strip("\n"), i


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.lower() in ("null", "~"):
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False

    if value.startswith(("\"", "'")) and value.endswith(value[0]) and len(value) >= 2:
        return value[1:-1]

    if value.startswith("[") and value.endswith("]"):
        try:
            return json.loads(value)
        except Exception:
            return _parse_inline_list_fallback(value[1:-1])

    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _split_key_value(text: str, *, source: str) -> Tuple[str, str]:
    if ":" not in text:
        raise RulesConfigError(f"{source}: 无法解析 key/value：{text}")
    key, val = text.split(":", 1)
    key = key.strip()
    if not key:
        raise RulesConfigError(f"{source}: 空 key：{text}")
    return key, val.strip()


def _count_indent(line: str) -> int:
    count = 0
    for ch in line:
        if ch == " ":
            count += 1
        else:
            break
    return count


def _skip_empty(tokens: List[Tuple[int, str]], i: int) -> int:
    while i < len(tokens) and tokens[i][1] == "":
        i += 1
    return i


def _strip_comment(content: str) -> str:
    in_single = False
    in_double = False
    for idx, ch in enumerate(content):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return content[:idx].rstrip()
    return content


def _parse_inline_list_fallback(inner: str) -> List[Any]:
    """Parse YAML-style inline lists like [ERROR, STYLE] without quoting."""
    items: List[Any] = []
    token = []
    in_single = False
    in_double = False
    for ch in inner:
        if ch == "'" and not in_double:
            in_single = not in_single
            token.append(ch)
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            token.append(ch)
            continue
        if ch == "," and not in_single and not in_double:
            text = "".join(token).strip()
            if text:
                items.append(_parse_scalar(text))
            token = []
            continue
        token.append(ch)

    text = "".join(token).strip()
    if text:
        items.append(_parse_scalar(text))

    return items
