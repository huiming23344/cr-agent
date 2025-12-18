from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from cr_agent.rules import RULE_DOMAINS
from cr_agent.rules.loader import _load_yaml  # reuse minimal YAML loader
from cr_agent.models import Tag


@dataclass(frozen=True)
class RepoProfile:
    name: str = ""
    priority: int = 100
    match_paths: Tuple[re.Pattern, ...] = tuple()
    domains: Optional[Tuple[Tag, ...]] = None
    skip_regex: Tuple[re.Pattern, ...] = tuple()
    skip_basenames: Tuple[str, ...] = tuple()


@dataclass(frozen=True)
class ProfileConfig:
    repos: List[RepoProfile] = field(default_factory=list)
    default: RepoProfile = field(default_factory=RepoProfile)

    def match_repo(self, repo_path: str) -> RepoProfile:
        normalized = str(Path(repo_path).resolve())
        for profile in sorted(self.repos, key=lambda p: (p.priority, p.name)):
            if any(pat.search(normalized) for pat in profile.match_paths):
                return profile
        return self.default


def load_profile(path: Path) -> ProfileConfig:
    data = _load_yaml(Path(path).expanduser().resolve())
    if not isinstance(data, dict):
        raise ValueError("profile 文件必须是 YAML 映射")

    repos_raw = data.get("repos", [])
    if repos_raw is None:
        repos_raw = []
    if not isinstance(repos_raw, list):
        raise ValueError("profile.repos 必须是列表")

    repos: List[RepoProfile] = []
    for item in repos_raw:
        if not isinstance(item, dict):
            continue
        repos.append(_parse_repo_profile(item))

    default_profile = _parse_repo_profile(data.get("default") or {}, allow_empty_match=True)

    return ProfileConfig(repos=repos, default=default_profile)


def _parse_repo_profile(item: Dict, allow_empty_match: bool = False) -> RepoProfile:
    name = str(item.get("name") or "")
    priority = int(item.get("priority") or 100)

    match_paths_raw = item.get("match_paths") or []
    if not allow_empty_match and not match_paths_raw:
        raise ValueError("每个 repo 配置必须提供 match_paths（正则列表）")
    match_patterns = _compile_patterns(match_paths_raw)

    domains = _normalize_domains(item.get("domains"))
    skip_regex = _compile_patterns(item.get("skip_regex") or [])
    skip_basenames = tuple(str(b).strip() for b in item.get("skip_basenames") or [] if str(b).strip())

    return RepoProfile(
        name=name,
        priority=priority,
        match_paths=match_patterns,
        domains=domains,
        skip_regex=skip_regex,
        skip_basenames=skip_basenames,
    )


def _compile_patterns(patterns: Iterable) -> Tuple[re.Pattern, ...]:
    compiled: List[re.Pattern] = []
    for pat in patterns:
        text = str(pat).strip()
        if not text:
            continue
        compiled.append(re.compile(text))
    return tuple(compiled)


def _normalize_domains(value) -> Optional[Tuple[Tag, ...]]:
    if value is None:
        return None
    out: List[Tag] = []
    for item in value if isinstance(value, (list, tuple, set)) else [value]:
        text = str(item).strip().upper()
        if not text:
            continue
        if text not in RULE_DOMAINS:
            raise ValueError(f"domain '{text}' 不受支持，必须属于 {RULE_DOMAINS}")
        tag = text  # type: ignore[assignment]
        if tag not in out:
            out.append(tag)  # type: ignore[arg-type]
    return tuple(out) if out else None
