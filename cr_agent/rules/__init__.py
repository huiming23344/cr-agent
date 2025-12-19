"""Rule loading/matching utilities for cr-agent."""

from pathlib import Path
from typing import Dict, List, Optional

from .loader import (
    RULE_DOMAINS,
    SUPPORTED_LANGUAGES,
    RulesCatalog,
    RuleIndex,
    RuleMeta,
    load_rules_catalog,
    load_rules_index,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = _REPO_ROOT / "coding-standards" / "registry.yaml"

GLOBAL_RULES_CATALOG: Optional[RulesCatalog]
GLOBAL_RULES_BY_LANGUAGE: Dict[str, List[RuleMeta]]
GLOBAL_RULES_BY_DOMAIN: Dict[str, List[RuleMeta]]
GLOBAL_RULES_BY_LANGUAGE_DOMAIN: Dict[str, Dict[str, List[RuleMeta]]]


def _empty_catalog() -> RulesCatalog:
    return RulesCatalog(by_id={}, by_language={}, by_domain={}, by_language_domain={})


def _load_default_catalog() -> RulesCatalog:
    try:
        if DEFAULT_REGISTRY_PATH.exists():
            return load_rules_catalog(registry_path=DEFAULT_REGISTRY_PATH)
    except Exception:
        pass
    return _empty_catalog()


GLOBAL_RULES_CATALOG = _load_default_catalog()
GLOBAL_RULES_BY_LANGUAGE = GLOBAL_RULES_CATALOG.by_language
GLOBAL_RULES_BY_DOMAIN = GLOBAL_RULES_CATALOG.by_domain
GLOBAL_RULES_BY_LANGUAGE_DOMAIN = GLOBAL_RULES_CATALOG.by_language_domain


def get_rules_catalog() -> RulesCatalog:
    """Return default rules catalog, loading it on-demand if needed."""
    global GLOBAL_RULES_CATALOG, GLOBAL_RULES_BY_LANGUAGE, GLOBAL_RULES_BY_DOMAIN, GLOBAL_RULES_BY_LANGUAGE_DOMAIN
    if not GLOBAL_RULES_CATALOG or not GLOBAL_RULES_CATALOG.by_id:
        catalog = _load_default_catalog()
        GLOBAL_RULES_CATALOG = catalog
        GLOBAL_RULES_BY_LANGUAGE = catalog.by_language
        GLOBAL_RULES_BY_DOMAIN = catalog.by_domain
        GLOBAL_RULES_BY_LANGUAGE_DOMAIN = catalog.by_language_domain
    return GLOBAL_RULES_CATALOG


__all__ = [
    "RULE_DOMAINS",
    "SUPPORTED_LANGUAGES",
    "RuleIndex",
    "RuleMeta",
    "RulesCatalog",
    "GLOBAL_RULES_CATALOG",
    "GLOBAL_RULES_BY_LANGUAGE",
    "GLOBAL_RULES_BY_DOMAIN",
    "GLOBAL_RULES_BY_LANGUAGE_DOMAIN",
    "get_rules_catalog",
    "load_rules_catalog",
    "load_rules_index",
]
