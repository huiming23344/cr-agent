from __future__ import annotations

from collections import defaultdict
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cr_agent.rules.loader import load_rules_catalog


def _repo_root() -> Path:
    return ROOT


def _print_rules(catalog) -> None:
    print("\n=== 按语言 -> domain 分组 ===")
    for language in sorted(catalog.by_language.keys()):
        print(f"\n[{language}]")
        domain_map = catalog.by_language_domain.get(language, {})
        for domain in sorted(domain_map.keys()):
            print(f"  [{domain}]")
            for meta in domain_map[domain]:
                doc = str(meta.doc_path) if meta.doc_path else "-"
                print(f"  - {meta.rule_id}: {meta.title} (doc: {doc})")


def main():
    registry = _repo_root() / "coding-standards" / "registry.yaml"
    catalog = load_rules_catalog(registry_path=registry)
    print(f"Loaded {len(catalog.by_id)} rules from {registry}")
    _print_rules(catalog)


if __name__ == "__main__":
    main()
