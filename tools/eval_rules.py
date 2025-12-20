from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: List[str], *, cwd: Path, env: Optional[Dict[str, str]] = None) -> None:
    final_env = os.environ.copy()
    if env:
        final_env.update(env)
    subprocess.run(cmd, cwd=str(cwd), env=final_env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _load_expect(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _gather_cases(root: Path) -> List[Path]:
    return [p for p in root.rglob("expect.yaml") if p.is_file()]


def _init_repo(repo_dir: Path) -> None:
    _run(["git", "init"], cwd=repo_dir)
    _run(["git", "config", "user.name", "eval"], cwd=repo_dir)
    _run(["git", "config", "user.email", "eval@example.com"], cwd=repo_dir)
    # baseline commit (empty) to ensure subsequent commits have a parent
    _run(["git", "commit", "--allow-empty", "-m", "baseline"], cwd=repo_dir)


def _apply_base_and_patch(case_dir: Path, workdir: Path) -> None:
    base_dir = case_dir / "base"
    if base_dir.exists():
        for item in base_dir.rglob("*"):
            if item.is_file():
                target = workdir / item.relative_to(base_dir)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
        _run(["git", "add", "."], cwd=workdir)
        _run(["git", "commit", "-m", "base"], cwd=workdir)

    patch_file = case_dir / "patch.diff"
    if not patch_file.exists():
        raise FileNotFoundError(f"patch.diff not found in {case_dir}")
    _run(["git", "apply", str(patch_file)], cwd=workdir)
    _run(["git", "add", "."], cwd=workdir)
    _run(["git", "commit", "-m", "case"], cwd=workdir)


def _run_agent(repo_path: Path, profile: Path, env_file: Optional[str]) -> Path:
    env = os.environ.copy()
    env["CR_EVAL_MODE"] = "1"
    cmd = ["python", str(PROJECT_ROOT / "agent.py"), "--repo", str(repo_path), "--profile", str(profile)]
    if env_file:
        cmd.extend(["--env-file", str(Path(env_file).resolve())])
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=True)

    ndjson_files = list(repo_path.glob("cr_report_*.ndjson"))
    if not ndjson_files:
        raise FileNotFoundError(f"No NDJSON report generated under {repo_path}")
    return ndjson_files[-1]


def _read_ndjson(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _evaluate_case(case_dir: Path, profile: Path, env_file: Optional[str]) -> Dict:
    expect = _load_expect(case_dir / "expect.yaml")
    rule_id = str(expect.get("rule_id") or "")
    case_name = case_dir.name

    with tempfile.TemporaryDirectory() as tmp:
        repo_dir = Path(tmp) / "repo"
        repo_dir.mkdir(parents=True, exist_ok=True)
        _init_repo(repo_dir)
        _apply_base_and_patch(case_dir, repo_dir)
        ndjson_path = _run_agent(repo_dir, profile, env_file)
        records = _read_ndjson(ndjson_path)

    hits = [r for r in records if r.get("rule_id") == rule_id]
    actual_hit = bool(hits)
    result = {
        "case": case_name,
        "rule_id": rule_id,
        "expect_hit": bool(expect.get("expect_hit", True)),
        "actual_hit": actual_hit,
        "expected_count": expect.get("expect_count"),
        "actual_count": len(hits),
        "allow_other_rules": bool(expect.get("allow_other_rules", True)),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Run rule evaluation suite.")
    parser.add_argument("--cases", default="eval/cases", help="Root directory of test cases.")
    parser.add_argument("--profile", default="profiles/eval.yaml", help="Profile YAML path for evaluation.")
    parser.add_argument("--env-file", dest="env_file", help="Optional .env file passed to agent.")
    parser.add_argument("--out", default="results/eval.ndjson", help="Output NDJSON file.")
    args = parser.parse_args()

    cases_root = Path(args.cases).resolve()
    profile_path = Path(args.profile).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []
    for expect_path in _gather_cases(cases_root):
        case_dir = expect_path.parent
        print(f"[EVAL] Running case: {case_dir}")
        res = _evaluate_case(case_dir, profile_path, args.env_file)
        results.append(res)

    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[EVAL] Done. Results written to {out_path}")


if __name__ == "__main__":
    main()
