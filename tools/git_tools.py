"""
Git工具函数模块
使用GitPython包操作Git仓库
"""

from __future__ import annotations

import os

from pathlib import Path
from typing import List, Optional, Tuple

from git import Repo
from langchain.tools import tool

from cr_agent.models import AgentState, CommitDiff, FileContentRef, FileDiff

# === 按需读取的大小上限：1MB（写死） ===
MAX_FILE_BYTES = 1_000_000


@tool
def load_file_content(ref: FileContentRef) -> bytes:
    """
    按需读取文件内容（bytes）。不依赖工作区，直接从 Git 对象库读取。
    默认限制：1MB
    """
    repo = Repo(ref.repo_path)
    commit = repo.commit(ref.commit_sha)

    blob = commit.tree / ref.path  # KeyError if not found

    # 大小保护（尽量在读取前就拒绝）
    if blob.size is not None and blob.size > MAX_FILE_BYTES:
        raise ValueError(f"File too large (> {MAX_FILE_BYTES} bytes): {ref.path} @ {ref.commit_sha}")

    data = blob.data_stream.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"File exceeds {MAX_FILE_BYTES} bytes while reading: {ref.path} @ {ref.commit_sha}")
    return data


@tool
def load_file_text(ref: FileContentRef, encoding: str = "utf-8", errors: str = "replace") -> str:
    """按需读取文件内容（text）。默认 utf-8，错误替换。"""
    return load_file_content(ref).decode(encoding, errors)


def _count_added_deleted_from_patch(patch: str) -> Tuple[int, int]:
    """从 unified diff 文本粗略统计 + / - 行数（忽略 '+++', '---' 头部行）。"""
    added = 0
    deleted = 0
    for line in patch.splitlines():
        if line.startswith("+++ ") or line.startswith("--- "):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            deleted += 1
    return added, deleted


def _make_ref(repo_path: str, commit_sha: str, path: Optional[str],
              blob_sha: Optional[str], size: Optional[int], mode: Optional[int]) -> Optional[FileContentRef]:
    if not path:
        return None
    return FileContentRef(
        repo_path=repo_path,
        commit_sha=commit_sha,
        path=path,
        blob_sha=blob_sha,
        size=size,
        mode=mode,
    )

def get_last_commit_diff(state: AgentState):
    """
    获取最新一次提交相对父提交的结构化diff（适合AI代码理解），并补齐 before_ref/after_ref。
    
    Args:
        repo_path: Git仓库路径
        context_lines: 上下文行数，如果为None则从环境变量CONTEXT_LINES读取，默认3行
    """
    try:
        ctx = int(os.getenv('CONTEXT_LINES', '3'))
        resolved_repo_path = Path(state["repo_path"]).resolve()
        repo = Repo(str(resolved_repo_path))

        last_commit = repo.head.commit

        base = CommitDiff(
            repo_path=str(resolved_repo_path),
            commit_sha=last_commit.hexsha,
            parent_sha=last_commit.parents[0].hexsha if last_commit.parents else None,
            author_name=getattr(last_commit.author, "name", "") or "",
            author_email=getattr(last_commit.author, "email", "") or "",
            committed_datetime_iso=last_commit.committed_datetime.isoformat(),
            message=(last_commit.message or "").strip(),
            context_lines=ctx,
            is_initial_commit=not bool(last_commit.parents),
            note=None,
            files=[],
        )

        if not last_commit.parents:
            return {
                "commit_diff": CommitDiff(
                    **{**base.__dict__, "note": "Initial commit (no parent to compare with)"}  # type: ignore[arg-type]
                )
            }

        parent = last_commit.parents[0]

        diff_index = parent.diff(
            last_commit,
            create_patch=True,
            unified=ctx,
        )

        files: List[FileDiff] = []
        repo_path_str = str(resolved_repo_path)

        for change in diff_index:
            diff_bytes: bytes = change.diff or b""
            patch_text = diff_bytes.decode("utf-8", "replace") if diff_bytes else ""

            is_binary = ("GIT binary patch" in patch_text) or ("Binary files" in patch_text)
            added, deleted = _count_added_deleted_from_patch(patch_text)

            a_blob = getattr(change, "a_blob", None)
            b_blob = getattr(change, "b_blob", None)

            a_blob_sha = getattr(a_blob, "hexsha", None) if a_blob else None
            b_blob_sha = getattr(b_blob, "hexsha", None) if b_blob else None
            a_size = getattr(a_blob, "size", None) if a_blob else None
            b_size = getattr(b_blob, "size", None) if b_blob else None

            a_mode = getattr(change, "a_mode", None)
            b_mode = getattr(change, "b_mode", None)

            # 关键：补齐 before_ref / after_ref
            # before_ref 指向 parent commit 下的 a_path 版本
            # after_ref  指向 current commit 下的 b_path 版本
            before_ref = _make_ref(
                repo_path=repo_path_str,
                commit_sha=parent.hexsha,
                path=getattr(change, "a_path", None),
                blob_sha=a_blob_sha,
                size=a_size,
                mode=a_mode,
            )
            after_ref = _make_ref(
                repo_path=repo_path_str,
                commit_sha=last_commit.hexsha,
                path=getattr(change, "b_path", None),
                blob_sha=b_blob_sha,
                size=b_size,
                mode=b_mode,
            )

            fd = FileDiff(
                change_type=getattr(change, "change_type", "") or "",
                a_path=getattr(change, "a_path", None),
                b_path=getattr(change, "b_path", None),
                is_new_file=bool(getattr(change, "new_file", False)),
                is_deleted_file=bool(getattr(change, "deleted_file", False)),
                is_renamed_file=bool(getattr(change, "renamed_file", False)),
                rename_from=getattr(change, "rename_from", None),
                rename_to=getattr(change, "rename_to", None),
                a_blob_sha=a_blob_sha,
                b_blob_sha=b_blob_sha,
                a_mode=a_mode,
                b_mode=b_mode,
                is_binary=is_binary,
                patch=patch_text,
                added_lines=added,
                deleted_lines=deleted,
                before_ref=before_ref,
                after_ref=after_ref,
            )
            files.append(fd)

        return {
            "commit_diff": CommitDiff(
                **{**base.__dict__, "files": files}  # type: ignore[arg-type]
            )
        }
        


    except Exception as e:
        raise Exception(f"获取Git差异信息失败: {str(e)}") from e
