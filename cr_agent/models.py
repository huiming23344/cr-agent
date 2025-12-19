from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


# ---- Git data structures ----

@dataclass(frozen=True)
class FileContentRef:
    """Reference to file content inside the git object store."""

    repo_path: str
    commit_sha: str
    path: str
    blob_sha: Optional[str] = None
    size: Optional[int] = None
    mode: Optional[int] = None


@dataclass(frozen=True)
class FileDiff:
    change_type: str
    a_path: Optional[str]
    b_path: Optional[str]
    is_new_file: bool
    is_deleted_file: bool
    is_renamed_file: bool
    rename_from: Optional[str]
    rename_to: Optional[str]
    a_blob_sha: Optional[str]
    b_blob_sha: Optional[str]
    a_mode: Optional[int]
    b_mode: Optional[int]
    is_binary: bool
    patch: str
    added_lines: int
    deleted_lines: int
    before_ref: Optional[FileContentRef]
    after_ref: Optional[FileContentRef]


@dataclass(frozen=True)
class CommitDiff:
    repo_path: str
    commit_sha: str
    parent_sha: Optional[str]
    author_name: str
    author_email: str
    committed_datetime_iso: str
    message: str
    context_lines: int
    is_initial_commit: bool
    note: Optional[str] = None
    files: List[FileDiff] = field(default_factory=list)


# ---- Review result schema ----

Severity = Literal["info", "minor", "major", "critical"]
Category = Literal[
    "bug",
    "security",
    "performance",
    "concurrency",
    "reliability",
    "api",
    "style",
    "test",
    "documentation",
    "build",
    "other",
]


class CRIssue(BaseModel):
    severity: Severity
    category: Category
    message: str = Field(..., min_length=3)
    file_path: Optional[str] = None
    line_start: Optional[int] = Field(default=None, ge=1)
    line_end: Optional[int] = Field(default=None, ge=1)
    suggestion: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    rule_ids: List[str] = Field(default_factory=list)


class FileCRResult(BaseModel):
    file_path: str
    change_type: str
    summary: str = Field(..., min_length=3)
    overall_severity: Severity
    approved: bool
    issues: List[CRIssue] = Field(default_factory=list)
    rule_ids: List[str] = Field(default_factory=list)
    needs_human_review: bool = False
    meta: Dict[str, Any] = Field(default_factory=dict)


# ---- Tagging + tag-based review schema ----

Tag = Literal["STYLE", "ERROR", "API", "CONC", "PERF", "SEC", "TEST", "CONFIG"]


class FileTaggingLLMResult(BaseModel):
    """LLM output for tagging (file_path is filled by the caller)."""

    tags: List[Tag] = Field(default_factory=list)
    reasoning: Optional[str] = None


class FileTaggingResult(BaseModel):
    """LLM-generated tags for a single file diff."""

    file_path: str
    tags: List[Tag] = Field(default_factory=list)
    reasoning: Optional[str] = None


class TagCRLLMResult(BaseModel):
    """LLM output for a tag-specific review (file_path/tag are filled by the caller)."""

    summary: str = Field(..., min_length=3)
    overall_severity: Severity
    approved: bool
    issues: List[CRIssue] = Field(default_factory=list)
    needs_human_review: bool = False
    meta: Dict[str, Any] = Field(default_factory=dict)


class TagCRResult(BaseModel):
    """Review result for one (file, tag) dimension."""

    file_path: str
    tag: Tag
    summary: str = Field(..., min_length=3)
    overall_severity: Severity
    approved: bool
    issues: List[CRIssue] = Field(default_factory=list)
    needs_human_review: bool = False
    rule_ids: List[str] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


class AgentState(TypedDict):
    repo_path: str
    commit_diff: CommitDiff
    file_cr_result: List[FileCRResult]
