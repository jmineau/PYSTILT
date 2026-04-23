"""Resolved project roots and path helpers."""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path


def resolve_directory(
    directory: str | Path | None = None, *, prefix: str = "pystilt_"
) -> Path:
    """Return a resolved directory path, creating a temp root when omitted."""
    if directory is None:
        return Path(tempfile.mkdtemp(prefix=prefix))
    directory = Path(directory)
    if directory.parent == Path("."):
        directory = directory.resolve()
    return directory


def is_cloud_project(project: str) -> bool:
    """Return True when *project* is an object-storage URI."""
    return project.startswith(("s3://", "gs://"))


def project_slug(project: str) -> str:
    """Derive a DNS-safe/local-safe slug from a project path or URI."""
    raw = project.rstrip("/")
    if "://" in raw:
        raw = raw.split("://", 1)[1]
    parts = [part for part in raw.split("/") if part]
    candidate = parts[-1] if parts else "project"
    slug = candidate.lower().replace("_", "-")
    slug = re.sub(r"[^a-z0-9-]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "project"


def uri_join(root: str, *parts: str) -> str:
    """Join path fragments onto a local path or object-store URI."""
    clean_parts = [part.strip("/") for part in parts if part and part.strip("/")]
    if "://" in root:
        base = root.rstrip("/")
        if not clean_parts:
            return base
        return f"{base}/{'/'.join(clean_parts)}"

    path = Path(root)
    for part in clean_parts:
        path /= part
    return str(path)


@dataclass(frozen=True, slots=True)
class ProjectLayout:
    """Resolved local directories and durable refs for one model instance."""

    project_ref: str
    output_ref: str
    is_cloud_project: bool
    is_cloud_output: bool
    project_dir: Path
    output_dir: Path

    @classmethod
    def resolve(
        cls,
        project: str | Path | None,
        output_dir: str | Path | None,
    ) -> ProjectLayout:
        """Resolve project/output refs and any required local working roots."""
        project_ref = str(project or "")
        output_ref = str(output_dir) if output_dir is not None else project_ref
        if not project_ref and output_dir is not None:
            project_ref = output_ref

        project_is_cloud = is_cloud_project(project_ref)
        output_is_cloud = is_cloud_project(output_ref)

        project_dir: str | Path | None = None
        output_local_dir: str | Path | None = None
        if not project_is_cloud:
            project_dir = project or (
                output_dir if output_dir is not None and not output_is_cloud else None
            )
        if not output_is_cloud:
            output_local_dir = output_dir or project or None

        return cls(
            project_ref=project_ref,
            output_ref=output_ref,
            is_cloud_project=project_is_cloud,
            is_cloud_output=output_is_cloud,
            project_dir=resolve_directory(project_dir),
            output_dir=resolve_directory(output_local_dir),
        )

    @property
    def project_root(self) -> str:
        """Return the canonical project ref or resolved local directory."""
        return self.project_ref if self.is_cloud_project else str(self.project_dir)

    @property
    def output_root(self) -> str:
        """Return the canonical durable output ref or resolved local directory."""
        return self.output_ref if self.is_cloud_output else str(self.output_dir)
