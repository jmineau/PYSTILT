"""Meteorology stream selection and shared archive staging helpers."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import cast

import pandas as pd

from stilt.errors import ConfigValidationError, MeteorologyError

logger = logging.getLogger(__name__)


class MetID(str):
    """Identifier for one meteorology stream, used to key receptor footprints."""

    def __new__(cls, name: str):
        if "_" in name:
            raise ValueError(
                "MetID cannot contain underscores, which are reserved for delimiting sim_id components."
            )
        return super().__new__(cls, name)


class MetArchive:
    """Shared meteorology archive used to resolve and stage met inputs."""

    def __init__(self, root: Path | str | None = None):
        self.root = None if root is None else Path(root).expanduser().resolve()

    def resolve_directory(self, directory: Path | str) -> Path:
        """Resolve one stream directory relative to the archive root when needed."""
        path = Path(directory).expanduser()
        if path.is_absolute() or self.root is None:
            return path.resolve()
        return (self.root / path).resolve()

    def stage_files(self, files: list[Path], target_dir: Path | str) -> list[Path]:
        """Materialize met files into a compute-local directory via link-or-copy."""
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)

        staged: list[Path] = []
        staged_sources: dict[Path, Path] = {}
        for src in files:
            src = Path(src)
            resolved_src = src.resolve()
            if src.parent == target:
                existing = staged_sources.get(src)
                if existing is not None:
                    continue
                staged_sources[src] = resolved_src
                staged.append(src)
                continue

            dst = target / src.name
            existing = staged_sources.get(dst)
            if existing is not None:
                if existing != resolved_src:
                    logger.warning(
                        "met archive has duplicate basename %s at %s and %s; staging %s",
                        src.name,
                        existing,
                        resolved_src,
                        existing,
                    )
                continue

            staged_sources[dst] = resolved_src
            if dst.exists() or dst.is_symlink():
                staged.append(dst)
                continue

            try:
                dst.symlink_to(resolved_src)
            except OSError:
                shutil.copy2(src, dst)
            staged.append(dst)
        return staged


class MetStream:
    """Meteorology source configuration plus file lookup for one named stream."""

    def __init__(
        self,
        met_id: MetID | str,
        directory: Path | str,
        file_format: str,
        file_tres: pd.Timedelta | str,
        n_min: int = 1,
        subgrid_enable: Path | bool = False,
        subgrid_bounds=None,
        subgrid_buffer: float = 0.1,
        subgrid_levels: int | None = None,
        archive: MetArchive | None = None,
    ):
        self.id = MetID(met_id)
        self.archive = archive or MetArchive()
        self.directory = self.archive.resolve_directory(directory)
        self.file_format = file_format
        self.file_tres = pd.to_timedelta(file_tres)
        self.n_min = int(n_min)
        self.subgrid_enable = subgrid_enable
        self.subgrid_bounds = subgrid_bounds
        self.subgrid_buffer = subgrid_buffer
        self.subgrid_levels = subgrid_levels

        # Cache the stream directory inventory for repeated local inspections.
        self._available_files: list[Path] | None = None
        self._broken_links: list[Path] | None = None

    @classmethod
    def _dedupe_matched_files(cls, paths: list[Path]) -> list[Path]:
        """Resolve symlinks and collapse duplicate archive entries."""
        return sorted(
            dict.fromkeys(path.resolve() for path in paths),
            key=lambda path: path.name,
        )

    def _scan_inventory(self) -> tuple[list[Path], list[Path]]:
        """Return usable files and broken symlink entries under the stream root."""
        files: list[Path] = []
        broken: list[Path] = []

        for path in self.directory.rglob("*"):
            if ".lock" in path.name or path.is_dir():
                continue
            if path.is_symlink() and not path.exists():
                broken.append(path)
                continue
            if path.exists() and path.is_file():
                files.append(path)

        files.sort()
        broken.sort()
        return files, broken

    @property
    def available_files(self) -> list[Path]:
        """List meteorology files currently discoverable for this stream."""
        if not self._available_files:
            self._available_files, self._broken_links = self._scan_inventory()
        return self._available_files

    @property
    def broken_links(self) -> list[Path]:
        """Broken symlink entries seen under the stream root."""
        if self._broken_links is None:
            self._available_files, self._broken_links = self._scan_inventory()
        return cast(list[Path], self._broken_links)

    def required_files(self, r_time, n_hours: int) -> list[Path]:
        """Return source met files required to cover one simulation time window."""
        r_time = pd.Timestamp(r_time)
        sim_end = r_time + pd.Timedelta(hours=n_hours)

        earlier = min(r_time, sim_end)
        later = max(r_time, sim_end)

        met_start = earlier.floor(self.file_tres)  # type: ignore[arg-type]
        met_end = later

        if n_hours < 0:
            met_end_ceil = later.ceil(self.file_tres)  # type: ignore[arg-type]
            if later < met_end_ceil:
                met_end = met_end_ceil

        met_times = pd.date_range(met_start, met_end, freq=self.file_tres)
        request = list(dict.fromkeys(t.strftime(self.file_format) for t in met_times))

        files = [
            path
            for path in self.available_files
            if any(path.name.startswith(pattern) for pattern in request)
        ]
        files = self._dedupe_matched_files(files)
        broken = [
            path
            for path in self.broken_links
            if any(path.name.startswith(pattern) for pattern in request)
        ]
        broken = sorted(
            dict.fromkeys(broken),
            key=lambda path: path.name,
        )

        n_files = len(files)
        if n_files == 0 or n_files < self.n_min:
            detail = ""
            if broken:
                examples = ", ".join(path.name for path in broken[:3])
                detail = f" Matching archive entries were broken symlinks: {examples}."
            raise MeteorologyError(
                f"Insufficient meteorological files found. "
                f"Found: {n_files}, "
                f"Required: {self.n_min}."
                f"{detail}"
            )

        if self.subgrid_enable is False:
            return files

        raise ConfigValidationError(
            "Subgrid met cropping is not yet implemented. PYSTILT will rely on "
            "future arl-met cropping and ARL-writing support here. Leave "
            "subgrid_enable unset (False)."
        )

    def stage_files_for_simulation(
        self,
        *,
        r_time,
        n_hours: int,
        target_dir: Path | str,
    ) -> list[Path]:
        """Resolve and stage met files into a simulation-local compute directory."""
        return self.archive.stage_files(
            self.required_files(r_time=r_time, n_hours=n_hours),
            target_dir=target_dir,
        )
