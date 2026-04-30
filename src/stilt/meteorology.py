"""Meteorology source configuration, file lookup, and staging."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd

from stilt.errors import ConfigValidationError, MeteorologyError

logger = logging.getLogger(__name__)


class MetID(str):
    """Identifier for one meteorology source, used to key receptor footprints."""

    def __new__(cls, name: str):
        if "_" in name:
            raise ValueError(
                "MetID cannot contain underscores, which are reserved for delimiting sim_id components."
            )
        return super().__new__(cls, name)


class MetSource:
    """Meteorology source configuration, file lookup, and staging for one named met product."""

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
    ):
        self.id = MetID(met_id)
        self.directory = Path(directory).expanduser().resolve()
        self.file_format = file_format
        self.file_tres = pd.to_timedelta(file_tres)
        self.n_min = int(n_min)
        self.subgrid_enable = subgrid_enable
        self.subgrid_bounds = subgrid_bounds
        self.subgrid_buffer = subgrid_buffer
        self.subgrid_levels = subgrid_levels

    @staticmethod
    def _dedupe_matched_files(paths: list[Path]) -> list[Path]:
        """Resolve symlinks and collapse duplicate archive entries."""
        return sorted(
            dict.fromkeys(path.resolve() for path in paths),
            key=lambda path: path.name,
        )

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
        patterns = list(dict.fromkeys(t.strftime(self.file_format) for t in met_times))

        files: list[Path] = []
        missing: list[str] = []
        for pattern in patterns:
            matches = [
                p
                for p in self.directory.rglob(f"{pattern}*")
                if p.is_file() and ".lock" not in p.name
            ]
            if matches:
                files.extend(matches)
            else:
                missing.append(pattern)

        files = self._dedupe_matched_files(files)

        n_files = len(files)
        if n_files == 0 or n_files < self.n_min:
            detail = ""
            if missing:
                examples = ", ".join(missing[:3])
                detail = f" Patterns not found in {self.directory}: {examples}."
            raise MeteorologyError(
                f"Insufficient meteorological files found. "
                f"Found: {n_files}, Required: {self.n_min}.{detail}"
            )

        if missing:
            examples = ", ".join(missing[:3])
            logger.warning(
                "Met patterns not found (simulation may lack temporal coverage): %s",
                examples,
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
        return self._stage_files(
            self.required_files(r_time=r_time, n_hours=n_hours),
            target_dir=target_dir,
        )

    def _stage_files(self, files: list[Path], target_dir: Path | str) -> list[Path]:
        """Materialize met files into a compute-local directory via link-or-copy."""
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)

        staged: list[Path] = []
        staged_sources: dict[Path, Path] = {}
        for src in files:
            src = Path(src)
            resolved_src = src.resolve()
            if src.parent == target:
                if src not in staged_sources:
                    staged_sources[src] = resolved_src
                    staged.append(src)
                continue

            dst = target / src.name
            existing = staged_sources.get(dst)
            if existing is not None:
                if existing != resolved_src:
                    logger.warning(
                        "met source has duplicate basename %s at %s and %s; staging %s",
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
