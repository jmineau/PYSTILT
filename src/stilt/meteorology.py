"""Meteorology source configuration, file lookup, and staging."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from stilt.errors import ConfigValidationError, MeteorologyError

if TYPE_CHECKING:
    from arlmet.sources import MeteorologySource as ArlmetSource

logger = logging.getLogger(__name__)


class MetID(str):
    """Identifier for one meteorology stream, used to key receptor footprints."""

    def __new__(cls, name: str):
        if "_" in name:
            raise ValueError(
                "MetID cannot contain underscores, which are reserved for delimiting sim_id components."
            )
        return super().__new__(cls, name)


def _build_arlmet_source(name: str, kwargs: dict[str, Any]) -> ArlmetSource:
    """
    Construct an arlmet MeteorologySource by name, with optional kwargs.

    Raises ImportError (pointing to stilt[cloud]) if fsspec/s3fs are absent
    when the caller actually tries to download. The import here is lazy so that
    arlmet's core (subsetting) works without the cloud extra.
    """
    try:
        import arlmet.sources as _src
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "arlmet is required for source-mode meteorology. "
            "Install with: pip install pystilt[cloud]"
        ) from exc

    # Build registry dynamically from arlmet's public surface so new sources
    # are automatically available without changes here.
    registry: dict[str, type] = {
        cls.name: cls  # type: ignore[attr-defined]
        for attr in _src.__all__
        if (cls := getattr(_src, attr, None)) is not None
        and isinstance(cls, type)
        and issubclass(cls, _src.MeteorologySource)
        and cls is not _src.MeteorologySource
    }

    if name not in registry:
        raise ConfigValidationError(
            f"Unknown arlmet source {name!r}. Available: {sorted(registry)}."
        )

    return registry[name](**kwargs)


class MetStream:
    """
    Runtime handler that resolves and stages met files for one named met stream.

    In *archive mode* (``source_type=None``) it globs an existing local
    directory using ``file_format`` / ``file_tres`` to discover required files.

    In *source mode* (``source_type`` set to an arlmet source name) it delegates
    file resolution and optional on-download cropping to an
    ``arlmet.MeteorologySource`` instance, then stages the result.

    Subgridding of local archive files is handled inside ``_stage_files`` via
    ``arlmet.extract_subset``, caching cropped copies in ``subgrid_dir``.
    """

    def __init__(
        self,
        met_id: MetID | str,
        directory: Path | str,
        file_format: str | None = None,
        file_tres: pd.Timedelta | str | None = None,
        n_min: int = 1,
        source_type: str | None = None,
        source_kwargs: dict[str, Any] | None = None,
        backend: str = "s3",
        subgrid_enable: bool = False,
        subgrid_bounds=None,
        subgrid_buffer: float = 0.2,
        subgrid_levels: int | None = None,
        subgrid_dir: Path | None = None,
    ):
        self.id = MetID(met_id)
        self.directory = Path(directory).expanduser().resolve()
        self.file_format = file_format
        self.file_tres = pd.to_timedelta(file_tres) if file_tres is not None else None
        self.n_min = int(n_min)
        self.source_type = source_type
        self.source_kwargs = source_kwargs or {}
        self.backend = backend
        self.subgrid_enable = subgrid_enable
        self.subgrid_bounds = subgrid_bounds
        self.subgrid_buffer = subgrid_buffer
        self.subgrid_levels = subgrid_levels
        self.subgrid_dir = (
            Path(subgrid_dir).expanduser().resolve() if subgrid_dir else None
        )

        # Lazily constructed arlmet source instance (source mode only)
        self._arlmet_source: ArlmetSource | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dedupe_matched_files(paths: list[Path]) -> list[Path]:
        """Resolve symlinks and collapse duplicate archive entries."""
        return sorted(
            dict.fromkeys(path.resolve() for path in paths),
            key=lambda path: path.name,
        )

    def _get_arlmet_source(self) -> ArlmetSource:
        """Return (and cache) the arlmet source instance."""
        if self._arlmet_source is None:
            assert self.source_type is not None
            self._arlmet_source = _build_arlmet_source(
                self.source_type, self.source_kwargs
            )
        return self._arlmet_source

    def _effective_bbox(self) -> tuple[float, float, float, float]:
        """Return (west, south, east, north) bbox with buffer applied."""
        b = self.subgrid_bounds
        buf = self.subgrid_buffer
        if b is None:
            raise ValueError("subgrid_bounds is required to compute effective bbox.")
        if buf is None or buf < 0:
            raise ValueError("subgrid_buffer must be a non-negative number.")
        return (b.xmin - buf, b.ymin - buf, b.xmax + buf, b.ymax + buf)

    def _resolved_subgrid_dir(self) -> Path:
        """Return the subgrid cache directory, auto-picking directory/subgrid."""
        return (
            self.subgrid_dir
            if self.subgrid_dir is not None
            else self.directory / "subgrid"
        )

    def _level_indices(self) -> list[int] | None:
        """Return level indices to keep, or None to keep all."""
        if self.subgrid_levels is None:
            return None
        return list(range(self.subgrid_levels))

    # ------------------------------------------------------------------
    # File resolution
    # ------------------------------------------------------------------

    def _fetch_from_source(self, r_time: pd.Timestamp, n_hours: int) -> list[Path]:
        """Resolve required files via an arlmet source (download / cached download)."""
        sim_end = r_time + pd.Timedelta(hours=n_hours)
        t_start: pd.Timestamp = min(r_time, sim_end)  # type: ignore[assignment]
        t_end: pd.Timestamp = max(r_time, sim_end)  # type: ignore[assignment]

        bbox = self._effective_bbox() if self.subgrid_enable else None

        source = self._get_arlmet_source()
        try:
            files = source.fetch(
                t_start,
                t_end,
                local_dir=self.directory,
                backend=self.backend,
                bbox=bbox,
            )
        except ImportError as exc:
            raise ImportError(
                f"{exc}\n\n"
                "Downloading meteorology requires the cloud extra. "
                "Install with: pip install pystilt[cloud]"
            ) from exc

        n_files = len(files)
        if n_files == 0 or n_files < self.n_min:
            raise MeteorologyError(
                f"Insufficient meteorological files found. "
                f"Found: {n_files}, Required: {self.n_min}."
            )
        return files

    def required_files(self, r_time, n_hours: int) -> list[Path]:
        """Return source met files required to cover one simulation time window."""
        _r_time = cast(pd.Timestamp, pd.Timestamp(r_time))

        if self.source_type is not None:
            return self._fetch_from_source(_r_time, n_hours)

        # Archive-glob mode
        sim_end = _r_time + pd.Timedelta(hours=n_hours)

        earlier = min(_r_time, sim_end)
        later = max(_r_time, sim_end)

        met_start = earlier.floor(self.file_tres)  # type: ignore[arg-type]
        met_end = later

        if n_hours < 0:
            met_end_ceil = later.ceil(self.file_tres)  # type: ignore[arg-type]
            if later < met_end_ceil:
                met_end = met_end_ceil

        met_times = pd.date_range(met_start, met_end, freq=self.file_tres)
        patterns = list(dict.fromkeys(t.strftime(self.file_format) for t in met_times))  # type: ignore[arg-type]

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

        return files

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
        """
        Materialize met files into a compute-local directory via link-or-copy.

        When subgrid_enable is True and we are in archive mode (source_type is
        None), each source file is first subsetted via arlmet.extract_subset into
        subgrid_dir (shared cache), then the cached copy is linked/copied into the
        per-simulation target. In source mode, files are already cropped on fetch.
        """
        # Resolve subgridded paths for archive-mode subsetting
        if self.subgrid_enable and self.source_type is None:
            files = self._subset_archive_files(files)

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

    def _subset_archive_files(self, files: list[Path]) -> list[Path]:
        """Subset archive files via arlmet.extract_subset, caching in subgrid_dir."""
        try:
            from arlmet import extract_subset
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "arlmet is required for met subsetting. "
                "Install with: pip install pystilt"
            ) from exc

        subgrid_dir = self._resolved_subgrid_dir()
        subgrid_dir.mkdir(parents=True, exist_ok=True)
        bbox = self._effective_bbox()
        levels = self._level_indices()

        subsetted: list[Path] = []
        for src in files:
            cache_path = subgrid_dir / src.name
            if not cache_path.exists():
                logger.info("Subsetting %s → %s", src.name, cache_path)
                extract_subset(src, cache_path, bbox=bbox, levels=levels)
            subsetted.append(cache_path)
        return subsetted
