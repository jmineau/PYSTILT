"""Finding, downloading, cropping, and staging meteorology files."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pandas as pd
from pandas.tseries.frequencies import to_offset

from stilt.config import MetConfig
from stilt.config.meteorology import arlmet_sources
from stilt.errors import MeteorologyError

if TYPE_CHECKING:
    from arlmet.sources import MeteorologySource as ArlmetSource

logger = logging.getLogger(__name__)


class MetStream:
    """
    Meteorology files for one met stream, found locally or downloaded.

    Without ``source``, files are found under ``directory`` from
    ``file_format`` and ``file_tres``. With ``source`` set to an arlmet
    source name, arlmet downloads the files, cropping them as it goes when
    subgridding is on. Local files are cropped with
    ``arlmet.extract_subset`` into ``subgrid_dir``, which all simulations
    share.

    Parameters
    ----------
    name : str
        Name of the met in the config.
    config : MetConfig
        Its settings.
    """

    def __init__(self, name: str, config: MetConfig):
        self.name = name
        self.config = config
        #: ``config.directory``, made absolute.
        self.directory = config.directory.expanduser().resolve()
        self._arlmet_source: ArlmetSource | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dedupe_matched_files(paths: list[Path]) -> list[Path]:
        """Resolve symlinks and drop duplicate files, sorted by name."""
        return sorted(
            dict.fromkeys(path.resolve() for path in paths),
            key=lambda path: path.name,
        )

    def _get_arlmet_source(self) -> ArlmetSource:
        """Return the arlmet source, building it on first use."""
        if self._arlmet_source is None:
            assert self.config.source is not None
            cls = arlmet_sources()[self.config.source]
            self._arlmet_source = cls(**self.config.source_kwargs)
        return self._arlmet_source

    def _effective_bbox(self) -> tuple[float, float, float, float]:
        """Return ``(west, south, east, north)`` of the subgrid bounds plus the buffer."""
        b = self.config.subgrid_bounds
        buf = self.config.subgrid_buffer
        if b is None:
            raise ValueError("subgrid_bounds is required to compute effective bbox.")
        if buf is None or buf < 0:
            raise ValueError("subgrid_buffer must be a non-negative number.")
        return (b.xmin - buf, b.ymin - buf, b.xmax + buf, b.ymax + buf)

    def _resolved_subgrid_dir(self) -> Path:
        """Return the directory for cropped files, ``<directory>/subgrid`` by default."""
        if self.config.subgrid_dir is None:
            return self.directory / "subgrid"
        return self.config.subgrid_dir.expanduser().resolve()

    def _level_indices(self) -> list[int] | None:
        """Return the indices of the lowest ``subgrid_levels`` levels, or None to keep all."""
        if self.config.subgrid_levels is None:
            return None
        return list(range(self.config.subgrid_levels))

    # ------------------------------------------------------------------
    # File resolution
    # ------------------------------------------------------------------

    def _fetch_from_source(self, r_time: pd.Timestamp, n_hours: int) -> list[Path]:
        """Return the files for a run from the arlmet source, downloading any not yet in ``directory``."""
        sim_end = r_time + pd.Timedelta(hours=n_hours)
        t_start: pd.Timestamp = min(r_time, sim_end)  # type: ignore[assignment]
        t_end: pd.Timestamp = max(r_time, sim_end)  # type: ignore[assignment]

        bbox = self._effective_bbox() if self.config.subgrid_enable else None

        source = self._get_arlmet_source()
        try:
            files = source.fetch(
                t_start,
                t_end,
                local_dir=self.directory,
                backend=self.config.backend,
                bbox=bbox,
            )
        except ImportError as exc:
            raise ImportError(
                f"{exc}\n\n"
                "Downloading meteorology requires the cloud extra. "
                "Install with: pip install pystilt[cloud]"
            ) from exc

        n_files = len(files)
        if n_files == 0 or n_files < self.config.n_min:
            raise MeteorologyError(
                f"Insufficient number of meteorological files found. "
                f"Found: {n_files}, Required: {self.config.n_min}."
            )
        return files

    def required_files(self, r_time, n_hours: int) -> list[Path]:
        """
        Return the met files that cover one simulation.

        Parameters
        ----------
        r_time : datetime-like
            Receptor time.
        n_hours : int
            Simulation length in hours, negative for backward runs.

        Returns
        -------
        list of Path

        Raises
        ------
        MeteorologyError
            Fewer than ``n_min`` files were found.
        """
        _r_time = cast(pd.Timestamp, pd.Timestamp(r_time))

        if self.config.source is not None:
            return self._fetch_from_source(_r_time, n_hours)

        # Archive-glob mode
        sim_end = _r_time + pd.Timedelta(hours=n_hours)
        assert isinstance(sim_end, pd.Timestamp)  # not NaT: _r_time is a time

        earlier = min(_r_time, sim_end)
        later = max(_r_time, sim_end)

        file_format, file_tres = self.config.file_format, self.config.file_tres
        # MetConfig requires both when there is no source.
        assert file_format is not None and file_tres is not None
        tres = to_offset(pd.to_timedelta(file_tres)).freqstr
        met_start = earlier.floor(tres)
        met_end = later

        if n_hours < 0:
            met_end_ceil = later.ceil(tres)
            # As in STILT-R: a release in the last hour of a file interpolates
            # against the next file's first hour; anywhere else it doesn't.
            if later.floor("h") + pd.Timedelta(hours=1) == met_end_ceil:  # type: ignore[arg-type]
                met_end = met_end_ceil

        met_times = pd.date_range(met_start, met_end, freq=tres)
        patterns = list(dict.fromkeys(t.strftime(file_format) for t in met_times))

        files: list[Path] = []
        missing: list[str] = []
        for pattern in patterns:
            # Backup copies (name~<timestamp>~, name.~1~, name~) all end in "~".
            matches = [
                p
                for p in self.directory.rglob(f"{pattern}*")
                if p.is_file() and ".lock" not in p.name and not p.name.endswith("~")
            ]
            if matches:
                files.extend(matches)
            else:
                missing.append(pattern)

        files = self._dedupe_matched_files(files)

        n_files = len(files)
        if n_files == 0 or n_files < self.config.n_min:
            detail = ""
            if missing:
                examples = ", ".join(missing[:3])
                detail = f" Patterns not found in {self.directory}: {examples}."
            raise MeteorologyError(
                f"Insufficient number of meteorological files found. "
                f"Found: {n_files}, Required: {self.config.n_min}.{detail}"
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
        """Find the met files for one simulation and link them into ``target_dir``."""
        return self._stage_files(
            self.required_files(r_time=r_time, n_hours=n_hours),
            target_dir=target_dir,
        )

    def _stage_files(self, files: list[Path], target_dir: Path | str) -> list[Path]:
        """
        Link met files into ``target_dir``, copying when a link fails.

        With subgridding on and no ``source``, each file is cropped into
        ``subgrid_dir`` first and the cropped copy is linked. Downloaded files
        were already cropped.
        """
        # Resolve subgridded paths for archive-mode subsetting
        if self.config.subgrid_enable and self.config.source is None:
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
        """Crop local files into ``subgrid_dir``, reusing crops that already exist."""
        from arlmet import extract_subset

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
