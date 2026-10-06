"""
Meteorology: what a met is (:class:`MetConfig`), and finding, downloading, and cropping its files (:class:`Met`).
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from pandas.tseries.frequencies import to_offset
from pydantic import BaseModel, ConfigDict, Field, model_validator

from stilt._atomic import atomic_path
from stilt.exceptions import MeteorologyError
from stilt.spatial import Bounds, haversine_km

if TYPE_CHECKING:
    from arlmet.archives import Archive

logger = logging.getLogger(__name__)


class MetConfig(BaseModel):
    """
    Config for one met, as written under ``mets:`` in ``config.yaml``.

    Give either ``download`` to download ARL files with arlmet, or
    ``file_format`` and ``file_tres`` to find them in ``directory``. With
    ``download``, other keys are options for that archive (such as
    ``domain`` for ``nams``). Any other unknown key is an error.

    A run records its met in its ``_settings.yaml`` without the fields in
    :attr:`UNRECORDED`: the directories, so moving the met files does not
    change which runs are complete, and ``download_from`` and ``n_min``,
    which change no particle. A project requires ``directory`` for every
    met, and ``subgrid_dir`` when it crops local files.
    """

    model_config = ConfigDict(extra="allow")

    #: Fields left out of a run's recorded settings, since they change no particle.
    UNRECORDED: ClassVar[frozenset[str]] = frozenset(
        {"directory", "subgrid_dir", "download_from", "n_min"}
    )

    kind: Literal["files"] = Field(
        "files",
        description=(
            "How the meteorology is stored: ``files``, ARL files in "
            "``directory`` or downloaded into it. The one kind today."
        ),
    )
    directory: Path | None = Field(
        None,
        description=(
            "Directory holding the ARL meteorology files. Downloads are saved "
            "here. A project requires it. In a project, a relative path starts "
            "from the project directory; ``~`` and ``$VARIABLES`` are expanded."
        ),
    )
    subgrid_dir: Path | None = Field(
        None,
        description=(
            "Directory for the cropped files, shared by every simulation that "
            "uses this meteorology. Each crop box gets its own folder inside "
            "it. Required when cropping your own files; not used with "
            "``download``, which crops files as it downloads them. A relative "
            "path starts from the project directory."
        ),
    )

    download: str | None = Field(
        None,
        description=(
            "Name of a NOAA ARL archive to download files from with arlmet, "
            "such as ``hrrr``, ``nam12``, ``gdas1``, or ``gfs0p25``. When set, "
            "``file_format`` and ``file_tres`` are not needed. Options for the "
            "archive (such as ``domain: ak``) go in the same entry."
        ),
    )
    download_from: Literal["s3", "ftp", "http"] = Field(
        "s3",
        description="Server to download from: ``s3``, ``ftp``, or ``http``.",
    )
    file_format: str | None = Field(
        None,
        description=(
            "``strftime`` pattern for the start of each file name, such as "
            "``%Y%m%d_%H``. Files whose names start with it are found anywhere "
            "under ``directory``. Required when ``download`` is not set."
        ),
    )
    file_tres: str | None = Field(
        None,
        description=(
            "Time covered by each file, as a pandas time string such as "
            "``6h``. Required when ``download`` is not set."
        ),
    )
    n_min: int = Field(
        1,
        description="Minimum number of files a simulation needs. Fewer fails the simulation.",
    )
    subgrid_enable: bool = Field(
        False,
        description="Crop the meteorology to ``subgrid_bounds`` before running.",
    )
    subgrid_bounds: Bounds | None = Field(
        None,
        description="Longitude/latitude box to crop the meteorology to.",
    )
    subgrid_buffer: float = Field(
        0.2,
        ge=0,
        description="Margin added to every side of ``subgrid_bounds``, in degrees.",
    )
    subgrid_levels: int | None = Field(
        None,
        description="Number of vertical levels to keep, counted from the surface. Unset keeps all.",
    )

    @model_validator(mode="after")
    def _validate_mode(self) -> Self:
        """Check the archive and its options, and the fields each mode needs."""
        extra = self.download_options
        if self.download is not None:
            from arlmet.archives import ARCHIVES

            if self.download not in ARCHIVES:
                raise ValueError(
                    f"Unknown ARL archive {self.download!r} to download. "
                    f"Available: {sorted(ARCHIVES)}."
                )
            try:
                inspect.signature(ARCHIVES[self.download]).bind(**extra)
            except TypeError as exc:
                raise ValueError(
                    f"ARL archive {self.download!r} does not take the options "
                    f"{sorted(extra)} ({exc})."
                ) from None
        elif extra:
            raise ValueError(
                f"Unknown met settings {sorted(extra)}. Only a met with "
                "download takes extra options."
            )
        if self.download is None and (
            self.file_format is None or self.file_tres is None
        ):
            raise ValueError(
                "file_format and file_tres are required to find local files. "
                "Set download to an ARL archive name, such as hrrr, to "
                "download the files instead."
            )
        if self.subgrid_enable and self.subgrid_bounds is None:
            raise ValueError("subgrid_bounds is required when subgrid_enable=True.")
        return self

    def settings(self) -> dict[str, Any]:
        """
        Return what a run records of this met: every field but the :attr:`UNRECORDED` ones.

        ``kind`` is left out while it is ``files``, the kind every met was
        before it existed, so the records written then still match.
        """
        exclude = set(self.UNRECORDED) | ({"kind"} if self.kind == "files" else set())
        return self.model_dump(mode="json", exclude=exclude)

    @property
    def download_options(self) -> dict[str, Any]:
        """Extra fields, passed as keyword arguments to the arlmet archive."""
        return dict(self.model_extra) if self.model_extra else {}


def _time(value: Any) -> pd.Timestamp:
    """Return *value* as a Timestamp, raising for a missing time."""
    time = pd.Timestamp(value)
    if not isinstance(time, pd.Timestamp):  # NaT
        raise ValueError(f"Not a time: {value!r}")
    return time


def run_window(r_time: Any, n_hours: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Return the time a run covers, ``(start, end)`` in time order.

    From the receptor time *r_time* to *n_hours* later, or earlier for a
    backward run. This is what a transport model's meteorology must cover.
    """
    start = _time(r_time)
    other = _time(start + pd.Timedelta(hours=n_hours))
    return min(start, other), max(start, other)


def _cover(
    window: tuple[Any, Any], hour_after: bool, file_tres: str | None = None
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Return the first and last time files must cover for *window*.

    With *hour_after*, the hour after the window's end is added, unless the
    end is on a file boundary. HYSPLIT needs it for a backward run, because
    it interpolates the release time between two hours: for a release in
    the last hour of a file that hour is in the next file, which STILT-R
    also reads. *file_tres* is the period of one file; without it
    (downloads, whose archive files are 6 hours or longer) the next hour is
    always added, which never adds a file a release on a file boundary does
    not need.
    """
    earlier, later = (_time(t) for t in window)
    if hour_after and (file_tres is None or later != later.floor(file_tres)):
        later = _time(later.floor("h") + pd.Timedelta(hours=1))
    return earlier, later


class Met:
    """
    The meteorology files of one met, found locally or downloaded.

    Without ``download``, files are found under ``directory`` from
    ``file_format`` and ``file_tres``. With ``download`` set to an ARL
    archive name, arlmet downloads the files, cropping them as it goes when
    subgridding is on. Local files are cropped with
    ``arlmet.extract_subset`` into :attr:`crop_dir`, which all simulations
    share.

    Parameters
    ----------
    name : str
        Name of the met in the config.
    config : MetConfig
        Its settings.
    """

    def __init__(self, name: str, config: MetConfig):
        if config.directory is None:
            raise ValueError(f"Met {name!r} has no directory.")
        self.name = name
        self.config = config
        #: ``config.directory``, made absolute.
        self.directory = config.directory.expanduser().resolve()
        self._archive: Archive | None = None

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

    def _get_archive(self) -> Archive:
        """Return the arlmet archive to download from, building it on first use."""
        if self._archive is None:
            from arlmet.archives import get_archive

            assert self.config.download is not None
            self._archive = get_archive(
                self.config.download, **self.config.download_options
            )
        return self._archive

    def _effective_bbox(self) -> tuple[float, float, float, float]:
        """Return ``(west, south, east, north)`` of the subgrid bounds plus the buffer."""
        b = self.config.subgrid_bounds
        buf = self.config.subgrid_buffer
        if b is None:
            raise ValueError("subgrid_bounds is required to compute effective bbox.")
        return (b.xmin - buf, b.ymin - buf, b.xmax + buf, b.ymax + buf)

    @property
    def crop_dir(self) -> Path:
        """
        Directory holding this met's cropped files.

        It is a folder inside ``subgrid_dir`` named by a short hash of the
        crop box (``subgrid_bounds`` plus ``subgrid_buffer``) and
        ``subgrid_levels``. Changing any of them gives a new folder, so
        old crops are never reused for a different crop.
        """
        if self.config.subgrid_dir is None:
            raise ValueError("subgrid_dir is required to crop local files.")
        # Imported here: stilt.identity reads MetConfig from this module.
        from stilt.identity import settings_hash

        crop = {"bbox": self._effective_bbox(), "levels": self.config.subgrid_levels}
        key = settings_hash(crop)[:12]
        return self.config.subgrid_dir.expanduser().resolve() / key

    def _level_indices(self) -> list[int] | None:
        """Return the indices of the lowest ``subgrid_levels`` levels, or None to keep all."""
        if self.config.subgrid_levels is None:
            return None
        return list(range(self.config.subgrid_levels))

    # ------------------------------------------------------------------
    # File resolution
    # ------------------------------------------------------------------

    def _download(self, window: tuple[Any, Any], hour_after: bool) -> list[Path]:
        """Return the files that cover *window* from the ARL archive, downloading any not yet in ``directory``."""
        t_start, t_end = _cover(window, hour_after)

        bbox = self._effective_bbox() if self.config.subgrid_enable else None
        levels = self._level_indices() if self.config.subgrid_enable else None

        archive = self._get_archive()
        try:
            files = archive.fetch(
                t_start,
                t_end,
                dest_dir=self.directory,
                mirror=self.config.download_from,
                bbox=bbox,
                levels=levels,
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

    def required_files(self, r_time: Any, n_hours: int) -> list[Path]:
        """
        Return the met files that cover one simulation, as HYSPLIT reads them.

        Shorthand for :meth:`files_for` the run's window
        (:func:`run_window`), with the hour after the release for a
        backward run.

        Parameters
        ----------
        r_time : datetime-like
            Receptor time.
        n_hours : int
            Simulation length in hours, negative for backward runs.

        Raises
        ------
        MeteorologyError
            Fewer than ``n_min`` files were found.
        """
        return self.files_for(run_window(r_time, n_hours), hour_after=n_hours < 0)

    def files_for(
        self, window: tuple[Any, Any], *, hour_after: bool = False
    ) -> list[Path]:
        """
        Return the met files that cover *window*, ``(start, end)`` in time order.

        Downloads them first when ``download`` is set.

        Parameters
        ----------
        window : tuple of datetime-like
            The time to cover (:func:`run_window`).
        hour_after : bool, default False
            Also cover the hour after the window's end, unless the end is on
            a file boundary. HYSPLIT needs it for a backward run.

        Returns
        -------
        list of Path

        Raises
        ------
        MeteorologyError
            Fewer than ``n_min`` files were found.
        """
        if self.config.download is not None:
            return self._download(window, hour_after)

        # Local files
        file_format, file_tres = self.config.file_format, self.config.file_tres
        # MetConfig requires both when there is no download.
        assert file_format is not None and file_tres is not None
        tres = to_offset(pd.to_timedelta(file_tres)).freqstr
        earlier, later = _cover(window, hour_after, tres)
        met_times = pd.date_range(earlier.floor(tres), later, freq=tres)
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

    def readable(self, files: list[Path]) -> list[Path]:
        """
        Return the files a transport model should read in place of *files*.

        When local files are cropped, these are the cropped copies in
        :attr:`crop_dir`, made on first use. Downloaded files were cropped
        as they were downloaded. When two files share a name, the first is
        kept, since a model would otherwise read the same hours twice.
        """
        if self.config.subgrid_enable and self.config.download is None:
            files = self._crop_local_files(files)
        kept: dict[str, Path] = {}
        for path in files:
            first = kept.setdefault(path.name, path)
            if first is not path and first.resolve() != path.resolve():
                logger.warning(
                    "met %s has duplicate basename %s at %s and %s; using %s",
                    self.name,
                    path.name,
                    first,
                    path,
                    first,
                )
        return list(kept.values())

    def _crop_local_files(self, files: list[Path]) -> list[Path]:
        """
        Crop local files into :attr:`crop_dir`, reusing crops that already exist.

        Each crop is written to a temporary name and then renamed, so a
        worker never reads a half-written file. Two workers cropping the
        same file at once both finish, and the second rename wins.
        """
        from arlmet import extract_subset

        crop_dir = self.crop_dir
        crop_dir.mkdir(parents=True, exist_ok=True)
        bbox = self._effective_bbox()
        levels = self._level_indices()

        subsetted: list[Path] = []
        for src in files:
            cache_path = crop_dir / src.name
            if not cache_path.exists():
                logger.info("Subsetting %s → %s", src.name, cache_path)
                with atomic_path(cache_path) as tmp:
                    extract_subset(src, tmp, bbox=bbox, levels=levels)
            subsetted.append(cache_path)
        return subsetted


# ---------------------------------------------------------------------------
# Wind-error statistics
#
# Variograms of wind errors, for setting up transport-error runs.
#
# HYSPLIT's wind-error perturbation needs the standard deviation of the
# analysis wind error (``siguverr``) and its correlation scales in time
# (``tluverr``), height (``zcoruverr``), and horizontal distance
# (``horcoruverr``). Lin and Gerbig (2005, section 2.1) derive them from the
# differences between analyzed and observed winds. The standard deviation
# comes directly from the differences. Each scale comes from fitting the
# exponential variogram ::
#
#     γ(h) = σ² (1 − exp(−h / l))
#
# to half the mean squared difference of the error between pairs of points
# ``h`` apart in that coordinate. :func:`variogram` computes the empirical
# variogram and :func:`fit_variogram` fits the model to it.
#
# The errors are the analyzed wind minus the observed wind at each
# observation. arlmet samples the analysis at the observation points::
#
#     met = arlmet.sample_points(files, points, ["UWND", "VWND"], earth_relative=True)
#     u_err = met["UWND"] - observed_u
#
# The Wind Error Statistics guide goes from there to the four settings.
# ---------------------------------------------------------------------------


def _bin(
    sums: tuple[np.ndarray, np.ndarray, np.ndarray],
    edges: np.ndarray,
    lag: np.ndarray,
    sq: np.ndarray,
) -> None:
    """Add pairs to the per-bin sums of squared difference, lag, and count."""
    keep = (lag > 0) & (lag < edges[-1])
    lag, sq = lag[keep], sq[keep]
    sums[0][:] += np.histogram(lag, edges, weights=sq)[0]
    sums[1][:] += np.histogram(lag, edges, weights=lag)[0]
    sums[2][:] += np.histogram(lag, edges)[0]


def _pairs_1d(sums, edges: np.ndarray, errors: np.ndarray, coord: np.ndarray) -> None:
    """Bin all pairs of one group closer than the last edge along a 1-D coordinate."""
    order = np.argsort(coord, kind="stable")
    e, c = errors[order], coord[order]
    n = len(c)
    for d in range(1, n):
        i = np.arange(n - d)
        h = c[i + d] - c[i]
        if h.min() >= edges[-1]:
            break  # coordinates are sorted, so wider bands are farther still
        _bin(sums, edges, h, (e[i + d] - e[i]) ** 2)


def _pairs_geo(
    sums, edges: np.ndarray, errors: np.ndarray, lon: np.ndarray, lat: np.ndarray
) -> None:
    """Bin all pairs of one group by great-circle distance in km."""
    if len(errors) < 2:
        return
    i, j = np.triu_indices(len(errors), 1)
    h = haversine_km(lon[i], lat[i], lon[j], lat[j])
    _bin(sums, edges, h, (errors[i] - errors[j]) ** 2)


def variogram(
    errors: ArrayLike,
    lag: ArrayLike,
    *,
    group: ArrayLike | None = None,
    bins: ArrayLike,
) -> pd.DataFrame:
    """
    Return the empirical semivariogram of ``errors`` over a separation coordinate.

    For every pair of points in the same group, the squared difference of
    their errors is binned by their separation. The semivariogram of a bin
    is half the mean squared difference. Bins are half-open, ``[a, b)``.
    Pairs at zero separation or at or beyond the last edge are skipped.

    Parameters
    ----------
    errors : array-like
        One error per point, such as the analysis minus observed value of
        one wind component.
    lag : array-like
        Coordinate the separation is measured in, one value per point, such
        as minutes or meters. An ``(n, 2)`` array of longitude and latitude
        in degrees measures great-circle distance in km.
    group : array-like, optional
        Label per point. Only points with the same label are paired: the
        launch for a vertical variogram of radiosonde errors, the station
        for a time variogram, the observation time for a horizontal
        variogram of a network. ``None`` pairs every point with every
        other, which is fine for a few thousand points but not for a few
        hundred thousand.
    bins : array-like
        Separation bin edges, in the units of ``lag``.

    Returns
    -------
    pandas.DataFrame
        One row per non-empty bin, with columns ``lag`` (mean separation of
        the pairs), ``gamma`` (semivariogram, in the units of ``errors``
        squared), and ``n`` (number of pairs).
    """
    e = np.asarray(errors, dtype=float)
    coord = np.asarray(lag, dtype=float)
    edges = np.asarray(bins, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("bins must be increasing edges with at least two values.")
    if coord.ndim == 1:
        geo = False
    elif coord.ndim == 2 and coord.shape[1] == 2:
        geo = True
    else:
        raise ValueError("lag must be 1-D, or (n, 2) longitude/latitude.")
    if len(coord) != e.size:
        raise ValueError("errors and lag must have the same length.")

    valid = np.isfinite(e) & np.isfinite(coord).reshape(e.size, -1).all(axis=1)
    if group is None:
        labels = np.zeros(e.size, dtype=np.intp)
    else:
        labels = pd.factorize(pd.Series(np.asarray(group, dtype=object)))[0]
        if len(labels) != e.size:
            raise ValueError("group must have one label per point.")
        valid &= labels >= 0

    sums = tuple(np.zeros(edges.size - 1) for _ in range(3))
    idx = np.flatnonzero(valid)
    order = idx[np.argsort(labels[idx], kind="stable")]
    boundaries = np.flatnonzero(np.diff(labels[order])) + 1
    for members in np.split(order, boundaries):
        if members.size < 2:
            continue
        if geo:
            _pairs_geo(sums, edges, e[members], coord[members, 0], coord[members, 1])
        else:
            _pairs_1d(sums, edges, e[members], coord[members])

    sq, lag_sum, n = sums
    keep = n > 0
    return pd.DataFrame(
        {
            "lag": lag_sum[keep] / n[keep],
            "gamma": 0.5 * sq[keep] / n[keep],
            "n": n[keep].astype(int),
        }
    )


@dataclass(frozen=True)
class VariogramFit:
    """
    Exponential variogram ``σ² (1 − exp(−h / l))``, returned by :func:`fit_variogram`.

    Call it with separations to evaluate the model.

    Attributes
    ----------
    sigma : float
        Error standard deviation, the square root of the sill.
    length : float
        E-folding correlation scale, in the units of the separation it was
        fitted over.
    """

    sigma: float
    length: float

    def __call__(self, lag: ArrayLike) -> np.ndarray:
        """Return the model at ``lag``."""
        h = np.asarray(lag, dtype=float)
        return self.sigma**2 * (1.0 - np.exp(-h / self.length))


def fit_variogram(
    lag: ArrayLike, gamma: ArrayLike, *, sigma: float | None = None
) -> VariogramFit:
    """
    Fit an exponential variogram to an empirical one.

    Parameters
    ----------
    lag, gamma : array-like
        The empirical variogram, as returned by :func:`variogram`.
    sigma : float, optional
        Error standard deviation. When given, the sill is fixed at
        ``sigma²`` and only the correlation scale is fitted, as Lin and
        Gerbig's definition implies. Use it when the sample standard
        deviation is known. ``None`` fits both.

    Returns
    -------
    VariogramFit
    """
    from scipy.optimize import curve_fit

    h = np.asarray(lag, dtype=float)
    g = np.asarray(gamma, dtype=float)
    keep = np.isfinite(h) & np.isfinite(g) & (h > 0)
    h, g = h[keep], g[keep]
    if sigma is not None:
        if h.size < 1:
            raise ValueError("fit_variogram needs at least one finite point.")
        (length,), _ = curve_fit(
            lambda x, ell: VariogramFit(sigma, ell)(x),
            h,
            g,
            p0=[float(np.median(h))],
            bounds=(1e-9, np.inf),
        )
        return VariogramFit(float(sigma), float(length))
    if h.size < 2:
        raise ValueError(
            "fit_variogram needs at least two finite points to fit sigma too."
        )
    (length, sig), _ = curve_fit(
        lambda x, ell, s: VariogramFit(s, ell)(x),
        h,
        g,
        p0=[float(np.median(h)), float(np.sqrt(max(g.max(), 1e-12)))],
        bounds=(1e-9, np.inf),
    )
    return VariogramFit(float(sig), float(length))


__all__ = [
    "Met",
    "MetConfig",
    "VariogramFit",
    "fit_variogram",
    "run_window",
    "variogram",
]
