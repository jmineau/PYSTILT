"""
Write HYSPLIT's input files, run ``hycs_std``, and read the particles it writes.

:func:`write_inputs` fills a working directory with everything
``hycs_std`` reads, and :func:`read_particle_dat` reads the particle file
it writes. :class:`~stilt.transport.hysplit.HysplitModel` runs the two
around ``hycs_std`` in one directory; call them yourself to look at the
input files of a run, or to read a ``PARTICLE_STILT.DAT`` from elsewhere.
"""

import os
import platform
import signal
import subprocess
import warnings
from collections.abc import Sequence
from importlib.resources import files as pkg_files
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from stilt.exceptions import HYSPLITNotFoundError, SimulationError
from stilt.receptors import Receptor
from stilt.transport.hysplit.config import (
    WIND_ERROR_SETTINGS,
    ZI_ERROR_SETTINGS,
    HysplitConfig,
)
from stilt.transport.hysplit.control import ControlFile
from stilt.transport.hysplit.failures import (
    PARTICLE_LIMIT_PHRASE,
    FailureReason,
    failure_in,
)
from stilt.transport.hysplit.namelist import NameList

if TYPE_CHECKING:
    from arlmet import Grid

CONTROL_FILE = "CONTROL"
SETUP_FILE = "SETUP.CFG"
HYCS_STD_FILE = "hycs_std"
PARTICLE_STILT_FILE = "PARTICLE_STILT.DAT"
WINDERR_FILE = "WINDERR"
ZIERR_FILE = "ZIERR"
ZICONTROL_FILE = "ZICONTROL"
LOG_FILE = "stilt.log"
#: HYSPLIT's own file of warnings, such as the met running out.
WARNING_FILE = "WARNING"

#: The settings ``CONTROL`` takes; :class:`ControlFile` writes them.
CONTROL_SETTINGS = ("n_hours", "emisshrs", "w_option", "z_top")

#: Settings PYSTILT uses itself, in no HYSPLIT input file.
PYSTILT_SETTINGS = ("hnf_plume", "exe_dir", "data_dir")

#: Every setting that is not a ``SETUP.CFG`` entry. All other settings are.
NOT_IN_SETUP = frozenset(
    {
        *CONTROL_SETTINGS,
        "ziscale",  # ZICONTROL
        *WIND_ERROR_SETTINGS,
        *ZI_ERROR_SETTINGS,
        *PYSTILT_SETTINGS,
    }
)


def _bundled_exe_dir() -> Path:
    """
    Return the directory of the bundled ``hycs_std`` for this platform.

    The bundled builds are x86-64 Linux and x86-64 macOS. The macOS build
    also runs on Apple Silicon through Rosetta. Any other platform raises,
    rather than running a binary the operating system cannot execute. So
    does an install from the source archive, which carries no binary.
    """
    system = platform.system()
    machine = platform.machine().lower()
    subdir = None
    if system == "Linux" and machine in ("x86_64", "amd64"):
        subdir = "linux_x64"
    elif system == "Darwin" and machine in ("x86_64", "arm64"):
        subdir = "macos_x64"
    if subdir is not None:
        exe_dir = Path(str(pkg_files("stilt.transport.hysplit") / "bin" / subdir))
        if (exe_dir / HYCS_STD_FILE).is_file():
            return exe_dir
    raise HYSPLITNotFoundError(
        f"No bundled HYSPLIT binary for {system} {platform.machine()}. "
        "Build hycs_std for this machine and set exe_dir in config.yaml "
        "to the directory that holds it."
    )


def _bundled_data_dir() -> Path:
    """Return the directory of HYSPLIT's bundled data tables."""
    return Path(str(pkg_files("stilt.transport.hysplit") / "data"))


#: HYSPLIT's names for the columns the particle table names otherwise
#: (:data:`stilt.particles.PARTICLE_SCHEMA`); the rest keep HYSPLIT's.
PARTICLE_COLUMNS = {"indx": "particle", "time": "age", "long": "lon", "lati": "lat"}


def read_particle_dat(path: str | Path, columns: Sequence[str]) -> pd.DataFrame:
    """
    Read a ``PARTICLE_STILT.DAT`` file as a particle table.

    The ``varsiwant`` codes ``indx``, ``time``, ``long``, and ``lati``
    become the particle table's ``particle``, ``age``, ``lon``, and ``lat``
    (:data:`PARTICLE_COLUMNS`).

    Parameters
    ----------
    path : str or Path
        The file HYSPLIT wrote.
    columns : sequence of str
        Its columns, the ``varsiwant`` of the run, in order.

    Returns
    -------
    pandas.DataFrame
        One row per particle per output step, as HYSPLIT wrote them. The
        release heights and the near-field correction are added after the
        run (:func:`stilt.transport.run_model`).

    Raises
    ------
    ValueError
        If the file has a different number of columns.
    """
    path = Path(path)
    names = [PARTICLE_COLUMNS.get(name, name) for name in columns]
    # numpy.loadtxt reads large files much faster than pandas and handles
    # HYSPLIT's variable-width spacing.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="loadtxt: input contained no data",
            category=UserWarning,
        )
        values = np.loadtxt(path, skiprows=1, ndmin=2)

    if values.size == 0:
        return pd.DataFrame(columns=pd.Index(names))
    if values.shape[1] != len(names):
        raise ValueError(
            f"{path.name} has {values.shape[1]} columns, expected {len(names)} "
            f"from varsiwant={list(columns)!r}."
        )
    return pd.DataFrame(values, columns=pd.Index(names))


# -- what each input file holds -----------------------------------------------


def setup_entries(params: HysplitConfig) -> dict[str, Any]:
    """
    Return the ``SETUP.CFG`` namelist entries of *params*, leaving out unset ones.

    Every setting not in :data:`NOT_IN_SETUP` is an entry, in declaration
    order. The driver adds ``KMSL``, ``IVMAX``, and ``WINDERRTF``, and
    ``MAXPAR`` when ``maxpar`` is unset, which depend on the receptor or
    follow from other settings.
    """
    entries = {
        name: getattr(params, name)
        for name in HysplitConfig.model_fields
        if name not in NOT_IN_SETUP and getattr(params, name) is not None
    }
    entries["zicontroltf"] = zicontroltf(params)
    if params.seed is not None:
        entries["seed"] = setup_seed(params.seed)
    return entries


def setup_seed(seed: int) -> int:
    """
    Return the ``SEED`` value written to ``SETUP.CFG`` for a user seed.

    HYSPLIT sets its generator state to ``-1 + SEED``. Under ``krand=2`` it
    reinitializes only from a negative state, and every state of -1 or more
    gives the same stream. Writing ``-(|seed| + 1)`` puts the state at
    ``-(|seed| + 2)``. That is negative, different for each ``|seed|``, and
    never the unseeded default (``SEED = 0``). A patched HYSPLIT that uses
    ``SEED`` directly maps a negative ``SEED`` to the same state, so the
    value works with both builds.
    """
    return -(abs(seed) + 1)


def release_locations(receptor: Receptor) -> int:
    """
    Return the number of horizontal locations *receptor* releases particles from.

    HYSPLIT divides ``numpar`` among them. A :class:`~stilt.receptors.ColumnReceptor`
    is one location: HYSPLIT joins its two heights into one line source.
    """
    return len({(lat, lon) for lat, lon, _ in receptor.coords()})


def check_release(workdir: Path, receptor: Receptor, config: HysplitConfig) -> None:
    """
    Raise when HYSPLIT stopped releasing before every location of *receptor* got its particles.

    HYSPLIT fills the locations in order and stops at ``MAXPAR``, writing
    ``exceeding puff limit`` to its ``WARNING`` file, so the last
    locations get fewer particles or none. From one location that is
    harmless (:data:`~stilt.transport.hysplit.failures.PARTICLE_LIMIT_PHRASE`).

    Raises
    ------
    SimulationError
        With ``reason`` ``PARTICLE_LIMIT``, when *receptor* has several
        locations and the ``WARNING`` file in *workdir* says HYSPLIT
        reached ``MAXPAR``.
    """
    locations = release_locations(receptor)
    warning = workdir / WARNING_FILE
    if locations < 2 or not warning.exists():
        return
    if PARTICLE_LIMIT_PHRASE in warning.read_text(encoding="utf-8", errors="replace"):
        fix = (
            "Raise maxpar or leave it unset." if config.maxpar else "Set maxpar higher."
        )
        raise SimulationError(
            f"HYSPLIT stopped releasing at maxpar = {config.maxpar_for(locations)} "
            f"before each of the receptor's {locations} locations got its share of "
            f"numpar = {config.numpar}. {fix}",
            reason=FailureReason.PARTICLE_LIMIT,
        )


#: How far inside the met's domain, in grid cells, a particle at the last time
#: step of a run that stopped early must be for :func:`check_met_stop` to fail
#: the run. Measured with the test met and stored HRRR runs: particles that
#: left the domain ended 1 to 2 cells from its edge, and in runs the met
#: stopped, the particle furthest inside was 34 cells in or more.
MET_STOP_CELLS = 5


def cells_from_edge(
    lon: Any, lat: Any, zagl: Any, grid: "Grid", z_top: float, levels: int
) -> np.ndarray:
    """
    Return how far each position is from the nearest edge of the met's domain, in grid cells.

    The edges are the four sides of *grid* and the domain top, *z_top*
    meters above ground. A cell at the top is ``z_top / levels`` deep: the
    met's *levels* spread evenly up to the top. A global grid's columns
    wrap around in longitude, so it has no east or west side. A position
    outside the grid has a negative distance.

    Parameters
    ----------
    lon, lat, zagl : array-like
        Positions: degrees, and meters above ground.
    grid : arlmet.Grid
        The met's horizontal grid.
    z_top : float
        The top of the domain, in meters above ground.
    levels : int
        The number of vertical levels in the met.

    Returns
    -------
    numpy.ndarray
        The distance of each position, in grid cells.
    """
    x, y = grid.fractional_indices(np.asarray(lon), np.asarray(lat))
    edges = [y, grid.ny - 1 - y]
    if not grid.wraps_lon:
        edges += [x, grid.nx - 1 - x]
    edges.append((z_top - np.asarray(zagl, dtype=float)) / (z_top / levels))
    return np.minimum.reduce(edges)


def check_met_stop(
    particles: pd.DataFrame, n_hours: int, grid: "Grid", levels: int, z_top: float
) -> None:
    """
    Raise when a run stopped before ``n_hours`` with particles well inside the met's domain.

    Particles that leave the met's domain end at its edge, and a run whose
    particles all left early is complete. When the met runs out, HYSPLIT
    can stop every particle where it is, write no warning, and report a
    complete run. So a run that stopped before ``n_hours`` fails when any
    particle at its last time step is more than :data:`MET_STOP_CELLS`
    grid cells from every edge of the domain (:func:`cells_from_edge`):
    the sides of the met grid and the top.

    The particle table must hold a row for every time step (``outdt: 0``).
    With fewer rows, the last row of a particle that left can be far
    inside the domain. A table without the ``particle``, ``age``, ``lon``,
    ``lat``, and ``zagl`` columns is not checked.

    Parameters
    ----------
    particles : pandas.DataFrame
        The particles of one run.
    n_hours : int
        The length of the run asked for, negative for a backward run.
    grid : arlmet.Grid
        The horizontal grid of the met the run read.
    levels : int
        The number of vertical levels in that met.
    z_top : float
        The top of the domain, in meters above ground.

    Raises
    ------
    SimulationError
        With ``reason`` ``MET_COVERAGE``, when the particles stopped
        before ``n_hours`` and one at the last time step is more than
        :data:`MET_STOP_CELLS` grid cells inside the domain.
    """
    if particles.empty or not {"particle", "age", "lon", "lat", "zagl"} <= set(
        particles.columns
    ):
        return
    reach = particles["age"].abs()
    stop = reach.max()
    if stop >= abs(n_hours) * 60:
        return
    last = particles[reach == stop]
    inside = cells_from_edge(
        last["lon"], last["lat"], last["zagl"], grid, z_top, levels
    ).max()
    if inside > MET_STOP_CELLS:
        share = len(last) / particles["particle"].nunique()
        raise SimulationError(
            f"The particles stopped {stop / 60:.1f} h into the {abs(n_hours)} h "
            f"run, up to {inside:.1f} grid cells inside the met's domain, not "
            f"at its edge: the met ran out ({share:.0%} of the particles were "
            "still in the air).",
            reason=FailureReason.MET_COVERAGE,
        )


def ziscale_factors(params: HysplitConfig) -> list[float] | None:
    """
    Return the hourly mixed-layer factors for ``ZICONTROL``, or ``None`` when unscaled.

    A single ``ziscale`` value is repeated for every hour of the run and a
    list is used as given. Factors that are all 1.0 give ``None``.
    """
    values = params.hourly_ziscale()
    if all(v == 1.0 for v in values):
        return None
    return values


def zicontroltf(params: HysplitConfig) -> int:
    """Return HYSPLIT's ``ZICONTROLTF`` flag: 1 when ``ziscale`` scales the mixed layer."""
    return int(ziscale_factors(params) is not None)


def winderr(params: HysplitConfig) -> list[float] | None:
    """Return the ``WINDERR`` values, in the file's order, or ``None`` when unset."""
    values = [getattr(params, name) for name in WIND_ERROR_SETTINGS]
    return None if values[0] is None else values


def zierr(params: HysplitConfig) -> list[float] | None:
    """Return the ``ZIERR`` values, in the file's order, or ``None`` when unset."""
    values = [getattr(params, name) for name in ZI_ERROR_SETTINGS]
    return None if values[0] is None else values


def winderrtf(params: HysplitConfig) -> int:
    """Return HYSPLIT's ``WINDERRTF`` flag: 1 for wind errors, 2 for mixed-layer errors, 3 for both."""
    return (winderr(params) is not None) + 2 * (zierr(params) is not None)


def _write_values(path: Path, values: list[float] | None) -> None:
    """Write one value per line to ``path``; nothing is written when ``values`` is ``None``."""
    if values is not None:
        path.write_text("\n".join(str(v) for v in values) + "\n", encoding="utf-8")


def write_inputs(
    workdir: str | Path,
    receptor: Receptor,
    config: HysplitConfig,
    met_files: Sequence[Path],
) -> None:
    """
    Write everything ``hycs_std`` reads for one receptor into *workdir*.

    Links ``hycs_std`` (from ``config.exe_dir``, or the bundled build) and
    HYSPLIT's data tables (the bundled ones, replaced by any in
    ``config.data_dir``), and writes ``CONTROL`` and ``SETUP.CFG``, plus
    ``ZICONTROL``, ``WINDERR``, and ``ZIERR`` when the settings call for
    them. *workdir* is created if needed, and should be empty.

    Parameters
    ----------
    workdir : str or Path
        Directory to write into.
    receptor : Receptor
        Where and when particles are released. Its ``altitude_ref`` sets
        HYSPLIT's ``KMSL``.
    config : HysplitConfig
        Transport and error settings.
    met_files : sequence of Path
        Meteorology files, in the order HYSPLIT should read them.

    Raises
    ------
    HYSPLITNotFoundError
        If there is no ``hycs_std`` to link.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # Link the binary and the data tables, as STILT-R does. Only hycs_std is
    # taken from exe_dir: a custom build directory usually holds a whole
    # HYSPLIT exec/ tree.
    exe_dir = Path(config.exe_dir) if config.exe_dir is not None else _bundled_exe_dir()
    exe = exe_dir / HYCS_STD_FILE
    if not exe.is_file():
        raise HYSPLITNotFoundError(
            f"No {HYCS_STD_FILE!r} executable in {exe_dir}. Check exe_dir."
        )
    links = {HYCS_STD_FILE: exe}
    links.update({f.name: f for f in _bundled_data_dir().iterdir()})
    if config.data_dir is not None:  # its tables replace the bundled ones
        links.update(
            {f.name: f for f in Path(config.data_dir).iterdir() if f.is_file()}
        )
    for name, target in links.items():
        (workdir / name).symlink_to(target.resolve())

    ControlFile(
        receptor=receptor,
        n_hours=config.n_hours,
        emisshrs=config.emisshrs,
        w_option=config.w_option,
        z_top=config.z_top,
        met_files=list(met_files),
    ).write(workdir / CONTROL_FILE)

    entries = setup_entries(config)
    # HYSPLIT's KMSL: 0 for heights above ground, 1 above sea level.
    entries["kmsl"] = 1 if receptor.altitude_ref == "msl" else 0
    entries["maxpar"] = config.maxpar_for(release_locations(receptor))
    entries["ivmax"] = len(config.varsiwant)  # number of output variables
    entries["winderrtf"] = winderrtf(config)
    setup = NameList("SETUP")
    setup.update(entries)
    setup.write(workdir / SETUP_FILE)

    factors = ziscale_factors(config)
    if factors is not None:
        (workdir / ZICONTROL_FILE).write_text(
            "\n".join([str(len(factors)), *(str(v) for v in factors)]) + "\n"
        )
    _write_values(workdir / WINDERR_FILE, winderr(config))
    _write_values(workdir / ZIERR_FILE, zierr(config))


def run_hycs_std(workdir: Path, timeout: int | None = None) -> None:
    """
    Run the ``hycs_std`` in *workdir*, writing its output to ``stilt.log``.

    *workdir* holds the inputs :func:`write_inputs` wrote. With *timeout*
    seconds the run is stopped when it takes longer; without it, it may run
    as long as HYSPLIT does.

    Raises
    ------
    SimulationError
        With ``reason`` ``TIMEOUT`` when the run exceeded *timeout*
        seconds, or the :class:`FailureReason` of a known failure message in
        the log or HYSPLIT's ``WARNING`` file, such as ``no more
        meteorology`` (``MET_COVERAGE``). Particles that all left the met's
        domain are not a failure.
    """
    log_path = workdir / LOG_FILE
    with (
        log_path.open("w", encoding="utf-8") as handle,
        subprocess.Popen(
            [str(workdir / HYCS_STD_FILE)],
            cwd=workdir,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        ) as proc,
    ):
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired as e:
            _terminate(proc)
            raise SimulationError(
                f"HYSPLIT ran longer than the {timeout} s timeout.",
                reason=FailureReason.TIMEOUT,
            ) from e
    messages = log_path.read_text(encoding="utf-8", errors="replace")
    warnings = workdir / WARNING_FILE
    if warnings.exists():
        messages += warnings.read_text(encoding="utf-8", errors="replace")
    found = failure_in(messages)
    if found is not None:
        reason, line = found
        raise SimulationError(f"HYSPLIT: {line}", reason=reason)


def _terminate(proc: subprocess.Popen[Any]) -> None:
    """Stop a HYSPLIT process group with SIGTERM, then SIGKILL if it does not exit."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=3)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.wait()
