"""
Write HYSPLIT's input files, run ``hycs_std``, and read the particles it writes.

:func:`write_inputs` fills a working directory with everything
``hycs_std`` reads, and :func:`read_particle_dat` reads the particle file
it writes. :class:`~stilt.transport.hysplit.HysplitModel` runs the two
around ``hycs_std`` in one directory; call them yourself to look at the
input files of a run, or to read a ``PARTICLE_STILT.DAT`` from elsewhere.
"""

import datetime as dt
import os
import shutil
import signal
import subprocess
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from stilt.exceptions import HYSPLITNotFoundError, SimulationError
from stilt.transport.hysplit.config import (
    FIXED_SETUP,
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
#: HYSPLIT's folders, as its download unpacks: the programs and the data tables.
EXEC_DIR = "exec"
BDYFILES_DIR = "bdyfiles"
#: HYSPLIT's data tables, which a run links into its workdir.
DATA_TABLES = ("ASCDATA.CFG", "LANDUSE.ASC", "ROUGLEN.ASC", "TERRAIN.ASC")
PARTICLE_STILT_FILE = "PARTICLE_STILT.DAT"
WINDERR_FILE = "WINDERR"
ZIERR_FILE = "ZIERR"
ZICONTROL_FILE = "ZICONTROL"
LOG_FILE = "stilt.log"
#: HYSPLIT's own file of warnings, such as the met running out.
WARNING_FILE = "WARNING"
#: HYSPLIT's own file of progress and error messages.
MESSAGE_FILE = "MESSAGE"
#: How many lines of ``MESSAGE`` a run that ended badly adds to its log.
MESSAGE_TAIL_LINES = 20

#: The settings ``CONTROL`` takes; :class:`ControlFile` writes them.
CONTROL_SETTINGS = ("n_hours", "emisshrs", "w_option", "z_top")

#: Settings PYSTILT uses itself, in no HYSPLIT input file.
PYSTILT_SETTINGS = ("directory",)

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


def find_hycs_std(directory: str | Path | None = None) -> Path:
    """
    Return the ``hycs_std`` a run uses: ``exec/hycs_std`` under *directory*, else the first on ``PATH``.

    PYSTILT does not include HYSPLIT, so the build is yours. *directory* is
    the HYSPLIT folder as unpacked (``directory`` under ``models: hysplit:``
    in ``config.yaml``), which holds ``exec/`` and ``bdyfiles/``.

    Raises
    ------
    HYSPLITNotFoundError
        If *directory* has no ``exec/hycs_std``, or its ``hycs_std`` is not
        executable; without *directory*, if no ``hycs_std`` is on ``PATH``.
    """
    if directory is not None:
        exe = Path(directory) / EXEC_DIR / HYCS_STD_FILE
        if not exe.is_file():
            raise HYSPLITNotFoundError(
                f"No {EXEC_DIR}/{HYCS_STD_FILE} in {directory}. Set directory "
                "(under models: hysplit: in config.yaml) to the HYSPLIT folder "
                f"as unpacked, the one that holds {EXEC_DIR}/ and {BDYFILES_DIR}/."
            )
        if not os.access(exe, os.X_OK):
            raise HYSPLITNotFoundError(
                f"{exe} is not executable, so it cannot run. A HYSPLIT folder "
                "unpacked on Windows, or copied from there, loses that permission; "
                f"make it executable with: chmod +x {exe}"
            )
        return exe
    found = shutil.which(HYCS_STD_FILE)
    if found is None:
        raise HYSPLITNotFoundError(
            f"No {HYCS_STD_FILE} on PATH. PYSTILT does not include HYSPLIT: get "
            "it (the installation guide says where), then set directory (under "
            "models: hysplit: in config.yaml) to the HYSPLIT folder, or put the "
            f"folder that holds an executable {HYCS_STD_FILE} on PATH."
        )
    return Path(found)


def data_tables(directory: str | Path | None = None) -> dict[str, Path | bytes]:
    """
    Return HYSPLIT's data tables a run uses, by name: a file to link, or the text to write.

    They come from ``bdyfiles/`` under *directory*, or, without
    *directory*, from a ``bdyfiles/`` folder beside the folder of the
    ``hycs_std`` on ``PATH``, as HYSPLIT's download lays them out. PYSTILT
    includes none. HYSPLIT's own ``ASCDATA.CFG`` says the tables are in
    ``../bdyfiles/``; a run links them into its workdir, so its
    ``ASCDATA.CFG`` is returned as text with that line, the sixth, set to
    ``'./'`` and the rest kept.

    Raises
    ------
    HYSPLITNotFoundError
        If the folder is missing or lacks one of the four tables, or,
        without *directory*, no ``hycs_std`` is on ``PATH``.
    """
    if directory is not None:
        folder = Path(directory) / BDYFILES_DIR
    else:
        folder = find_hycs_std().resolve().parent.parent / BDYFILES_DIR
    missing = [name for name in DATA_TABLES if not (folder / name).is_file()]
    if missing:
        raise HYSPLITNotFoundError(
            f"No {', '.join(missing)} in {folder}. PYSTILT includes no HYSPLIT "
            f"data tables: they are in the {BDYFILES_DIR}/ folder of HYSPLIT's "
            f"download, beside its {EXEC_DIR}/. Set directory (under models: "
            "hysplit: in config.yaml) to a HYSPLIT folder that has both."
        )
    tables: dict[str, Path | bytes] = {name: folder / name for name in DATA_TABLES}
    tables["ASCDATA.CFG"] = _tables_here(folder / "ASCDATA.CFG")
    return tables


#: The sixth line of ``ASCDATA.CFG``, where HYSPLIT reads the tables: the
#: workdir, where a run links them.
_TABLES_HERE = b"'./'           directory location of data files\n"


def _tables_here(path: Path) -> bytes:
    """Return the ``ASCDATA.CFG`` at *path* with its sixth line, where the tables are, as ``'./'``."""
    lines = path.read_bytes().splitlines(keepends=True)
    if len(lines) >= 6:
        lines[5] = _TABLES_HERE
    return b"".join(lines)


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
        release heights are added after the run
        (:func:`stilt.transport.run_model`), and the near-field correction
        when a footprint is made (:func:`stilt.calc_footprint`).

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

    The entries of :data:`FIXED_SETUP` come first, then every setting not
    in :data:`NOT_IN_SETUP`, in declaration order, under their own names.
    The driver adds
    ``KMSL``, ``IVMAX``, and ``WINDERRTF``, and ``MAXPAR`` when ``maxpar``
    is unset, which depend on the release locations or follow from other
    settings.
    """
    entries = dict(FIXED_SETUP)
    entries.update(
        (name, getattr(params, name))
        for name in HysplitConfig.model_fields
        if name not in NOT_IN_SETUP and getattr(params, name) is not None
    )
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


def release_locations(locations: Sequence[tuple[float, float, float]]) -> int:
    """
    Return the number of horizontal locations among the release *locations*.

    HYSPLIT divides ``numpar`` among them. Two release locations at one
    latitude and longitude are one: HYSPLIT joins their heights into one
    line source.
    """
    return len({(lat, lon) for lat, lon, _ in locations})


def check_release(
    workdir: Path,
    locations: Sequence[tuple[float, float, float]],
    config: HysplitConfig,
) -> None:
    """
    Raise when HYSPLIT stopped releasing before every one of *locations* got its particles.

    HYSPLIT fills the locations in order and stops at ``MAXPAR``, writing
    ``exceeding puff limit`` to its ``WARNING`` file, so the last
    locations get fewer particles or none. From one location that is
    harmless (:data:`~stilt.transport.hysplit.failures.PARTICLE_LIMIT_PHRASE`).

    Raises
    ------
    SimulationError
        With ``reason`` ``PARTICLE_LIMIT``, when *locations* hold several
        horizontal locations and the ``WARNING`` file in *workdir* says
        HYSPLIT reached ``MAXPAR``.
    """
    count = release_locations(locations)
    warning = workdir / WARNING_FILE
    if count < 2 or not warning.exists():
        return
    if PARTICLE_LIMIT_PHRASE in warning.read_text(encoding="utf-8", errors="replace"):
        fix = (
            "Raise maxpar or leave it unset." if config.maxpar else "Set maxpar higher."
        )
        raise SimulationError(
            f"HYSPLIT stopped releasing at maxpar = {config.maxpar_for(count)} "
            f"before each of the {count} release locations got its share of "
            f"numpar = {config.numpar}. {fix}",
            reason=FailureReason.PARTICLE_LIMIT,
        )


#: How far inside the met's domain, in grid cells, a particle at the last time
#: step of a run that stopped early must be for :func:`check_met_stop` to fail
#: the run. Measured on HRRR (3 km), NAM (12 km) and NARR (32 km) grids and
#: crops of them: particles that left the domain ended 1 to 2 cells from its
#: edge, and in runs the met stopped on a domain a hundred cells across or
#: more, the particle furthest inside was 34 cells in or more.
MET_STOP_CELLS = 5


def cells_from_edge(lon: Any, lat: Any, grid: "Grid") -> np.ndarray:
    """
    Return how far each position is from the nearest side of the met grid, in grid cells.

    A global grid's columns wrap around in longitude, so it has no east or
    west side. A position outside the grid has a negative distance. The
    domain top is not an edge: HYSPLIT keeps particles from rising above it.

    Parameters
    ----------
    lon, lat : array-like
        Positions, in degrees.
    grid : arlmet.Grid
        The met's horizontal grid.

    Returns
    -------
    numpy.ndarray
        The distance of each position, in grid cells.
    """
    x, y = grid.fractional_indices(np.asarray(lon), np.asarray(lat))
    edges = [y, grid.ny - 1 - y]
    if not grid.wraps_lon:
        edges += [x, grid.nx - 1 - x]
    return np.minimum.reduce(edges)


def check_met_stop(particles: pd.DataFrame, n_hours: int, grid: "Grid") -> None:
    """
    Raise when a run stopped before ``n_hours`` with particles well inside the met's domain.

    Particles that leave the met's domain end at its edge, and a run whose
    particles all left early is complete. When the met runs out, HYSPLIT
    can stop every particle where it is, write no warning, and report a
    complete run. So a run that stopped before ``n_hours`` fails when any
    particle at its last time step is more than :data:`MET_STOP_CELLS`
    grid cells from every side of the met grid (:func:`cells_from_edge`).

    On a domain only a few dozen cells across, a met stop late in a run,
    when nearly every particle has already left, can pass as a domain exit.
    The footprint lost in such a run is small.

    The particle table must hold a row for every time step (``outdt: 0``).
    With fewer rows, the last row of a particle that left can be far
    inside the domain. A table without the ``particle``, ``age``, ``lon``,
    and ``lat`` columns is not checked.

    Parameters
    ----------
    particles : pandas.DataFrame
        The particles of one run.
    n_hours : int
        The length of the run asked for, negative for a backward run.
    grid : arlmet.Grid
        The horizontal grid of the met the run read.

    Raises
    ------
    SimulationError
        With ``reason`` ``MET_COVERAGE``, when the particles stopped
        before ``n_hours`` and one at the last time step is more than
        :data:`MET_STOP_CELLS` grid cells inside the met grid.
    """
    if particles.empty or not {"particle", "age", "lon", "lat"} <= set(
        particles.columns
    ):
        return
    reach = particles["age"].abs()
    stop = reach.max()
    if stop >= abs(n_hours) * 60:
        return
    last = particles[reach == stop]
    inside = cells_from_edge(last["lon"], last["lat"], grid).max()
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
    release_time: dt.datetime,
    locations: Sequence[tuple[float, float, float]],
    config: HysplitConfig,
    met_files: Sequence[Path],
    kmsl: int = 0,
) -> None:
    """
    Write everything ``hycs_std`` reads for one run into *workdir*.

    Links ``hycs_std`` (from ``config.directory``, or else ``PATH``:
    :func:`find_hycs_std`) and HYSPLIT's data tables
    (:func:`data_tables`), and writes ``CONTROL`` and ``SETUP.CFG``, plus
    ``ZICONTROL``, ``WINDERR``, and ``ZIERR`` when the settings call for
    them. *workdir* is created if needed, and should be empty.

    Parameters
    ----------
    workdir : str or Path
        Directory to write into.
    release_time : datetime
        The release time, in UTC.
    locations : sequence of tuple of float
        The release locations, each ``(latitude, longitude, height)`` in
        degrees and meters, in the order ``CONTROL`` lists them.
    config : HysplitConfig
        Transport and error settings.
    met_files : sequence of Path
        Meteorology files, in the order HYSPLIT should read them.
    kmsl : {0, 1}, default 0
        HYSPLIT's ``KMSL``: 0 when the release heights are above ground, 1
        when they are above sea level.

    Raises
    ------
    HYSPLITNotFoundError
        If there is no ``hycs_std`` to link (:func:`find_hycs_std`).
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # Link the binary and the data tables, as STILT-R does; only hycs_std is
    # taken from exec/, which holds dozens of other programs.
    links = {HYCS_STD_FILE: find_hycs_std(config.directory)}
    for name, table in data_tables(config.directory).items():
        if isinstance(table, bytes):
            (workdir / name).write_bytes(table)
        else:
            links[name] = table
    for name, target in links.items():
        (workdir / name).symlink_to(target.resolve())

    ControlFile(
        release_time=release_time,
        locations=list(locations),
        n_hours=config.n_hours,
        emisshrs=config.emisshrs,
        w_option=config.w_option,
        z_top=config.z_top,
        met_files=list(met_files),
    ).write(workdir / CONTROL_FILE)

    entries = setup_entries(config)
    entries["kmsl"] = kmsl
    entries["maxpar"] = config.maxpar_for(release_locations(locations))
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
    as long as HYSPLIT does. If this process is interrupted while it waits
    (``KeyboardInterrupt`` or ``SystemExit``), the run is stopped too.

    Raises
    ------
    SimulationError
        With ``reason`` ``TIMEOUT`` when the run exceeded *timeout*
        seconds, the :class:`FailureReason` of a known failure message in
        the log or HYSPLIT's ``WARNING`` file, such as ``no more
        meteorology`` (``MET_COVERAGE``), or ``EXIT_STATUS`` when
        ``hycs_std`` ended with a non-zero exit status or was killed by a
        signal, such as the kernel's out-of-memory killer, and the log has
        no known message. Its log then ends with the last lines of
        HYSPLIT's ``MESSAGE`` file. Particles that all left the met's
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
        except BaseException:
            # Ctrl-C, or a worker told to stop: HYSPLIT runs in a session of
            # its own, so no signal reached it and it would run on, unseen.
            _terminate(proc)
            raise
    messages = log_path.read_text(encoding="utf-8", errors="replace")
    warnings = workdir / WARNING_FILE
    if warnings.exists():
        messages += warnings.read_text(encoding="utf-8", errors="replace")
    found = failure_in(messages)
    if found is not None:
        reason, line = found
        raise SimulationError(f"HYSPLIT: {line}", reason=reason)
    if proc.returncode != 0:
        raise _exit_status_error(workdir, proc.returncode)


def _exit_status_error(workdir: Path, status: int) -> SimulationError:
    """
    Return the error for a ``hycs_std`` that ended with *status* but no known message.

    *status* is ``subprocess``'s: negative for the signal that killed the
    process. The tail of HYSPLIT's ``MESSAGE`` file is added to the log,
    where it says how far the run got.
    """
    if status < 0:
        try:
            name = signal.Signals(-status).name
        except ValueError:  # a real-time signal has no name
            name = f"signal {-status}"
        how = f"was killed by {name}"
    else:
        how = f"exited with status {status}"
    message = f"HYSPLIT {how} before it finished."
    path = workdir / MESSAGE_FILE
    if path.exists():
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(lines[-MESSAGE_TAIL_LINES:])
        with (workdir / LOG_FILE).open("a", encoding="utf-8") as log:
            log.write(f"\nPYSTILT: HYSPLIT {how}. The end of {MESSAGE_FILE}:\n{tail}\n")
    return SimulationError(message, reason=FailureReason.EXIT_STATUS)


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
