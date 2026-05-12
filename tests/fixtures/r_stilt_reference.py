"""
Single source of truth for the seeded R-STILT reference cases.

Defines all canonical scenarios used by the PYSTILT fidelity integration tests.
Each scenario is represented by a :class:`ReferenceScenario` instance that knows
how to build the PYSTILT objects it needs.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Shared reference constants
# ---------------------------------------------------------------------------

REFERENCE_MET = "hrrr"
REFERENCE_TIME = dt.datetime(2021, 1, 15, 6, 0)
REFERENCE_SUMMER_TIME = dt.datetime(2021, 7, 15, 6, 0)
REFERENCE_LONGITUDE = -112.0
REFERENCE_LATITUDE = 40.5
REFERENCE_ALTITUDE = 5.0

REFERENCE_N_HOURS = -6
REFERENCE_NUMPAR = 1000
REFERENCE_KRAND = 2
REFERENCE_SEED = 42
REFERENCE_MET_FILE_FORMAT = "%Y%m%d_%H"
REFERENCE_MET_FILE_INTERVAL_HOURS = 6

REFERENCE_XMIN = -113.0
REFERENCE_XMAX = -111.0
REFERENCE_YMIN = 39.5
REFERENCE_YMAX = 41.5
REFERENCE_XRES = 0.01
REFERENCE_YRES = 0.01

REFERENCE_VARSIWANT = [
    "time",
    "indx",
    "long",
    "lati",
    "zagl",
    "foot",
    "mlht",
    "dens",
    "samt",
    "sigw",
    "tlgr",
]

# Trajectory columns present when hnf_plume=True
_TRAJ_COLS_WITH_HNF: tuple[str, ...] = (
    "time",
    "indx",
    "long",
    "lati",
    "zagl",
    "foot",
    "mlht",
    "dens",
    "samt",
    "sigw",
    "tlgr",
    "foot_no_hnf_dilution",
)

# Trajectory columns present when hnf_plume=False (no foot_no_hnf_dilution column)
_TRAJ_COLS_NO_HNF: tuple[str, ...] = (
    "time",
    "indx",
    "long",
    "lati",
    "zagl",
    "foot",
    "mlht",
    "dens",
    "samt",
    "sigw",
    "tlgr",
)


# ---------------------------------------------------------------------------
# ReferenceScenario dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceScenario:
    """
    All parameters needed to run a seeded PYSTILT fidelity case and compare
    against R-STILT run live.

    Attributes
    ----------
    name
        Short slug used as a scenario identifier (e.g. ``"point"``).
    description
        Human-readable label shown in pytest output.
    receptor_type
        ``"point"`` | ``"column"`` | ``"multipoint"``.
    longitude, latitude
        Scalar for point/column receptors; tuple for multipoint.
    altitude
        Scalar for point; ``(bottom, top)`` tuple for column;
        tuple of per-point altitudes for multipoint.
    n_hours, numpar, krand, seed
        Transport controls passed to :class:`~stilt.config.ModelConfig`.
    hnf_plume
        Whether to apply hyper-near-field plume dilution.
    smooth_factor
        Gaussian footprint smoothing scale (1.0 = default).
    time_integrate
        If True, collapse footprint over all time steps.
    xmin, xmax, ymin, ymax, xres, yres
        Output footprint grid.
    compare_columns
        Trajectory column names used in comparisons.
    """

    name: str
    description: str
    receptor_type: Literal["point", "column", "multipoint"]
    longitude: float | tuple[float, ...]
    latitude: float | tuple[float, ...]
    altitude: float | tuple[float, ...]
    n_hours: int
    numpar: int
    krand: int
    seed: int
    hnf_plume: bool
    smooth_factor: float
    time_integrate: bool
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    xres: float
    yres: float
    compare_columns: tuple[str, ...]
    # Optional — override per-scenario; defaults match the shared reference constants
    time: dt.datetime = field(default_factory=lambda: REFERENCE_TIME)
    met_file_format: str = field(default_factory=lambda: REFERENCE_MET_FILE_FORMAT)
    met_file_tres: str = "6h"
    # If set, this scenario's trajectory is physically identical to the named scenario.
    # The fixture skips the R+HYSPLIT trajectory run and test_trajectory_matches_r skips.
    shares_trajectory_with: str | None = field(default=None)

    # ------------------------------------------------------------------
    # PYSTILT object factories (lazy imports to avoid circular load)
    # ------------------------------------------------------------------

    def make_receptor(self):
        """Return the appropriate Receptor subtype for this scenario."""
        from stilt.receptors import ColumnReceptor, MultiPointReceptor, PointReceptor

        if self.receptor_type == "point":
            return PointReceptor(
                self.time,
                float(self.longitude),  # type: ignore[arg-type]
                float(self.latitude),  # type: ignore[arg-type]
                float(self.altitude),  # type: ignore[arg-type]
            )
        if self.receptor_type == "column":
            bottom, top = self.altitude  # type: ignore[misc]
            return ColumnReceptor(
                self.time,
                float(self.longitude),  # type: ignore[arg-type]
                float(self.latitude),  # type: ignore[arg-type]
                bottom=float(bottom),
                top=float(top),
            )
        return MultiPointReceptor(
            self.time,
            longitudes=list(self.longitude),  # type: ignore[arg-type]
            latitudes=list(self.latitude),  # type: ignore[arg-type]
            altitudes=list(self.altitude),  # type: ignore[arg-type]
        )

    def make_grid(self):
        """Return the :class:`~stilt.config.Grid` for this scenario."""
        from stilt.config import Grid

        return Grid(
            xmin=self.xmin,
            xmax=self.xmax,
            ymin=self.ymin,
            ymax=self.ymax,
            xres=self.xres,
            yres=self.yres,
        )

    def make_footprint_config(self):
        """Return the :class:`~stilt.config.FootprintConfig` for this scenario."""
        from stilt.config import FootprintConfig

        return FootprintConfig(
            grid=self.make_grid(),
            smooth_factor=self.smooth_factor,
            time_integrate=self.time_integrate,
        )

    def make_model_config(self, met_dir):
        """Return a :class:`~stilt.config.ModelConfig` configured for this scenario."""
        from stilt.config import ModelConfig

        return ModelConfig.model_validate(
            {
                "mets": {
                    REFERENCE_MET: {
                        "directory": met_dir,
                        "file_format": self.met_file_format,
                        "file_tres": self.met_file_tres,
                    }
                },
                "n_hours": self.n_hours,
                "numpar": self.numpar,
                "krand": self.krand,
                "seed": self.seed,
                "hnf_plume": self.hnf_plume,
                "varsiwant": REFERENCE_VARSIWANT,
                "footprints": {"default": self.make_footprint_config()},
            }
        )

    def py_sim_id(self) -> str:
        """Return the PYSTILT simulation ID (``met_YYYYMMDDHHMM_location``)."""
        from stilt.simulation import SimID

        return str(SimID.from_parts(REFERENCE_MET, self.make_receptor()))

    def r_sim_id(self) -> str:
        """Return the R-STILT simulation ID (receptor part only, no met prefix)."""
        from stilt.simulation import SimID

        return str(SimID.from_parts(REFERENCE_MET, self.make_receptor()).receptor)


# ---------------------------------------------------------------------------
# Canonical scenarios
# ---------------------------------------------------------------------------

#: Baseline single-point WBB receptor with default footprint settings.
POINT = ReferenceScenario(
    name="point",
    description="Single WBB point receptor, 6 h backward, default smooth_factor, hnf_plume=True",
    receptor_type="point",
    longitude=REFERENCE_LONGITUDE,
    latitude=REFERENCE_LATITUDE,
    altitude=REFERENCE_ALTITUDE,
    n_hours=REFERENCE_N_HOURS,
    numpar=REFERENCE_NUMPAR,
    krand=REFERENCE_KRAND,
    seed=REFERENCE_SEED,
    hnf_plume=True,
    smooth_factor=1.0,
    time_integrate=False,
    xmin=REFERENCE_XMIN,
    xmax=REFERENCE_XMAX,
    ymin=REFERENCE_YMIN,
    ymax=REFERENCE_YMAX,
    xres=REFERENCE_XRES,
    yres=REFERENCE_YRES,
    compare_columns=_TRAJ_COLS_WITH_HNF,
)

#: Column receptor spanning 0–1000 m AGL at the WBB location.
COLUMN = ReferenceScenario(
    name="column",
    description="Column receptor (0–1000 m AGL), 6 h backward, hnf_plume=True",
    receptor_type="column",
    longitude=-112.0,
    latitude=40.5,
    altitude=(0.0, 1000.0),  # (bottom, top)
    n_hours=REFERENCE_N_HOURS,
    numpar=REFERENCE_NUMPAR,
    krand=REFERENCE_KRAND,
    seed=REFERENCE_SEED,
    hnf_plume=True,
    smooth_factor=1.0,
    time_integrate=False,
    xmin=REFERENCE_XMIN,
    xmax=REFERENCE_XMAX,
    ymin=REFERENCE_YMIN,
    ymax=REFERENCE_YMAX,
    xres=REFERENCE_XRES,
    yres=REFERENCE_YRES,
    compare_columns=_TRAJ_COLS_WITH_HNF,
)

#: Three-location multipoint receptor spread across the SLV.
MULTIPOINT = ReferenceScenario(
    name="multipoint",
    description="Three-location multipoint receptor, 6 h backward, hnf_plume=True",
    receptor_type="multipoint",
    longitude=(-112.0, -111.5, -111.0),
    latitude=(40.5, 41.0, 41.5),
    altitude=(0.0, 500.0, 1000.0),
    n_hours=REFERENCE_N_HOURS,
    numpar=REFERENCE_NUMPAR,
    krand=REFERENCE_KRAND,
    seed=REFERENCE_SEED,
    hnf_plume=True,
    smooth_factor=1.0,
    time_integrate=False,
    xmin=-113.0,
    xmax=-110.5,
    ymin=39.5,
    ymax=42.0,
    xres=REFERENCE_XRES,
    yres=REFERENCE_YRES,
    compare_columns=_TRAJ_COLS_WITH_HNF,
)

#: Point receptor with hyper-near-field plume dilution disabled.
NO_HNF = ReferenceScenario(
    name="no_hnf",
    description="Point receptor, hnf_plume=False — no near-field plume dilution correction",
    receptor_type="point",
    longitude=REFERENCE_LONGITUDE,
    latitude=REFERENCE_LATITUDE,
    altitude=REFERENCE_ALTITUDE,
    n_hours=REFERENCE_N_HOURS,
    numpar=REFERENCE_NUMPAR,
    krand=REFERENCE_KRAND,
    seed=REFERENCE_SEED,
    hnf_plume=False,
    smooth_factor=1.0,
    time_integrate=False,
    xmin=REFERENCE_XMIN,
    xmax=REFERENCE_XMAX,
    ymin=REFERENCE_YMIN,
    ymax=REFERENCE_YMAX,
    xres=REFERENCE_XRES,
    yres=REFERENCE_YRES,
    compare_columns=_TRAJ_COLS_NO_HNF,
)

#: Point receptor with Gaussian footprint smoothing at half the default scale.
SMOOTH = ReferenceScenario(
    name="smooth",
    description="Point receptor, smooth_factor=0.5 — reduced Gaussian kernel smoothing",
    receptor_type="point",
    longitude=REFERENCE_LONGITUDE,
    latitude=REFERENCE_LATITUDE,
    altitude=REFERENCE_ALTITUDE,
    n_hours=REFERENCE_N_HOURS,
    numpar=REFERENCE_NUMPAR,
    krand=REFERENCE_KRAND,
    seed=REFERENCE_SEED,
    hnf_plume=True,
    smooth_factor=0.5,
    time_integrate=False,
    xmin=REFERENCE_XMIN,
    xmax=REFERENCE_XMAX,
    ymin=REFERENCE_YMIN,
    ymax=REFERENCE_YMAX,
    xres=REFERENCE_XRES,
    yres=REFERENCE_YRES,
    compare_columns=_TRAJ_COLS_WITH_HNF,
    shares_trajectory_with="point",
)

#: Point receptor with all time steps collapsed into a single 2-D footprint layer.
TIME_INTEGRATE = ReferenceScenario(
    name="time_integrate",
    description="Point receptor, time_integrate=True — single 2-D footprint layer",
    receptor_type="point",
    longitude=REFERENCE_LONGITUDE,
    latitude=REFERENCE_LATITUDE,
    altitude=REFERENCE_ALTITUDE,
    n_hours=REFERENCE_N_HOURS,
    numpar=REFERENCE_NUMPAR,
    krand=REFERENCE_KRAND,
    seed=REFERENCE_SEED,
    hnf_plume=True,
    smooth_factor=1.0,
    time_integrate=True,
    xmin=REFERENCE_XMIN,
    xmax=REFERENCE_XMAX,
    ymin=REFERENCE_YMIN,
    ymax=REFERENCE_YMAX,
    xres=REFERENCE_XRES,
    yres=REFERENCE_YRES,
    compare_columns=_TRAJ_COLS_WITH_HNF,
    shares_trajectory_with="point",
)

#: Point receptor, smooth_factor=0 — pure binning, no Gaussian spread.
#
# R-STILT calls permute.f90 with a 1×1 kernel; PYSTILT uses np.bincount with no
# convolution step.  This path is not exercised by any other fidelity scenario
# and has different edge-case behaviour (particle at a cell boundary goes to
# exactly one cell with full weight, not spread across neighbours).
SMOOTH_ZERO = ReferenceScenario(
    name="smooth_zero",
    description="Point receptor, smooth_factor=0.0 — 1×1 identity kernel, no Gaussian spread",
    receptor_type="point",
    longitude=REFERENCE_LONGITUDE,
    latitude=REFERENCE_LATITUDE,
    altitude=REFERENCE_ALTITUDE,
    n_hours=REFERENCE_N_HOURS,
    numpar=REFERENCE_NUMPAR,
    krand=REFERENCE_KRAND,
    seed=REFERENCE_SEED,
    hnf_plume=True,
    smooth_factor=0.0,
    time_integrate=False,
    xmin=REFERENCE_XMIN,
    xmax=REFERENCE_XMAX,
    ymin=REFERENCE_YMIN,
    ymax=REFERENCE_YMAX,
    xres=REFERENCE_XRES,
    yres=REFERENCE_YRES,
    compare_columns=_TRAJ_COLS_WITH_HNF,
    shares_trajectory_with="point",
)

#: Point receptor, coarse grid (0.05° × 0.05°) — 5× larger cells than the default.
#
# Coarser resolution produces different kernel sizes in grid-cell units (the same
# physical sigma covers fewer cells) and changes the bin-assignment edge cases.
# Exercises that grid-cell start computation, rasterization, and kernel scaling
# are consistent at resolutions coarser than the reference 0.01°.
COARSE_GRID = ReferenceScenario(
    name="coarse_grid",
    description="Point receptor, xres=yres=0.05° — 5× coarser than reference grid",
    receptor_type="point",
    longitude=REFERENCE_LONGITUDE,
    latitude=REFERENCE_LATITUDE,
    altitude=REFERENCE_ALTITUDE,
    n_hours=REFERENCE_N_HOURS,
    numpar=REFERENCE_NUMPAR,
    krand=REFERENCE_KRAND,
    seed=REFERENCE_SEED,
    hnf_plume=True,
    smooth_factor=1.0,
    time_integrate=False,
    xmin=REFERENCE_XMIN,
    xmax=REFERENCE_XMAX,
    ymin=REFERENCE_YMIN,
    ymax=REFERENCE_YMAX,
    xres=0.05,
    yres=0.05,
    compare_columns=_TRAJ_COLS_WITH_HNF,
    shares_trajectory_with="point",
)

#: Full 24-hour backward run from the reference receptor.
#
# Uses four met files (20151209.00z through 20151209.18z) plus the boundary
# 20151210.00z file — all present in the stilt-tutorials met directory.
# This is the most realistic scenario for operational SLV methane inversions
# and tests met-file chaining, longer trajectory accumulation, and that the
# trajectory parser handles 24 hours of HYSPLIT particle output correctly.
DAY_BACKWARD = ReferenceScenario(
    name="day_backward",
    description="Point receptor, n_hours=-24 — full day backward, four met files",
    receptor_type="point",
    longitude=REFERENCE_LONGITUDE,
    latitude=REFERENCE_LATITUDE,
    altitude=REFERENCE_ALTITUDE,
    n_hours=-24,
    numpar=REFERENCE_NUMPAR,
    krand=REFERENCE_KRAND,
    seed=REFERENCE_SEED,
    hnf_plume=True,
    smooth_factor=1.0,
    time_integrate=False,
    xmin=REFERENCE_XMIN,
    xmax=REFERENCE_XMAX,
    ymin=REFERENCE_YMIN,
    ymax=REFERENCE_YMAX,
    xres=REFERENCE_XRES,
    yres=REFERENCE_YRES,
    compare_columns=_TRAJ_COLS_WITH_HNF,
)

#: Summer convective PBL — boundary layer depth drives HNF dilution very differently
#: from the winter reference scenarios.
HRRR_SUMMER = ReferenceScenario(
    name="hrrr_summer",
    description="Point receptor, 2021-07-15 06Z — summer convective PBL, hnf_plume=True",
    receptor_type="point",
    longitude=REFERENCE_LONGITUDE,
    latitude=REFERENCE_LATITUDE,
    altitude=REFERENCE_ALTITUDE,
    n_hours=REFERENCE_N_HOURS,
    numpar=REFERENCE_NUMPAR,
    krand=REFERENCE_KRAND,
    seed=REFERENCE_SEED,
    hnf_plume=True,
    smooth_factor=1.0,
    time_integrate=False,
    xmin=REFERENCE_XMIN,
    xmax=REFERENCE_XMAX,
    ymin=REFERENCE_YMIN,
    ymax=REFERENCE_YMAX,
    xres=REFERENCE_XRES,
    yres=REFERENCE_YRES,
    compare_columns=_TRAJ_COLS_WITH_HNF,
    time=REFERENCE_SUMMER_TIME,
)

#: Receptor placed near the SW corner of the footprint domain so that a
#: significant fraction of back-trajectories exit the grid.  Exercises
#: particle-binning at grid boundaries and the out-of-domain mask.
EDGE_RECEPTOR = ReferenceScenario(
    name="edge_receptor",
    description="Receptor at SW domain corner (-112.95°, 39.55°) — particle binning near grid boundary",
    receptor_type="point",
    longitude=-112.95,
    latitude=39.55,
    altitude=REFERENCE_ALTITUDE,
    n_hours=REFERENCE_N_HOURS,
    numpar=REFERENCE_NUMPAR,
    krand=REFERENCE_KRAND,
    seed=REFERENCE_SEED,
    hnf_plume=True,
    smooth_factor=1.0,
    time_integrate=False,
    xmin=REFERENCE_XMIN,
    xmax=REFERENCE_XMAX,
    ymin=REFERENCE_YMIN,
    ymax=REFERENCE_YMAX,
    xres=REFERENCE_XRES,
    yres=REFERENCE_YRES,
    compare_columns=_TRAJ_COLS_WITH_HNF,
)

#: Three-location multipoint receptor in summer — highest-risk combination for
#: operational SLV inversions: HNF correction + convective PBL + multipoint.
SUMMER_MULTIPOINT = ReferenceScenario(
    name="summer_multipoint",
    description="Three-location multipoint receptor, 2021-07-15 06Z — summer HNF + multipoint",
    receptor_type="multipoint",
    longitude=(-112.0, -111.5, -111.0),
    latitude=(40.5, 41.0, 41.5),
    altitude=(0.0, 500.0, 1000.0),
    n_hours=REFERENCE_N_HOURS,
    numpar=REFERENCE_NUMPAR,
    krand=REFERENCE_KRAND,
    seed=REFERENCE_SEED,
    hnf_plume=True,
    smooth_factor=1.0,
    time_integrate=False,
    xmin=-113.0,
    xmax=-110.5,
    ymin=39.5,
    ymax=42.0,
    xres=REFERENCE_XRES,
    yres=REFERENCE_YRES,
    compare_columns=_TRAJ_COLS_WITH_HNF,
    time=REFERENCE_SUMMER_TIME,
)

#: Receptor at 0.5 m AGL — maximises HNF plume dilution correction.
#: At near-surface height sigma is very small so the HNF conditional
#: (plume < pbl_mixing) is almost always active, amplifying foot by 1/sigma.
LOW_AGL = ReferenceScenario(
    name="low_agl",
    description="Point receptor at zagl=0.5 m — near-surface HNF stress",
    receptor_type="point",
    longitude=REFERENCE_LONGITUDE,
    latitude=REFERENCE_LATITUDE,
    altitude=0.5,
    n_hours=REFERENCE_N_HOURS,
    numpar=REFERENCE_NUMPAR,
    krand=REFERENCE_KRAND,
    seed=REFERENCE_SEED,
    hnf_plume=True,
    smooth_factor=1.0,
    time_integrate=False,
    xmin=REFERENCE_XMIN,
    xmax=REFERENCE_XMAX,
    ymin=REFERENCE_YMIN,
    ymax=REFERENCE_YMAX,
    xres=REFERENCE_XRES,
    yres=REFERENCE_YRES,
    compare_columns=_TRAJ_COLS_WITH_HNF,
)

#: Receptor 0.2° from western and 0.1° from southern edge of the bbox-cropped
#: HRRR met domain (bbox = -114, 39, -110, 42).  Back-trajectories rapidly
#: exit the met boundary, exercising HYSPLIT's boundary termination and the
#: footprint gridding with a highly sparse particle set.
#:
#: Domain is shifted SW to include the receptor (-113.8°, 39.1°), which lies
#: outside the standard WBB domain.  Using the standard domain would place all
#: particles outside the grid and produce trivially-zero footprints that test
#: nothing meaningful.
MET_GRID_EDGE = ReferenceScenario(
    name="met_grid_edge",
    description=(
        "Receptor at (-113.8°, 39.1°) near SW edge of cropped HRRR met domain"
    ),
    receptor_type="point",
    longitude=-113.8,
    latitude=39.1,
    altitude=REFERENCE_ALTITUDE,
    n_hours=REFERENCE_N_HOURS,
    numpar=REFERENCE_NUMPAR,
    krand=REFERENCE_KRAND,
    seed=REFERENCE_SEED,
    hnf_plume=True,
    smooth_factor=1.0,
    time_integrate=False,
    xmin=-114.0,
    xmax=-112.0,
    ymin=38.5,
    ymax=40.5,
    xres=REFERENCE_XRES,
    yres=REFERENCE_YRES,
    compare_columns=_TRAJ_COLS_WITH_HNF,
)

#: smooth_factor=2.0 — kernel spans many grid cells and the buffer extent
#: must be enlarged accordingly.  Tests buffer construction, edge zero-padding,
#: and that the larger convolution does not introduce spurious mass.
HIGH_SMOOTH = ReferenceScenario(
    name="high_smooth",
    description="Point receptor with smooth_factor=2.0 — large kernel buffer stress",
    receptor_type="point",
    longitude=REFERENCE_LONGITUDE,
    latitude=REFERENCE_LATITUDE,
    altitude=REFERENCE_ALTITUDE,
    n_hours=REFERENCE_N_HOURS,
    numpar=REFERENCE_NUMPAR,
    krand=REFERENCE_KRAND,
    seed=REFERENCE_SEED,
    hnf_plume=True,
    smooth_factor=2.0,
    time_integrate=False,
    xmin=REFERENCE_XMIN,
    xmax=REFERENCE_XMAX,
    ymin=REFERENCE_YMIN,
    ymax=REFERENCE_YMAX,
    xres=REFERENCE_XRES,
    yres=REFERENCE_YRES,
    compare_columns=_TRAJ_COLS_WITH_HNF,
    shares_trajectory_with="point",
)

#: All canonical scenarios in priority order.
ALL_SCENARIOS: list[ReferenceScenario] = [
    POINT,
    COLUMN,
    MULTIPOINT,
    NO_HNF,
    SMOOTH,
    TIME_INTEGRATE,
    SMOOTH_ZERO,
    COARSE_GRID,
    DAY_BACKWARD,
    HRRR_SUMMER,
    EDGE_RECEPTOR,
    SUMMER_MULTIPOINT,
    LOW_AGL,
    MET_GRID_EDGE,
    HIGH_SMOOTH,
]


# ---------------------------------------------------------------------------
# Helper functions (delegate to POINT scenario)
# ---------------------------------------------------------------------------


def reference_python_met_file_tres() -> str:
    return f"{REFERENCE_MET_FILE_INTERVAL_HOURS}h"


def reference_receptor():
    return POINT.make_receptor()


def reference_grid():
    return POINT.make_grid()
