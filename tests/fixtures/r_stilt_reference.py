"""
Single source of truth for the seeded R-STILT reference cases.

Defines all canonical scenarios used by the PYSTILT fidelity integration tests.
Each scenario is represented by a :class:`ReferenceScenario` instance that knows
how to build the PYSTILT objects it needs.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Shared reference constants
# ---------------------------------------------------------------------------

REFERENCE_MET = "hrrr"
REFERENCE_TIME = dt.datetime(2015, 12, 10, 0, 0)
REFERENCE_LONGITUDE = -112.0
REFERENCE_LATITUDE = 40.5
REFERENCE_ALTITUDE = 5.0

REFERENCE_N_HOURS = -6
REFERENCE_NUMPAR = 1000
REFERENCE_KRAND = 2
REFERENCE_SEED = 42
REFERENCE_MET_FILE_FORMAT = "%Y%m%d.%Hz.hrrra"
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

    # ------------------------------------------------------------------
    # PYSTILT object factories (lazy imports to avoid circular load)
    # ------------------------------------------------------------------

    def make_receptor(self):
        """Return the appropriate Receptor subtype for this scenario."""
        from stilt.receptors import ColumnReceptor, MultiPointReceptor, PointReceptor

        if self.receptor_type == "point":
            return PointReceptor(
                REFERENCE_TIME,
                float(self.longitude),  # type: ignore[arg-type]
                float(self.latitude),  # type: ignore[arg-type]
                float(self.altitude),  # type: ignore[arg-type]
            )
        if self.receptor_type == "column":
            bottom, top = self.altitude  # type: ignore[misc]
            return ColumnReceptor(
                REFERENCE_TIME,
                float(self.longitude),  # type: ignore[arg-type]
                float(self.latitude),  # type: ignore[arg-type]
                bottom=float(bottom),
                top=float(top),
            )
        return MultiPointReceptor(
            REFERENCE_TIME,
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
                        "file_format": REFERENCE_MET_FILE_FORMAT,
                        "file_tres": reference_python_met_file_tres(),
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
)

#: All canonical scenarios in priority order (POINT has committed fixtures first).
ALL_SCENARIOS: list[ReferenceScenario] = [
    POINT,
    COLUMN,
    MULTIPOINT,
    NO_HNF,
    SMOOTH,
    TIME_INTEGRATE,
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
