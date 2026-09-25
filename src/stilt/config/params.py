"""Core STILT and HYSPLIT parameter models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self


class ModelParams(BaseModel):
    """Core STILT run controls."""

    n_hours: int = Field(
        -24,
        description="Number of hours to run each simulation; negative indicates backward in time.",
    )
    numpar: int = Field(
        200,
        description=(
            "Number of particles released per simulation. Higher values reduce "
            "stochastic noise in footprints at the cost of runtime and memory."
        ),
    )
    hnf_plume: bool = Field(
        True,
        description=(
            "If true, apply a vertical gaussian plume model to rescale the effective dilution depth for particles in the hyper near-field. This acts to scale up the influence of hyper-local fluxes on the receptor. If enabled, requires varsiwant to include a minimum of dens, tlgr, sigw, foot, mlht, samt. Default is enabled."
        ),
    )
    rm_dat: bool = Field(
        True,
        description="Remove HYSPLIT binary output files (*.dat) after parsing to save disk space.",
    )
    timeout: int | None = Field(
        None,
        description=(
            "Wall-clock cap in seconds on a single hycs_std run. A wedged HYSPLIT process "
            "otherwise blocks its worker forever (proc.wait has no deadline), so one bad "
            "receptor can hold a batch worker until the Slurm wall time kills it. With a "
            "timeout the run raises HYSPLITTimeoutError, which the execution loop already "
            "records as a failure and moves past. Leave unset to wait indefinitely."
        ),
    )
    exe_dir: Path | None = Field(
        None,
        description=(
            "Directory containing a custom ``hycs_std`` build to run instead of the "
            "binary bundled with PYSTILT. Recorded with the trajectory parameters, so "
            "outputs say which build produced them. A build that writes release-time "
            "(t=0) rows to PARTICLE_STILT.DAT makes multipoint and slant receptors exact."
        ),
    )
    varsiwant: list[
        Literal[
            "time",
            "indx",
            "long",
            "lati",
            "zagl",
            "sigw",
            "tlgr",
            "zsfc",
            "icdx",
            "temp",
            "samt",
            "foot",
            "shtf",
            "tcld",
            "dmas",
            "dens",
            "rhfr",
            "sphu",
            "lcld",
            "zloc",
            "dswf",
            "wout",
            "mlht",
            "rain",
            "crai",
            "pres",
            "whtf",
            "temz",
            "zfx1",
        ]
    ] = Field(
        default_factory=lambda: [
            "time",
            "indx",
            "long",
            "lati",
            "zagl",
            "foot",
            "mlht",
            "pres",
            "dens",
            "samt",
            "sigw",
            "tlgr",
        ],
        description=(
            "`hycs_std` particle variables kept in trajectory output. Defaults "
            "to the minimum required variables including 'time', 'indx', "
            "'long', 'lati', 'zagl', 'foot', 'mlht', 'dens', 'samt', "
            "'sigw', 'tlgr'."
        ),
    )


class TransportParams(BaseModel):
    """HYSPLIT transport and turbulence parameterization."""

    capemin: float = Field(
        -1.0,
        description="Minimum CAPE (J/kg) for convective mixing; -1 disables CAPE-triggered enhanced mixing.",
    )
    cmass: int = Field(
        0,
        description="Compute grid output in concentration units (0) or mass units (1).",
    )
    conage: int = Field(
        48, description="Particle age in hours for puff/particle conversion handling."
    )
    cpack: int = Field(1, description="Binary concentration-grid packing mode.")
    delt: int = Field(
        1,
        description="Integration timestep in minutes; 0 lets HYSPLIT choose automatically.",
    )
    dxf: int = Field(
        1, description="Horizontal X-grid adjustment factor for ensemble runs."
    )
    dyf: int = Field(
        1, description="Horizontal Y-grid adjustment factor for ensemble runs."
    )
    dzf: float = Field(
        0.01, description="Vertical grid adjustment factor for ensemble runs."
    )
    efile: str = Field(
        "",
        description="Temporal emissions file name; blank disables file-driven emissions.",
    )
    emisshrs: float = Field(
        0.01,
        description="Duration of emissions in fractional hours.",
    )
    frhmax: float = Field(3.0, description="Maximum horizontal puff-rounding value.")
    frhs: float = Field(
        1.0, description="Standard horizontal puff-rounding fraction for merging."
    )
    frme: float = Field(
        0.1, description="Mass-rounding fraction used by enhanced merging."
    )
    frmr: float = Field(
        0.0, description="Mass-removal fraction used by enhanced merging."
    )
    frts: float = Field(0.1, description="Temporal puff-rounding fraction.")
    frvs: float = Field(0.01, description="Vertical puff-rounding fraction.")
    hscale: int = Field(
        10800, description="Horizontal Lagrangian timescale in seconds."
    )
    ichem: int = Field(
        8,
        description="Chemistry mode; 8 selects STILT particle-in-cell output.",
    )
    idsp: int = Field(
        2,
        description="Dispersion scheme; 1 uses HYSPLIT and 2 uses STILT.",
    )
    initd: int = Field(
        0,
        description="Initial particle distribution mode.",
    )
    k10m: int = Field(
        1,
        description="Use 10 m winds and 2 m temperatures as the lowest meteorology level when available.",
    )
    kagl: int = Field(
        1,
        description="For trajectories, write heights as AGL (1) or MSL (0).",
    )
    kbls: int = Field(
        1,
        description="PBL stability method: fluxes (1) or wind/temperature profiles (2).",
    )
    kblt: int = Field(
        5,
        description="PBL turbulence scheme; PYSTILT defaults to Hanna (5).",
    )
    kdef: int = Field(
        0,
        description="Horizontal turbulence from vertical mixing (0) or deformation (1).",
    )
    khinp: int = Field(
        0,
        description="Maximum particle age read from PARINIT during continuous restart runs.",
    )
    khmax: int = Field(
        9999,
        description="Maximum particle or trajectory age in hours.",
    )
    kmix0: int = Field(150, description="Minimum mixing depth in meters.")
    kmixd: int = Field(
        3,
        description="Mixing-depth method: input, temperature, TKE, or modified Richardson.",
    )
    kmsl: Literal[0, 1] | None = Field(
        None,
        description=(
            "Interpret start altitudes as AGL (0) or MSL (1). "
            "When unset, PYSTILT derives this from each receptor's altitude_ref."
        ),
    )
    kpuff: int = Field(
        0, description="Horizontal puff-growth mode: linear (0) or empirical (1)."
    )
    krand: int = Field(
        4,
        description="Random-number mode for turbulence, repeatability, and diagnostic no-mixing runs.",
    )
    seed: int | None = Field(
        None,
        description=(
            "Random-number seed written to SETUP.CFG. Do not rely on it: with "
            "the bundled hycs_std and krand=2, runs with different seeds came "
            "out identical (neither the turbulence nor the wind-error draw "
            "changed), and krand=4 and 10-13 randomize the seed themselves. "
            "Only krand=4 gives a different draw from run to run."
        ),
    )
    krnd: int = Field(6, description="Enhanced-merging interval in hours.")
    kspl: int = Field(1, description="Standard particle-splitting interval in hours.")
    kwet: int = Field(
        1,
        description="Use meteorological precipitation, or an external ARL rain file when set to 2.",
    )
    kzmix: int = Field(
        0,
        description="Vertical mixing adjustment mode; 0 none, 1 PBL-average, 2 TVMIX scaling.",
    )
    maxdim: int = Field(
        1,
        description="Maximum pollutant species carried on one particle, mainly for chemistry runs.",
    )
    maxpar: int | None = Field(
        None, description="Maximum number of particles allowed in a simulation."
    )
    mgmin: int = Field(10, description="Minimum meteorological subgrid size.")
    mhrs: int = Field(9999, description="Trajectory restart duration limit in hours.")
    nbptyp: int = Field(
        1,
        description="Number of particle-size bins created around each pollutant size entry.",
    )
    ncycl: int = Field(
        0,
        description="PARDUMP output cycle time.",
    )
    ndump: int = Field(
        0,
        description="Write particle dumps every n hours; 0 disables dumps.",
    )
    ninit: int = Field(
        1,
        description="Particle initialization mode for restart, add, or replace workflows.",
    )
    nstr: int = Field(0, description="Trajectory restart interval in hours.")
    nturb: int = Field(
        0,
        description="Turbulence mode selector; 0 is on/default, 1 disables turbulence.",
    )
    nver: int = Field(0, description="Trajectory vertical split number.")
    outdt: int = Field(
        0,
        description="Minutes between STILT endpoint writes to PARTICLE.DAT; negative disables output.",
    )
    p10f: int = Field(1, description="Dust threshold-velocity sensitivity factor.")
    pinbc: str = Field(
        "",
        description="Particle input file used for boundary-condition particles.",
    )
    pinpf: str = Field(
        "",
        description="Particle input file for initialization or boundary-condition runs.",
    )
    poutf: str = Field(
        "",
        description="Particle output file name.",
    )
    qcycle: int = Field(
        0, description="Emission cycling period in hours; 0 disables cycling."
    )
    rhb: float = Field(
        80.0,
        description="Relative-humidity threshold used to define cloud base.",
    )
    rht: float = Field(
        60.0,
        description="Relative-humidity threshold below which cloud top is considered to end.",
    )
    splitf: int = Field(
        1,
        description="Automatic horizontal split-size factor; negative disables auto sizing.",
    )
    tkerd: float = Field(0.18, description="Unstable TKE ratio w'²/(u'²+v'²).")
    tkern: float = Field(0.18, description="Stable TKE ratio w'²/(u'²+v'²).")
    tlfrac: float = Field(
        0.1,
        description="Fraction of the vertical Lagrangian timescale used to set the STILT timestep.",
    )
    tout: float = Field(
        0.0,
        description="Trajectory output interval in minutes.",
    )
    tratio: float = Field(0.75, description="Advection stability ratio.")
    tvmix: float = Field(
        1.0,
        description="Scale factor applied to vertical mixing coefficients for selected KZMIX modes.",
    )
    veght: float = Field(
        0.5,
        description="Height threshold used to accumulate STILT footprint residence time.",
    )
    vscale: int = Field(
        200,
        description="Vertical Lagrangian timescale in seconds for neutral PBL conditions.",
    )
    vscaleu: int = Field(
        200,
        description="Vertical Lagrangian timescale in seconds for unstable PBL conditions.",
    )
    vscales: int = Field(
        -1,
        description="Vertical Lagrangian timescale in seconds for stable PBL conditions.",
    )
    w_option: int = Field(
        0,
        description="Vertical motion method; 0 met vertical velocity, 1 isob, 2 isen, 3 dens, 4 sigma.",
    )
    wbbh: int = Field(
        0, description="Height where fixed vertical motion switches from rise to fall."
    )
    wbwf: int = Field(
        0, description="Fixed fall velocity used by vertical-motion options 9 or 10."
    )
    wbwr: int = Field(
        0, description="Fixed rise velocity used by vertical-motion option 9."
    )
    wvert: bool = Field(
        False,
        description="Use the WRF vertical interpolation scheme for vertical velocity when true.",
    )
    z_top: float = Field(
        25000.0,
        description="Top of model domain, in meters above ground level; defaults to 25000.0",
    )
    ziscale: float | list[float] | list[list[float]] = Field(
        1.0,
        description=(
            "Factor on the mixed-layer height, written to HYSPLIT's ZICONTROL "
            "file. 1.0 (the default) leaves it unscaled; any other value turns "
            "scaling on. A scalar applies to every hour of the run; a list gives "
            "one factor per hour from the release, and later hours are unscaled. "
            "HYSPLIT applies kmix0 after the factor, so the mixed layer never "
            "drops below kmix0. At most 150 hourly factors. A negative value "
            "uses the meteorology's own PBL height where the met files carry one."
        ),
    )


class ErrorParams(BaseModel):
    """Transport error trajectory parameters for XY and ZI perturbations."""

    siguverr: float | None = Field(
        None,
        description="Standard deviation of horizontal wind error [m/s]",
    )
    tluverr: float | None = Field(
        None,
        description="Standard deviation of horiztontal wind error timescale [min]",
    )
    zcoruverr: float | None = Field(
        None,
        description="Vertical correlation length scale of horizontal wind error [m]",
    )
    horcoruverr: float | None = Field(
        None,
        description="Horizontal correlation length scale of horizontal wind error [km]",
    )
    sigzierr: float | None = Field(
        None,
        description="Standard deviation of mixed-layer height errors [%]",
    )
    tlzierr: float | None = Field(
        None,
        description="Standard deviation of mixed layer height timescale [min]",
    )
    horcorzierr: float | None = Field(
        None,
        description="Horizontal correlation length scale of mixed-layer height errors [km]",
    )

    error_realizations: int = Field(
        1,
        ge=1,
        description=(
            "Number of error trajectories to run per simulation. Each is an "
            "independent draw of the perturbation field; transport_error "
            "averages their variance estimates, which cuts the perturbed "
            "side's sampling noise by 1/sqrt(N) (the shared main run bounds "
            "the overall gain at sqrt(2)). More than one requires krand=4: "
            "HYSPLIT draws the perturbation from the seed it randomizes only "
            "in that mode, and the bundled hycs_std ignores SETUP.CFG's seed "
            "for it, so other modes would repeat the same draw."
        ),
    )

    XYERR_PARAMS: ClassVar[tuple[str, ...]] = (
        "siguverr",
        "tluverr",
        "zcoruverr",
        "horcoruverr",
    )
    ZIERR_PARAMS: ClassVar[tuple[str, ...]] = (
        "sigzierr",
        "tlzierr",
        "horcorzierr",
    )

    @model_validator(mode="after")
    def _validate_error_params(self) -> Self:
        """Validate grouped wind and mixed-layer perturbation parameters."""
        for name, params in [
            ("XY", self._xyerr_params()),
            ("ZI", self._zierr_params()),
        ]:
            is_na = [pd.isna(v) for v in params.values()]
            if any(is_na) and not all(is_na):
                raise ValueError(
                    f"Inconsistent {name} error parameters: all must be set or all None"
                )
        return self

    def _xyerr_params(self) -> dict[str, float | None]:
        """Return the horizontal wind-perturbation parameter set."""
        return {p: getattr(self, p) for p in self.XYERR_PARAMS}

    def _zierr_params(self) -> dict[str, float | None]:
        """Return the mixed-layer perturbation parameter set."""
        return {p: getattr(self, p) for p in self.ZIERR_PARAMS}

    @property
    def winderrtf(self) -> int:
        """HYSPLIT WINDERRTF flag encoding active error modes."""
        xyerr = all(v is not None for v in self._xyerr_params().values())
        zierr = all(v is not None for v in self._zierr_params().values())
        return xyerr + 2 * zierr

    @property
    def error_enabled(self) -> bool:
        """
        Whether an error-trajectory mode is configured (XY and/or ZI).

        When True, a run writes ``error_realizations`` wind-perturbed
        ``*_error`` trajectories alongside the main trajectory, so completion
        checks should require all of them to be present.
        """
        return self.winderrtf > 0


class STILTParams(ModelParams, TransportParams, ErrorParams):
    """
    All STILT/HYSPLIT parameters in one flat model.

    Every :class:`TransportParams` field is a ``SETUP.CFG`` namelist entry
    except the few HYSPLIT reads from ``CONTROL`` or ``ZICONTROL``;
    :meth:`setup_entries` applies that rule. :class:`ErrorParams` fields go to
    ``WINDERR`` / ``ZIERR`` (see ``ErrorParams.XYERR_PARAMS`` / ``ZIERR_PARAMS``).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    #: Fields HYSPLIT reads from CONTROL rather than SETUP.CFG.
    CONTROL_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"n_hours", "emisshrs", "w_option", "z_top"}
    )
    #: Fields written to ZICONTROL rather than SETUP.CFG.
    ZICONTROL_FIELDS: ClassVar[frozenset[str]] = frozenset({"ziscale"})
    #: Most hourly ZICONTROL factors HYSPLIT can hold (``ZIPRESC(150)`` in
    #: hymodelc.F); it reads more without a bounds check.
    MAX_ZISCALE_HOURS: ClassVar[int] = 150
    #: ModelParams fields that are SETUP.CFG entries.
    _MODEL_SETUP_FIELDS: ClassVar[frozenset[str]] = frozenset({"numpar", "varsiwant"})

    def setup_entries(self) -> dict[str, Any]:
        """Return the ``SETUP.CFG`` namelist entries (``None`` values omitted)."""
        names = [
            *(n for n in ModelParams.model_fields if n in self._MODEL_SETUP_FIELDS),
            *(
                n
                for n in TransportParams.model_fields
                if n not in self.CONTROL_FIELDS and n not in self.ZICONTROL_FIELDS
            ),
        ]
        entries = {n: getattr(self, n) for n in names if getattr(self, n) is not None}
        entries["zicontroltf"] = self.zicontroltf
        return entries

    @property
    def ziscale_factors(self) -> list[float] | None:
        """
        Hourly mixed-layer factors for ZICONTROL, or ``None`` when unscaled.

        A scalar ``ziscale`` is repeated for every hour of the run; a list is
        used as given. All factors equal to 1.0 means no scaling.
        """
        if isinstance(self.ziscale, int | float):
            values = [float(self.ziscale)] * max(abs(self.n_hours), 1)
        else:
            values = _hourly_ziscale(self.ziscale)
        if all(v == 1.0 for v in values):
            return None
        return values

    @property
    def zicontroltf(self) -> int:
        """HYSPLIT ZICONTROLTF flag: 1 when ``ziscale`` scales the mixed layer."""
        return int(self.ziscale_factors is not None)

    @model_validator(mode="after")
    def _validate_ziscale(self) -> Self:
        """Reject factors HYSPLIT would misread: empty, zero, or too many hours."""
        if isinstance(self.ziscale, int | float):
            values = [float(self.ziscale)]
        else:
            values = _hourly_ziscale(self.ziscale)
        if not values:
            raise ValueError("ziscale cannot be empty; use 1.0 for no scaling.")
        if any(v == 0.0 for v in values):
            raise ValueError(
                "ziscale of 0 would collapse the mixed layer to kmix0. STILT-R "
                "uses 0 to mean unset; use 1.0 for no scaling."
            )
        factors = self.ziscale_factors
        if factors is not None and len(factors) > self.MAX_ZISCALE_HOURS:
            raise ValueError(
                f"ziscale gives {len(factors)} hourly factors, but HYSPLIT holds at "
                f"most {self.MAX_ZISCALE_HOURS}. A scalar ziscale is repeated for "
                "every hour, so it needs abs(n_hours) <= "
                f"{self.MAX_ZISCALE_HOURS}; for longer runs give a list of up to "
                f"{self.MAX_ZISCALE_HOURS} factors, after which the mixed layer "
                "is unscaled."
            )
        return self

    @model_validator(mode="after")
    def _set_maxpar(self) -> Self:
        """Default ``maxpar`` to ``numpar`` when the user omits it."""
        if self.maxpar is None:
            self.maxpar = self.numpar
        return self

    @model_validator(mode="after")
    def _validate_error_realizations(self) -> Self:
        """Several realizations need HYSPLIT to draw a fresh perturbation each run."""
        if self.error_realizations > 1 and self.krand != 4:
            raise ValueError(
                f"error_realizations={self.error_realizations} requires krand=4 "
                f"(got krand={self.krand}): HYSPLIT randomizes the wind-error "
                "draw only in that mode, and the bundled hycs_std ignores the "
                "namelist seed for it, so every realization would repeat the "
                "same perturbation."
            )
        return self

    @model_validator(mode="after")
    def _validate_hnf_plume(self) -> Self:
        """Raise at construction time if hnf_plume=True but varsiwant is missing required columns."""
        if self.hnf_plume:
            required = {"dens", "samt", "sigw", "tlgr", "foot", "mlht"}
            missing = required - set(self.varsiwant)
            if missing:
                raise ValueError(
                    f"hnf_plume=True requires varsiwant to include: {sorted(missing)}"
                )
        return self


def _hourly_ziscale(raw: list[float] | list[list[float]]) -> list[float]:
    """Flatten a ``ziscale`` list, allowing STILT-R's one-element nested form."""
    items: list[Any] = list(raw)
    if items and isinstance(items[0], list):
        if len(items) != 1:
            raise ValueError(
                "Per-simulation ziscale lists are not supported. Pass one shared "
                "list of hourly factors for all simulations."
            )
        items = list(items[0])
    return [float(v) for v in items]


__all__ = ["ErrorParams", "ModelParams", "STILTParams", "TransportParams"]
