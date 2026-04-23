"""Core STILT and HYSPLIT parameter models."""

from __future__ import annotations

from typing import ClassVar, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from .fields import cfg_field


class ModelParams(BaseModel):
    """Core STILT run controls."""

    DEFAULT_TARGET: ClassVar[str | None] = None

    n_hours: int = Field(
        -24,
        description="Number of hours to run each simulation; negative indicates backward in time.",
    )
    numpar: int = cfg_field(
        200,
        description=(
            "Number of particles released per simulation. Higher values reduce "
            "stochastic noise in footprints at the cost of runtime and memory."
        ),
        target="setup",
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
    ] = cfg_field(
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
        target="setup",
        visibility="advanced",
    )


class TransportParams(BaseModel):
    """HYSPLIT transport and turbulence parameterization."""

    DEFAULT_TARGET: ClassVar[str | None] = "setup"

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
    emisshrs: float = cfg_field(
        0.01,
        description="Duration of emissions in fractional hours.",
        target="control",
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
    ichem: int = cfg_field(
        8,
        description="Chemistry mode; 8 selects STILT particle-in-cell output.",
        visibility="internal",
    )
    idsp: int = cfg_field(
        2,
        description="Dispersion scheme; 1 uses HYSPLIT and 2 uses STILT.",
        visibility="internal",
    )
    initd: int = Field(
        0,
        description="Initial particle distribution mode.",
    )
    k10m: int = Field(
        1,
        description="Use 10 m winds and 2 m temperatures as the lowest meteorology level when available.",
    )
    kagl: int = cfg_field(
        1,
        description="For trajectories, write heights as AGL (1) or MSL (0).",
        visibility="internal",
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
    seed: int | None = cfg_field(
        None,
        description=(
            "Optional HYSPLIT random-number seed written to SETUP.CFG. "
            "Use with krand values that preserve a fixed initial seed; krand=4 "
            "and 10-13 still randomize the initial seed."
        ),
        target="setup",
        visibility="advanced",
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
    ncycl: int = cfg_field(
        0,
        description="PARDUMP output cycle time.",
        visibility="internal",
    )
    ndump: int = cfg_field(
        0,
        description="Write particle dumps every n hours; 0 disables dumps.",
        visibility="internal",
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
    outdt: int = cfg_field(
        0,
        description="Minutes between STILT endpoint writes to PARTICLE.DAT; negative disables output.",
        visibility="advanced",
    )
    p10f: int = Field(1, description="Dust threshold-velocity sensitivity factor.")
    pinbc: str = cfg_field(
        "",
        description="Particle input file used for boundary-condition particles.",
        visibility="internal",
    )
    pinpf: str = cfg_field(
        "",
        description="Particle input file for initialization or boundary-condition runs.",
        visibility="internal",
    )
    poutf: str = cfg_field(
        "",
        description="Particle output file name.",
        visibility="internal",
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
    w_option: int = cfg_field(
        0,
        description="Vertical motion method; 0 met vertical velocity, 1 isob, 2 isen, 3 dens, 4 sigma.",
        target="control",
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
    z_top: float = cfg_field(
        25000.0,
        description="Top of model domain, in meters above ground level; defaults to 25000.0",
        target="control",
    )
    zicontroltf: int = cfg_field(
        0,
        description="Enable domain-wide PBL scaling from a ZICONTROL file.",
        visibility="advanced",
    )
    ziscale: float | list[float] | list[list[float]] = cfg_field(
        1.0,
        description=(
            "Manually scale the mixed-layer height. Scalars expand across the run; "
            "lists define shared hourly factors."
        ),
        target="zicontrol",
        visibility="advanced",
    )


class ErrorParams(BaseModel):
    """Transport error trajectory parameters for XY and ZI perturbations."""

    DEFAULT_TARGET: ClassVar[str | None] = None

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


class STILTParams(ModelParams, TransportParams, ErrorParams):
    """All STILT/HYSPLIT parameters in one flat model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _set_maxpar(self) -> Self:
        """Default ``maxpar`` to ``numpar`` when the user omits it."""
        if self.maxpar is None:
            self.maxpar = self.numpar
        return self


__all__ = ["ErrorParams", "ModelParams", "STILTParams", "TransportParams"]
