"""
The transport settings of a run.

:class:`TransportParams` holds every setting that shapes a run's particles.
Most are HYSPLIT's own (written to its ``SETUP.CFG``, ``CONTROL``,
``ZICONTROL``, ``WINDERR``, and ``ZIERR`` files, under the same names); a few
are used by PYSTILT itself. Each field records which, as
``json_schema_extra={"file": ...}``, and :func:`fields_in` lists them. The
HYSPLIT driver writes the files from that, so adding a setting is one field.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: Where a setting goes: one of HYSPLIT's input files, or PYSTILT itself.
SETUP: dict[str, Any] = {"file": "SETUP.CFG"}
CONTROL: dict[str, Any] = {"file": "CONTROL"}
ZICONTROL: dict[str, Any] = {"file": "ZICONTROL"}
WINDERR: dict[str, Any] = {"file": "WINDERR"}
ZIERR: dict[str, Any] = {"file": "ZIERR"}
PYSTILT: dict[str, Any] = {"file": "PYSTILT"}

#: Most hourly ZICONTROL factors HYSPLIT can hold (``ZIPRESC(150)`` in
#: hymodelc.F); it reads more without a bounds check.
MAX_ZISCALE_HOURS = 150


class TransportParams(BaseModel):
    """
    Every setting that shapes a run's particles.

    In ``config.yaml`` they are flat, top-level keys, and a variant may
    override any of them. Most are HYSPLIT ``SETUP.CFG`` entries with
    HYSPLIT's own names; see the HYSPLIT user guide for the full meaning of
    each. The wind-error group (``siguverr``, ``tluverr``, ``zcoruverr``,
    ``horcoruverr``) perturbs the particles' winds, and the mixed-layer group
    (``sigzierr``, ``tlzierr``, ``horcorzierr``) perturbs each particle's
    footprint by a random mixed-layer height error. Each group is set in full
    or not at all.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    #: Fields left out of a run's recorded settings. What they point at is
    #: recorded instead: the build's version and the checksums of data files
    #: that differ from the bundled ones.
    UNRECORDED: ClassVar[frozenset[str]] = frozenset({"exe_dir", "data_dir"})

    n_hours: int = Field(
        -24,
        description="Length of each simulation, in hours. Negative runs backward in time.",
        json_schema_extra=CONTROL,
    )
    numpar: int = Field(
        200,
        description=(
            "Number of particles released per simulation. More particles give a "
            "less noisy footprint and take longer to run."
        ),
        json_schema_extra=SETUP,
    )
    hnf_plume: bool = Field(
        True,
        description=(
            "Apply a vertical Gaussian plume model to particles in the hyper "
            "near-field. This shrinks their effective dilution depth and raises "
            "the influence of fluxes close to the receptor. Requires "
            "``varsiwant`` to include ``dens``, ``tlgr``, ``sigw``, ``foot``, "
            "``mlht``, and ``samt``."
        ),
        json_schema_extra=PYSTILT,
    )
    exe_dir: Path | None = Field(
        None,
        description=(
            "Directory holding a custom ``hycs_std`` build to run in place of "
            "the one bundled with PYSTILT. It is saved with each trajectory's "
            "parameters. A build that writes release-time (t = 0) rows to "
            "``PARTICLE_STILT.DAT`` gives exact release heights for multipoint "
            "and slant receptors."
        ),
        json_schema_extra=PYSTILT,
    )
    data_dir: Path | None = Field(
        None,
        description=(
            "Directory of HYSPLIT data tables (``ASCDATA.CFG``, ``LANDUSE.ASC``, "
            "``ROUGLEN.ASC``, ``TERRAIN.ASC``) to use in place of the bundled "
            "ones. A table it does not hold comes from the bundled set. The "
            "tables change the particles, so each table that differs from the "
            "bundled one is recorded with the run by its checksum."
        ),
        json_schema_extra=PYSTILT,
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
            "Particle variables ``hycs_std`` writes to the trajectory output. "
            "The default is the set footprints need, plus ``pres`` for "
            "pressure weighting."
        ),
        json_schema_extra=SETUP,
    )
    capemin: float = Field(
        -1.0,
        description=(
            "Convection option. -1 turns convection off, -2 uses the Grell "
            "scheme, and a positive value mixes vertically when CAPE exceeds "
            "it, in J/kg."
        ),
        json_schema_extra=SETUP,
    )
    cmass: int = Field(
        0,
        description="Compute grid concentrations (0) or grid mass (1).",
        json_schema_extra=SETUP,
    )
    conage: int = Field(
        48,
        description="Particle age at which particles and puffs convert, in hours.",
        json_schema_extra=SETUP,
    )
    cpack: int = Field(
        1,
        description="Packing of the binary concentration grid.",
        json_schema_extra=SETUP,
    )
    delt: float = Field(
        1.0,
        description=(
            "Integration time step, in minutes. 0 lets HYSPLIT choose; a "
            "negative value sets the minimum step."
        ),
        json_schema_extra=SETUP,
    )
    dxf: float = Field(
        1.0,
        description="Horizontal x-grid offset factor for ensemble runs.",
        json_schema_extra=SETUP,
    )
    dyf: float = Field(
        1.0,
        description="Horizontal y-grid offset factor for ensemble runs.",
        json_schema_extra=SETUP,
    )
    dzf: float = Field(
        0.01,
        description="Vertical offset factor for ensemble runs (0.01 is about 250 m).",
        json_schema_extra=SETUP,
    )
    efile: str = Field(
        "",
        description="Name of a time-varying emissions file. Blank uses none.",
        json_schema_extra=SETUP,
    )
    emisshrs: float = Field(
        0.01,
        description="Duration of the particle release, in hours.",
        json_schema_extra=CONTROL,
    )
    frhmax: float = Field(
        3.0,
        description="Maximum horizontal puff-rounding parameter.",
        json_schema_extra=SETUP,
    )
    frhs: float = Field(
        1.0,
        description="Horizontal puff-rounding fraction for merging.",
        json_schema_extra=SETUP,
    )
    frme: float = Field(
        0.1,
        description="Mass-rounding fraction for enhanced merging.",
        json_schema_extra=SETUP,
    )
    frmr: float = Field(
        0.0,
        description="Mass-removal fraction for enhanced merging.",
        json_schema_extra=SETUP,
    )
    frts: float = Field(
        0.1, description="Temporal puff-rounding fraction.", json_schema_extra=SETUP
    )
    frvs: float = Field(
        0.01, description="Vertical puff-rounding fraction.", json_schema_extra=SETUP
    )
    hscale: float = Field(
        10800.0,
        description="Horizontal Lagrangian timescale, in seconds.",
        json_schema_extra=SETUP,
    )
    ichem: int = Field(
        8,
        description="HYSPLIT chemistry and output mode. 8 is the STILT emulation mode.",
        json_schema_extra=SETUP,
    )
    idsp: int = Field(
        2,
        description="Particle dispersion scheme: 1 for HYSPLIT, 2 for STILT.",
        json_schema_extra=SETUP,
    )
    initd: int = Field(
        0,
        description="Initial distribution as particles, puffs, or a mix. 0 is 3D particles.",
        json_schema_extra=SETUP,
    )
    k10m: int = Field(
        1,
        description=(
            "Use the 10 m winds and 2 m temperature as the lowest meteorology "
            "level (1) or skip them (0)."
        ),
        json_schema_extra=SETUP,
    )
    kagl: int = Field(
        1,
        description="Write trajectory heights above ground (1) or above sea level (0).",
        json_schema_extra=SETUP,
    )
    kbls: int = Field(
        1,
        description=(
            "Derive boundary-layer stability from surface fluxes (1) or from "
            "wind and temperature profiles (2)."
        ),
        json_schema_extra=SETUP,
    )
    kblt: int = Field(
        5,
        description=(
            "Boundary-layer turbulence scheme: 1 Beljaars, 2 Kantha-Clayson, "
            "3 TKE, 4 measured variances, 5 Hanna."
        ),
        json_schema_extra=SETUP,
    )
    kdef: int = Field(
        0,
        description="Horizontal turbulence from vertical mixing (0) or wind deformation (1).",
        json_schema_extra=SETUP,
    )
    khinp: int = Field(
        0,
        description="Age, in hours, given to particles read from ``pinpf``. 0 keeps their own age.",
        json_schema_extra=SETUP,
    )
    khmax: int = Field(
        9999,
        description="Maximum particle or trajectory age, in hours.",
        json_schema_extra=SETUP,
    )
    kmix0: int = Field(
        150,
        description="Minimum mixed-layer depth, in meters.",
        json_schema_extra=SETUP,
    )
    kmixd: int = Field(
        3,
        description=(
            "Mixed-layer depth source: 0 from the meteorology, 1 from the "
            "temperature profile, 2 from the TKE profile, 3 from a modified "
            "Richardson number."
        ),
        json_schema_extra=SETUP,
    )
    kmsl: Literal[0, 1] | None = Field(
        None,
        description=(
            "Read release heights as above ground (0) or above sea level (1). "
            "Unset takes it from each receptor's ``altitude_ref``, and a value "
            "that disagrees with a receptor is an error."
        ),
        json_schema_extra=SETUP,
    )
    kpuff: int = Field(
        0,
        description="Horizontal puff growth: linear (0) or empirical (1).",
        json_schema_extra=SETUP,
    )
    krand: Literal[0, 1, 2, 3, 4, 10, 11, 12, 13] = Field(
        4,
        description=(
            "How HYSPLIT draws the random numbers for turbulence. 0 picks 2 "
            "when ``numpar`` is 5000 or less and 1 otherwise. 1 uses a "
            "precomputed table. 2 draws them during the run and is the only "
            "mode that uses ``seed``. 3 uses no random numbers (a diagnostic "
            "mode). 4 draws them during the run from a clock-based seed, so "
            "every run differs; there are about 5000 possible seeds. 10 to 13 "
            "are modes 0 to 3 with a random initial seed. HYSPLIT does not "
            "check this value and other values silently break the turbulence, "
            "so PYSTILT rejects them."
        ),
        json_schema_extra=SETUP,
    )
    seed: int | None = Field(
        None,
        description=(
            "Random seed for a reproducible run. Different seeds give "
            "different runs. Requires ``krand: 2``. The bundled ``hycs_std`` "
            "ignores the seed under ``krand`` 4 and 10 to 13, and under 1 uses "
            "it only for the initial turbulent velocity. PYSTILT writes it to "
            "``SETUP.CFG`` as ``-(abs(seed) + 1)``, because HYSPLIT reseeds its "
            "generator only from a negative value and gives every ``SEED`` of "
            "0 or more the same stream. Realization ``k`` of a variant runs "
            "with ``seed + k``, so realization 0 shares the unperturbed run's "
            "seed, as STILT-R's error run does."
        ),
        json_schema_extra=SETUP,
    )
    krnd: int = Field(
        6, description="Enhanced-merging interval, in hours.", json_schema_extra=SETUP
    )
    kspl: int = Field(
        1,
        description="Standard puff-splitting interval, in hours.",
        json_schema_extra=SETUP,
    )
    kwet: int = Field(
        1,
        description="Precipitation from the meteorology (1) or from an external ARL file (2).",
        json_schema_extra=SETUP,
    )
    kzmix: int = Field(
        0,
        description=(
            "Vertical mixing adjustment: 0 none, 1 a single PBL-average value, "
            "2 scale by ``tvmix``."
        ),
        json_schema_extra=SETUP,
    )
    maxdim: int = Field(
        1,
        description="Maximum number of pollutant species carried on one particle.",
        json_schema_extra=SETUP,
    )
    maxpar: int | None = Field(
        None,
        description="Maximum number of particles in a simulation. Unset uses ``numpar``.",
        json_schema_extra=SETUP,
    )
    mgmin: int = Field(
        10,
        description="Minimum meteorological subgrid size, in grid points.",
        json_schema_extra=SETUP,
    )
    mhrs: int = Field(
        9999,
        description="Trajectory restart duration limit, in hours.",
        json_schema_extra=SETUP,
    )
    nbptyp: int = Field(
        1,
        description="Number of particle-size bins per pollutant type.",
        json_schema_extra=SETUP,
    )
    ncycl: int = Field(
        0,
        description="Cycle time of the particle dump file, in hours.",
        json_schema_extra=SETUP,
    )
    ndump: int = Field(
        0,
        description="Interval between particle dumps, in hours. 0 writes none.",
        json_schema_extra=SETUP,
    )
    ninit: int = Field(
        1,
        description=(
            "Particle initialization from ``pinpf``: 0 none, 1 once at the "
            "start, 2 add every hour, 3 replace every hour."
        ),
        json_schema_extra=SETUP,
    )
    nstr: int = Field(
        0, description="Trajectory restart interval, in hours.", json_schema_extra=SETUP
    )
    nturb: int = Field(
        0,
        description="Turbulence on (0) or off (1).",
        json_schema_extra=SETUP,
    )
    nver: int = Field(
        0, description="Trajectory vertical split number.", json_schema_extra=SETUP
    )
    outdt: int = Field(
        0,
        description=(
            "Interval between particle outputs in ``PARTICLE_STILT.DAT``, in "
            "minutes. 0 writes every time step and a negative value writes none."
        ),
        json_schema_extra=SETUP,
    )
    p10f: float = Field(
        1.0,
        description="Dust threshold velocity sensitivity factor.",
        json_schema_extra=SETUP,
    )
    pinbc: str = Field(
        "",
        description="Particle input file for time-varying boundary conditions.",
        json_schema_extra=SETUP,
    )
    pinpf: str = Field(
        "",
        description="Particle input file for initialization or boundary-condition runs.",
        json_schema_extra=SETUP,
    )
    poutf: str = Field(
        "",
        description="Particle output file name.",
        json_schema_extra=SETUP,
    )
    qcycle: float = Field(
        0.0,
        description="Emission cycling period, in hours. 0 turns cycling off.",
        json_schema_extra=SETUP,
    )
    rhb: int = Field(
        80,
        description="Relative humidity that defines a cloud base, in percent.",
        json_schema_extra=SETUP,
    )
    rht: int = Field(
        60,
        description="Relative humidity that defines a cloud top, in percent.",
        json_schema_extra=SETUP,
    )
    splitf: float = Field(
        1.0,
        description=(
            "Factor for the automatic horizontal splitting size. A negative "
            "value turns the automatic sizing off."
        ),
        json_schema_extra=SETUP,
    )
    tkerd: float = Field(
        0.18,
        description="Ratio w'²/(u'²+v'²) of TKE components when unstable.",
        json_schema_extra=SETUP,
    )
    tkern: float = Field(
        0.18,
        description="Ratio w'²/(u'²+v'²) of TKE components when stable.",
        json_schema_extra=SETUP,
    )
    tlfrac: float = Field(
        0.1,
        description=(
            "Fraction of the vertical Lagrangian timescale used as the time "
            "step of the STILT dispersion scheme."
        ),
        json_schema_extra=SETUP,
    )
    tout: int = Field(
        0,
        description="Trajectory output interval, in minutes.",
        json_schema_extra=SETUP,
    )
    tratio: float = Field(
        0.75,
        description="Advection stability ratio (fraction of a grid cell per time step).",
        json_schema_extra=SETUP,
    )
    tvmix: float = Field(
        1.0,
        description="Vertical mixing scale factor, used by the ``kzmix`` scaling modes.",
        json_schema_extra=SETUP,
    )
    veght: float = Field(
        0.5,
        description=(
            "Height below which a particle's time counts toward the footprint. "
            "A value of 1 or less is a fraction of the mixed-layer height; a "
            "larger value is meters above ground."
        ),
        json_schema_extra=SETUP,
    )
    vscale: float = Field(
        200.0,
        description="Vertical Lagrangian timescale, in seconds.",
        json_schema_extra=SETUP,
    )
    vscaleu: float = Field(
        200.0,
        description="Vertical Lagrangian timescale in an unstable boundary layer, in seconds.",
        json_schema_extra=SETUP,
    )
    vscales: float = Field(
        -1.0,
        description=(
            "Vertical Lagrangian timescale in a stable boundary layer, in "
            "seconds. -1 uses the Hanna timescale, which varies with the "
            "turbulence, and then ``vscaleu`` is not used."
        ),
        json_schema_extra=SETUP,
    )
    w_option: int = Field(
        0,
        description=(
            "Vertical motion method: 0 the meteorology's vertical velocity, "
            "1 isobaric, 2 isentropic, 3 constant density, 4 constant sigma."
        ),
        json_schema_extra=CONTROL,
    )
    wbbh: float = Field(
        0.0,
        description=(
            "Height at which the fixed vertical velocity switches from rise to "
            "fall, in meters. Used by vertical motion option 9."
        ),
        json_schema_extra=SETUP,
    )
    wbwf: float = Field(
        0.0,
        description="Fixed fall velocity, in m/s. Used by vertical motion options 9 and 10.",
        json_schema_extra=SETUP,
    )
    wbwr: float = Field(
        0.0,
        description="Fixed rise velocity, in m/s. Used by vertical motion option 9.",
        json_schema_extra=SETUP,
    )
    wvert: bool = Field(
        False,
        description="Interpolate WRF fields vertically with the WRF scheme instead of HYSPLIT's.",
        json_schema_extra=SETUP,
    )
    z_top: float = Field(
        25000.0,
        description="Top of the model domain, in meters above ground.",
        json_schema_extra=CONTROL,
    )
    ziscale: float | list[float] = Field(
        1.0,
        description=(
            "Factor applied to the mixed-layer height, written to HYSPLIT's "
            "``ZICONTROL`` file. 1.0 leaves it unscaled. A single value applies "
            "to every hour of the run. A list gives one factor per hour from "
            "the release (at most 150), and later hours are unscaled. HYSPLIT "
            "applies ``kmix0`` after the factor, so the mixed layer never drops "
            "below ``kmix0``. A negative value uses the meteorology's own PBL "
            "height where the met files carry one."
        ),
        json_schema_extra=ZICONTROL,
    )
    siguverr: float | None = Field(
        None,
        description="Standard deviation of the horizontal wind error, in m/s.",
        json_schema_extra=WINDERR,
    )
    tluverr: float | None = Field(
        None,
        description="Correlation timescale of the horizontal wind error, in minutes.",
        json_schema_extra=WINDERR,
    )
    zcoruverr: float | None = Field(
        None,
        description="Vertical correlation length of the horizontal wind error, in meters.",
        json_schema_extra=WINDERR,
    )
    horcoruverr: float | None = Field(
        None,
        description="Horizontal correlation length of the horizontal wind error, in km.",
        json_schema_extra=WINDERR,
    )
    sigzierr: float | None = Field(
        None,
        description="Standard deviation of the mixed-layer height error, in percent.",
        json_schema_extra=ZIERR,
    )
    tlzierr: float | None = Field(
        None,
        description="Correlation timescale of the mixed-layer height error, in minutes.",
        json_schema_extra=ZIERR,
    )
    horcorzierr: float | None = Field(
        None,
        description="Horizontal correlation length of the mixed-layer height error, in km.",
        json_schema_extra=ZIERR,
    )

    @field_validator("ziscale", mode="before")
    @classmethod
    def _flatten_ziscale(cls, value: Any) -> Any:
        """
        Read STILT-R's one-element nested list, ``[[0.8, 0.9]]``, as ``[0.8, 0.9]``.

        Both mean the same factors, so they must hash the same.
        """
        if isinstance(value, list) and value and isinstance(value[0], list):
            if len(value) != 1:
                raise ValueError(
                    "Per-simulation ziscale lists are not supported. Pass one shared "
                    "list of hourly factors for all simulations."
                )
            return value[0]
        return value

    @model_validator(mode="after")
    def _validate_error_params(self) -> Self:
        """Require each error group to be set in full or not at all."""
        for name, fields in (("XY", fields_in("WINDERR")), ("ZI", fields_in("ZIERR"))):
            unset = [getattr(self, f) is None for f in fields]
            if any(unset) and not all(unset):
                raise ValueError(
                    f"Inconsistent {name} error parameters: all must be set or all None"
                )
        return self

    @model_validator(mode="after")
    def _validate_ziscale(self) -> Self:
        """Reject ziscale values that are empty, zero, or longer than 150 hours."""
        if isinstance(self.ziscale, int | float):
            hourly = [float(self.ziscale)] * max(abs(self.n_hours), 1)
        else:
            hourly = [float(v) for v in self.ziscale]
        if not hourly:
            raise ValueError("ziscale cannot be empty; use 1.0 for no scaling.")
        if any(v == 0.0 for v in hourly):
            raise ValueError(
                "ziscale of 0 would collapse the mixed layer to kmix0. STILT-R "
                "uses 0 to mean unset; use 1.0 for no scaling."
            )
        if any(v != 1.0 for v in hourly) and len(hourly) > MAX_ZISCALE_HOURS:
            raise ValueError(
                f"ziscale gives {len(hourly)} hourly factors, but HYSPLIT holds at "
                f"most {MAX_ZISCALE_HOURS}. A scalar ziscale is repeated for "
                "every hour, so it needs abs(n_hours) <= "
                f"{MAX_ZISCALE_HOURS}; for longer runs give a list of up to "
                f"{MAX_ZISCALE_HOURS} factors, after which the mixed layer "
                "is unscaled."
            )
        return self

    @model_validator(mode="after")
    def _validate_seed(self) -> Self:
        """Require ``krand=2`` when a seed is set."""
        if self.seed is not None and self.krand != 2:
            raise ValueError(
                f"seed={self.seed} requires krand=2 (got krand={self.krand}): the "
                "bundled hycs_std discards the namelist seed under krand=4 and "
                "10-13 and uses it only for the initial turbulent velocity under "
                "krand=1. Set krand=2, or drop the seed."
            )
        return self

    @model_validator(mode="after")
    def _validate_hnf_plume(self) -> Self:
        """Require the variables the near-field plume model reads when ``hnf_plume`` is on."""
        if self.hnf_plume:
            required = {"dens", "samt", "sigw", "tlgr", "foot", "mlht"}
            missing = required - set(self.varsiwant)
            if missing:
                raise ValueError(
                    f"hnf_plume=True requires varsiwant to include: {sorted(missing)}"
                )
        return self


def fields_in(file: str) -> list[str]:
    """
    Return the names of the settings that go to *file*, in declaration order.

    *file* is ``"SETUP.CFG"``, ``"CONTROL"``, ``"ZICONTROL"``, ``"WINDERR"``,
    ``"ZIERR"``, or ``"PYSTILT"`` for the settings PYSTILT uses itself.
    """
    return [
        name
        for name, info in TransportParams.model_fields.items()
        if isinstance(info.json_schema_extra, dict)
        and info.json_schema_extra.get("file") == file
    ]


__all__ = ["MAX_ZISCALE_HOURS", "TransportParams", "fields_in"]
