"""STILT and HYSPLIT run parameters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelParams(BaseModel):
    """Simulation length, particle count, and particle output settings."""

    n_hours: int = Field(
        -24,
        description="Length of each simulation, in hours. Negative runs backward in time.",
    )
    numpar: int = Field(
        200,
        description=(
            "Number of particles released per simulation. More particles give a "
            "less noisy footprint and take longer to run."
        ),
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
    )
    rm_dat: bool = Field(
        True,
        description=(
            "Delete HYSPLIT's particle files (``PARTICLE_STILT.DAT`` and "
            "``PARTICLE.DAT``) once they have been read, to save disk space."
        ),
    )
    timeout: int | None = Field(
        None,
        description=(
            "Time limit for one ``hycs_std`` run, in seconds. A run that "
            "exceeds it is stopped and recorded as a failed simulation, and the "
            "worker moves on to the next one. Unset waits indefinitely, so a "
            "hung HYSPLIT process can hold a batch worker until its job ends."
        ),
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
    )


class TransportParams(BaseModel):
    """
    HYSPLIT transport and turbulence settings.

    Most of these are ``SETUP.CFG`` namelist entries with HYSPLIT's own
    names. See the HYSPLIT user guide for the full meaning of each.
    """

    capemin: float = Field(
        -1.0,
        description=(
            "Convection option. -1 turns convection off, -2 uses the Grell "
            "scheme, and a positive value mixes vertically when CAPE exceeds "
            "it, in J/kg."
        ),
    )
    cmass: int = Field(
        0,
        description="Compute grid concentrations (0) or grid mass (1).",
    )
    conage: int = Field(
        48, description="Particle age at which particles and puffs convert, in hours."
    )
    cpack: int = Field(1, description="Packing of the binary concentration grid.")
    delt: float = Field(
        1.0,
        description=(
            "Integration time step, in minutes. 0 lets HYSPLIT choose; a "
            "negative value sets the minimum step."
        ),
    )
    dxf: float = Field(
        1.0, description="Horizontal x-grid offset factor for ensemble runs."
    )
    dyf: float = Field(
        1.0, description="Horizontal y-grid offset factor for ensemble runs."
    )
    dzf: float = Field(
        0.01,
        description="Vertical offset factor for ensemble runs (0.01 is about 250 m).",
    )
    efile: str = Field(
        "",
        description="Name of a time-varying emissions file. Blank uses none.",
    )
    emisshrs: float = Field(
        0.01,
        description="Duration of the particle release, in hours.",
    )
    frhmax: float = Field(
        3.0, description="Maximum horizontal puff-rounding parameter."
    )
    frhs: float = Field(
        1.0, description="Horizontal puff-rounding fraction for merging."
    )
    frme: float = Field(0.1, description="Mass-rounding fraction for enhanced merging.")
    frmr: float = Field(0.0, description="Mass-removal fraction for enhanced merging.")
    frts: float = Field(0.1, description="Temporal puff-rounding fraction.")
    frvs: float = Field(0.01, description="Vertical puff-rounding fraction.")
    hscale: float = Field(
        10800.0, description="Horizontal Lagrangian timescale, in seconds."
    )
    ichem: int = Field(
        8,
        description="HYSPLIT chemistry and output mode. 8 is the STILT emulation mode.",
    )
    idsp: int = Field(
        2,
        description="Particle dispersion scheme: 1 for HYSPLIT, 2 for STILT.",
    )
    initd: int = Field(
        0,
        description="Initial distribution as particles, puffs, or a mix. 0 is 3D particles.",
    )
    k10m: int = Field(
        1,
        description=(
            "Use the 10 m winds and 2 m temperature as the lowest meteorology "
            "level (1) or skip them (0)."
        ),
    )
    kagl: int = Field(
        1,
        description="Write trajectory heights above ground (1) or above sea level (0).",
    )
    kbls: int = Field(
        1,
        description=(
            "Derive boundary-layer stability from surface fluxes (1) or from "
            "wind and temperature profiles (2)."
        ),
    )
    kblt: int = Field(
        5,
        description=(
            "Boundary-layer turbulence scheme: 1 Beljaars, 2 Kantha-Clayson, "
            "3 TKE, 4 measured variances, 5 Hanna."
        ),
    )
    kdef: int = Field(
        0,
        description="Horizontal turbulence from vertical mixing (0) or wind deformation (1).",
    )
    khinp: int = Field(
        0,
        description="Age, in hours, given to particles read from ``pinpf``. 0 keeps their own age.",
    )
    khmax: int = Field(
        9999,
        description="Maximum particle or trajectory age, in hours.",
    )
    kmix0: int = Field(150, description="Minimum mixed-layer depth, in meters.")
    kmixd: int = Field(
        3,
        description=(
            "Mixed-layer depth source: 0 from the meteorology, 1 from the "
            "temperature profile, 2 from the TKE profile, 3 from a modified "
            "Richardson number."
        ),
    )
    kmsl: Literal[0, 1] | None = Field(
        None,
        description=(
            "Read release heights as above ground (0) or above sea level (1). "
            "Unset takes it from each receptor's ``altitude_ref``, and a value "
            "that disagrees with a receptor is an error."
        ),
    )
    kpuff: int = Field(
        0, description="Horizontal puff growth: linear (0) or empirical (1)."
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
    )
    krnd: int = Field(6, description="Enhanced-merging interval, in hours.")
    kspl: int = Field(1, description="Standard puff-splitting interval, in hours.")
    kwet: int = Field(
        1,
        description="Precipitation from the meteorology (1) or from an external ARL file (2).",
    )
    kzmix: int = Field(
        0,
        description=(
            "Vertical mixing adjustment: 0 none, 1 a single PBL-average value, "
            "2 scale by ``tvmix``."
        ),
    )
    maxdim: int = Field(
        1,
        description="Maximum number of pollutant species carried on one particle.",
    )
    maxpar: int | None = Field(
        None,
        description="Maximum number of particles in a simulation. Unset uses ``numpar``.",
    )
    mgmin: int = Field(
        10, description="Minimum meteorological subgrid size, in grid points."
    )
    mhrs: int = Field(9999, description="Trajectory restart duration limit, in hours.")
    nbptyp: int = Field(
        1,
        description="Number of particle-size bins per pollutant type.",
    )
    ncycl: int = Field(
        0,
        description="Cycle time of the particle dump file, in hours.",
    )
    ndump: int = Field(
        0,
        description="Interval between particle dumps, in hours. 0 writes none.",
    )
    ninit: int = Field(
        1,
        description=(
            "Particle initialization from ``pinpf``: 0 none, 1 once at the "
            "start, 2 add every hour, 3 replace every hour."
        ),
    )
    nstr: int = Field(0, description="Trajectory restart interval, in hours.")
    nturb: int = Field(
        0,
        description="Turbulence on (0) or off (1).",
    )
    nver: int = Field(0, description="Trajectory vertical split number.")
    outdt: int = Field(
        0,
        description=(
            "Interval between particle outputs in ``PARTICLE_STILT.DAT``, in "
            "minutes. 0 writes every time step and a negative value writes none."
        ),
    )
    p10f: float = Field(1.0, description="Dust threshold velocity sensitivity factor.")
    pinbc: str = Field(
        "",
        description="Particle input file for time-varying boundary conditions.",
    )
    pinpf: str = Field(
        "",
        description="Particle input file for initialization or boundary-condition runs.",
    )
    poutf: str = Field(
        "",
        description="Particle output file name.",
    )
    qcycle: float = Field(
        0.0, description="Emission cycling period, in hours. 0 turns cycling off."
    )
    rhb: int = Field(
        80,
        description="Relative humidity that defines a cloud base, in percent.",
    )
    rht: int = Field(
        60,
        description="Relative humidity that defines a cloud top, in percent.",
    )
    splitf: float = Field(
        1.0,
        description=(
            "Factor for the automatic horizontal splitting size. A negative "
            "value turns the automatic sizing off."
        ),
    )
    tkerd: float = Field(
        0.18, description="Ratio w'²/(u'²+v'²) of TKE components when unstable."
    )
    tkern: float = Field(
        0.18, description="Ratio w'²/(u'²+v'²) of TKE components when stable."
    )
    tlfrac: float = Field(
        0.1,
        description=(
            "Fraction of the vertical Lagrangian timescale used as the time "
            "step of the STILT dispersion scheme."
        ),
    )
    tout: int = Field(
        0,
        description="Trajectory output interval, in minutes.",
    )
    tratio: float = Field(
        0.75,
        description="Advection stability ratio (fraction of a grid cell per time step).",
    )
    tvmix: float = Field(
        1.0,
        description="Vertical mixing scale factor, used by the ``kzmix`` scaling modes.",
    )
    veght: float = Field(
        0.5,
        description=(
            "Height below which a particle's time counts toward the footprint. "
            "A value of 1 or less is a fraction of the mixed-layer height; a "
            "larger value is meters above ground."
        ),
    )
    vscale: float = Field(
        200.0,
        description="Vertical Lagrangian timescale, in seconds.",
    )
    vscaleu: float = Field(
        200.0,
        description="Vertical Lagrangian timescale in an unstable boundary layer, in seconds.",
    )
    vscales: float = Field(
        -1.0,
        description=(
            "Vertical Lagrangian timescale in a stable boundary layer, in "
            "seconds. -1 uses the Hanna timescale, which varies with the "
            "turbulence, and then ``vscaleu`` is not used."
        ),
    )
    w_option: int = Field(
        0,
        description=(
            "Vertical motion method: 0 the meteorology's vertical velocity, "
            "1 isobaric, 2 isentropic, 3 constant density, 4 constant sigma."
        ),
    )
    wbbh: float = Field(
        0.0,
        description=(
            "Height at which the fixed vertical velocity switches from rise to "
            "fall, in meters. Used by vertical motion option 9."
        ),
    )
    wbwf: float = Field(
        0.0,
        description="Fixed fall velocity, in m/s. Used by vertical motion options 9 and 10.",
    )
    wbwr: float = Field(
        0.0,
        description="Fixed rise velocity, in m/s. Used by vertical motion option 9.",
    )
    wvert: bool = Field(
        False,
        description="Interpolate WRF fields vertically with the WRF scheme instead of HYSPLIT's.",
    )
    z_top: float = Field(
        25000.0,
        description="Top of the model domain, in meters above ground.",
    )
    ziscale: float | list[float] | list[list[float]] = Field(
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
    )


class ErrorParams(BaseModel):
    """
    Transport-error settings for perturbed runs.

    Setting the four wind-error fields perturbs the particles' winds (HYSPLIT's
    ``WINDERR`` file). Setting the three mixed-layer fields perturbs each
    particle's footprint by a random mixed-layer height error (``ZIERR``).
    Each group must be set in full or not at all.
    """

    siguverr: float | None = Field(
        None,
        description="Standard deviation of the horizontal wind error, in m/s.",
    )
    tluverr: float | None = Field(
        None,
        description="Correlation timescale of the horizontal wind error, in minutes.",
    )
    zcoruverr: float | None = Field(
        None,
        description="Vertical correlation length of the horizontal wind error, in meters.",
    )
    horcoruverr: float | None = Field(
        None,
        description="Horizontal correlation length of the horizontal wind error, in km.",
    )
    sigzierr: float | None = Field(
        None,
        description="Standard deviation of the mixed-layer height error, in percent.",
    )
    tlzierr: float | None = Field(
        None,
        description="Correlation timescale of the mixed-layer height error, in minutes.",
    )
    horcorzierr: float | None = Field(
        None,
        description="Horizontal correlation length of the mixed-layer height error, in km.",
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
        """Require each error group to be set in full or not at all."""
        for name, fields in (("XY", self.XYERR_PARAMS), ("ZI", self.ZIERR_PARAMS)):
            unset = [getattr(self, f) is None for f in fields]
            if any(unset) and not all(unset):
                raise ValueError(
                    f"Inconsistent {name} error parameters: all must be set or all None"
                )
        return self

    @property
    def winderr(self) -> list[float] | None:
        """The wind-error values in ``WINDERR`` order, or ``None`` when unset."""
        values = [getattr(self, f) for f in self.XYERR_PARAMS]
        return None if values[0] is None else values

    @property
    def zierr(self) -> list[float] | None:
        """The mixed-layer error values in ``ZIERR`` order, or ``None`` when unset."""
        values = [getattr(self, f) for f in self.ZIERR_PARAMS]
        return None if values[0] is None else values

    @property
    def winderrtf(self) -> int:
        """HYSPLIT ``WINDERRTF`` flag: 1 for wind errors, 2 for mixed-layer errors, 3 for both."""
        return (self.winderr is not None) + 2 * (self.zierr is not None)

    @property
    def error_enabled(self) -> bool:
        """Whether wind or mixed-layer errors are set, making this a perturbed run."""
        return self.winderrtf > 0


class STILTParams(ModelParams, TransportParams, ErrorParams):
    """
    All STILT and HYSPLIT run parameters in one flat model.

    Each :class:`TransportParams` field is written to ``SETUP.CFG``, except
    those in ``CONTROL_FIELDS`` (written to ``CONTROL``) and ``ziscale``
    (written to ``ZICONTROL``). The :class:`ErrorParams` fields are written to
    ``WINDERR`` and ``ZIERR``.
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
        """Return the ``SETUP.CFG`` namelist entries, leaving out unset fields."""
        names = [
            *(n for n in ModelParams.model_fields if n in self._MODEL_SETUP_FIELDS),
            *(
                n
                for n in TransportParams.model_fields
                if n not in self.CONTROL_FIELDS and n not in self.ZICONTROL_FIELDS
            ),
        ]
        entries = {n: getattr(self, n) for n in names if getattr(self, n) is not None}
        entries.setdefault("maxpar", self.numpar)
        entries["zicontroltf"] = self.zicontroltf
        if self.seed is not None:
            entries["seed"] = self.setup_seed(self.seed)
        return entries

    @staticmethod
    def setup_seed(seed: int) -> int:
        """
        Return the ``SEED`` value written to ``SETUP.CFG`` for a user seed.

        HYSPLIT sets its generator state to ``-1 + SEED``. Under ``krand=2``
        it reinitializes only from a negative state, and every state of -1 or
        more gives the same stream. Writing ``-(|seed| + 1)`` puts the state at
        ``-(|seed| + 2)``. That is negative, different for each ``|seed|``, and
        never the unseeded default (``SEED = 0``). A patched HYSPLIT that uses
        ``SEED`` directly maps a negative ``SEED`` to the same state, so the
        value works with both builds.
        """
        return -(abs(seed) + 1)

    def realization_seed(self, realization: int) -> int | None:
        """
        Return the seed for one realization of a variant.

        Realization ``k`` runs with ``seed + k``, so realization 0 uses the
        configured seed, as STILT-R's single error run does. Returns ``None``
        when no seed is set.
        """
        if self.seed is None:
            return None
        return self.seed + realization

    @property
    def ziscale_factors(self) -> list[float] | None:
        """
        Hourly mixed-layer factors for ``ZICONTROL``, or ``None`` when unscaled.

        A single ``ziscale`` value is repeated for every hour of the run and a
        list is used as given. Factors that are all 1.0 give ``None``.
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
        """HYSPLIT ``ZICONTROLTF`` flag, 1 when ``ziscale`` scales the mixed layer."""
        return int(self.ziscale_factors is not None)

    @model_validator(mode="after")
    def _validate_ziscale(self) -> Self:
        """Reject ``ziscale`` values that are empty, zero, or longer than 150 hours."""
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
