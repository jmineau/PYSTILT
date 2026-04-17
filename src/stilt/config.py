"""Configuration models and YAML I/O utilities for STILT runs.

All STILT/HYSPLIT parameters live in a flat ``STILTParams`` base model.
``ModelConfig`` inherits from it and adds met/footprint/grid definitions.
Descriptions on each :func:`pydantic.Field` are the canonical source for docs,
while field metadata tags drive HYSPLIT namelist generation and public/internal
visibility.
"""

import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, TypeVar, cast

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

T = TypeVar("T", bound=BaseModel)

__all__ = [
    "Bounds",
    "FirstOrderLifetimeTransformSpec",
    "FootprintConfig",
    "Grid",
    "MetConfig",
    "ModelConfig",
    "ModelParams",
    "ParticleTransformSpec",
    "STILTParams",
    "TransportParams",
    "ErrorParams",
    "VerticalOperatorTransformSpec",
    "VerticalReference",
    "build_control_entries",
    "build_setup_entries",
    "cfg_field",
    "iter_documented_config_fields",
    "kmsl_from_vertical_reference",
    "validate_vertical_reference",
]


_EXCLUDE_FROM_YAML: set[str] = set()
_MISSING = object()
ConfigVisibility = Literal["public", "advanced", "internal"]
VerticalReference = Literal["agl", "msl"]


def validate_vertical_reference(reference: str) -> VerticalReference:
    """Return a normalized vertical reference or raise for invalid input."""
    normalized = reference.lower()
    if normalized not in {"agl", "msl"}:
        raise ValueError(
            f"Vertical reference must be 'agl' or 'msl'. Got {reference!r}."
        )
    return cast(VerticalReference, normalized)


def kmsl_from_vertical_reference(reference: VerticalReference) -> int:
    """Map a vertical reference onto the HYSPLIT ``KMSL`` control value."""
    return 0 if reference == "agl" else 1


def cfg_field(
    default: Any = _MISSING,
    *,
    description: str,
    visibility: ConfigVisibility = "public",
    target: str | None = None,
    namelist: str | None = None,
    default_factory: Callable[[], Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Create a Field with shared config metadata.

    Parameters
    ----------
    default : object, optional
        Field default. Use ``...`` for required fields. Omit when passing
        ``default_factory``.
    description : str
        Canonical parameter description used in generated docs.
    visibility : {"public", "advanced", "internal"}, default "public"
        Documentation visibility bucket for this field.
    target : str, optional
        Output target such as ``"setup"``, ``"control"``, or ``"zicontrol"``.
    namelist : str, optional
        Explicit HYSPLIT namelist variable name. Defaults to the field name.
    default_factory : callable, optional
        Pydantic default factory.
    """
    if default_factory is not None and default is not _MISSING:
        raise TypeError(
            "cfg_field() accepts either default or default_factory, not both."
        )

    meta = dict(kwargs.pop("json_schema_extra", {}) or {})
    meta["visibility"] = visibility
    if target is not None:
        meta["target"] = target
    if namelist is not None:
        meta["namelist"] = namelist

    field_kwargs = {
        "description": description,
        "json_schema_extra": meta,
        **kwargs,
    }
    if default_factory is not None:
        return Field(default_factory=default_factory, **field_kwargs)
    if default is _MISSING:
        default = ...
    return Field(default, **field_kwargs)


def _field_meta(field: Any) -> dict[str, Any]:
    """Return normalized metadata stored on a Pydantic field."""
    return dict(field.json_schema_extra or {})


class Bounds(BaseModel):
    """Immutable geographic bounding box used for spatial subsetting."""

    model_config = ConfigDict(frozen=True)

    xmin: float = cfg_field(
        ...,
        description="Minimum x-coordinate (longitude for geographic CRS).",
    )
    xmax: float = cfg_field(
        ...,
        description="Maximum x-coordinate (longitude for geographic CRS).",
    )
    ymin: float = cfg_field(
        ...,
        description="Minimum y-coordinate (latitude for geographic CRS).",
    )
    ymax: float = cfg_field(
        ...,
        description="Maximum y-coordinate (latitude for geographic CRS).",
    )
    projection: str = cfg_field(
        "+proj=longlat",
        description="PROJ string defining the coordinate reference system.",
    )


class Grid(Bounds):
    """Gridded spatial domain for footprint computation.

    Extends :class:`Bounds` with a cell resolution.
    """

    model_config = ConfigDict(frozen=True)

    xres: float = cfg_field(
        ...,
        description="Cell width in the projection's x units (degrees for geographic CRS).",
    )
    yres: float = cfg_field(
        ...,
        description="Cell height in the projection's y units (degrees for geographic CRS).",
    )

    @property
    def resolution(self) -> str:
        """Human-readable cell resolution string, e.g. ``'0.01x0.01'``."""
        return f"{self.xres}x{self.yres}"


class MetConfig(BaseModel):
    """Meteorology file discovery and optional subgridding.

    .. note::
       Subgrid cropping (``subgrid_enable``, ``subgrid_bounds``, etc.) is not
       yet implemented. PYSTILT will leverage future ``arl-met`` cropping and
       ARL-writing support here; until then, setting any subgrid field raises
       :exc:`ValueError` at construction time.
    """

    directory: Path = cfg_field(
        ...,
        description="Directory containing ARL meteorology files for this met stream.",
    )
    file_format: str = cfg_field(
        ...,
        description="Datetime format string used to discover meteorology filenames.",
    )
    file_tres: str = cfg_field(
        ...,
        description="Nominal time spacing between meteorology files.",
    )
    n_min: int = cfg_field(
        1,
        description="Minimum number of meteorology files required for a run.",
    )

    # --- Sub-grid cropping (reserved for future arl-met cropping/writing support) ---
    subgrid_enable: Path | bool = cfg_field(
        False,
        description="Enable meteorology subgridding before the run.",
        visibility="advanced",
    )
    subgrid_bounds: Bounds | None = cfg_field(
        None,
        description="Bounds used for meteorology subgridding.",
        visibility="advanced",
    )
    subgrid_buffer: float = cfg_field(
        0.2,
        description="Buffer added around the receptor domain when subgridding meteorology.",
        visibility="advanced",
    )
    subgrid_levels: int | None = cfg_field(
        None,
        description="Number of vertical levels to keep when subgridding meteorology.",
        visibility="advanced",
    )

    @model_validator(mode="after")
    def _no_subgrid(self) -> "MetConfig":
        """Reject subgrid settings until meteorological clipping is implemented."""
        if self.subgrid_enable is not False:
            raise ValueError(
                "subgrid_enable is not yet implemented. PYSTILT will rely on "
                "future arl-met cropping and ARL-writing support here. Leave "
                "subgrid_enable unset (default False)."
            )
        if self.subgrid_bounds is not None:
            raise ValueError(
                "subgrid_bounds is not yet implemented. PYSTILT will rely on "
                "future arl-met cropping and ARL-writing support here. Leave "
                "subgrid_bounds unset."
            )
        return self


class VerticalOperatorTransformSpec(BaseModel):
    """Declarative built-in transform for applying a vertical operator."""

    kind: Literal["vertical_operator"]
    mode: Literal["none", "uniform", "ak", "pwf", "ak_pwf", "integration", "tccon"]
    levels: list[float] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)
    pressure_levels: list[float] = Field(default_factory=list)
    coordinate: str = "xhgt"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_operator_shape(self) -> Self:
        """Validate that operator level and value arrays have matching lengths."""
        if self.mode not in {"none", "uniform"}:
            if not self.levels or not self.values:
                raise ValueError(
                    "Vertical operator transforms require non-empty levels and "
                    "values unless mode is 'none' or 'uniform'."
                )
            if len(self.levels) != len(self.values):
                raise ValueError(
                    "Vertical operator transform levels and values must have the "
                    "same length."
                )
        return self


class FirstOrderLifetimeTransformSpec(BaseModel):
    """Declarative built-in transform for first-order lifetime decay."""

    kind: Literal["first_order_lifetime"]
    lifetime_hours: float = Field(gt=0)
    time_column: str = "time"
    time_unit: str = "min"
    metadata: dict[str, Any] = Field(default_factory=dict)


ParticleTransformSpec = Annotated[
    VerticalOperatorTransformSpec | FirstOrderLifetimeTransformSpec,
    Field(discriminator="kind"),
]


class FootprintConfig(BaseModel):
    """Settings for a single named footprint product."""

    model_config = ConfigDict(frozen=True)

    grid: Grid = cfg_field(
        ...,
        description="Spatial domain and resolution for the footprint.",
    )
    smooth_factor: float = cfg_field(
        1.0,
        description="Factor by which to linearly scale footprint smoothing. Defaults to 1",
    )
    time_integrate: bool = cfg_field(
        False,
        description="If True, sum the footprint over all time steps to produce a single 2-D layer.",
    )
    error: bool = cfg_field(
        False,
        description=(
            "If True, also compute an error footprint from the error trajectories "
            'and store it alongside the main footprint under "{name}_error".'
        ),
        visibility="advanced",
    )
    transforms: list[ParticleTransformSpec] = cfg_field(
        description="Declarative particle transforms applied before rasterizing the footprint.",
        default_factory=list,
        visibility="advanced",
    )


def foot_names(foot_configs: dict[str, FootprintConfig]) -> list[str]:
    """Return all requested footprint artifact names, including error outputs."""
    names: list[str] = []
    for name, cfg in foot_configs.items():
        names.append(name)
        if cfg.error:
            names.append(f"{name}_error")
    return names


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
    """HYSPLIT transport and turbulence parameterization.

    All field names map directly to HYSPLIT namelist variables.  See the
    `HYSPLIT User's Guide <https://www.ready.noaa.gov/HYSPLIT.php>`_ for full
    parameter documentation.
    """

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
        """HYSPLIT WINDERRTF flag encoding active error modes.

        Returns
        -------
        int
            0 = no error runs, 1 = XY only, 2 = ZI only, 3 = both.
        """
        xyerr = all(v is not None for v in self._xyerr_params().values())
        zierr = all(v is not None for v in self._zierr_params().values())
        return xyerr + 2 * zierr


class STILTParams(ModelParams, TransportParams, ErrorParams):
    """All STILT/HYSPLIT parameters in one flat model.

    ``ModelConfig`` inherits from this and adds met/footprint/grid definitions.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _set_maxpar(self) -> Self:
        """Default ``maxpar`` to ``numpar`` when the user omits it."""
        if self.maxpar is None:
            self.maxpar = self.numpar
        return self


class ModelConfig(STILTParams):
    """Project-level config: all STILT params + met/footprint/grid definitions."""

    footprints: dict[str, FootprintConfig] = cfg_field(
        default_factory=dict,
        description="Named footprint products available for this model configuration.",
    )
    grids: dict[str, Grid] = cfg_field(
        default_factory=dict,
        description="Named grids referenced by footprint definitions.",
    )
    mets: dict[str, MetConfig] = cfg_field(
        default_factory=dict,
        description="Named meteorology streams available to the model.",
    )
    execution: dict[str, Any] = cfg_field(
        default_factory=dict,
        description="Execution backend settings such as local, Slurm, or Kubernetes options.",
        visibility="advanced",
    )
    skip_existing: bool = cfg_field(
        True,
        description=(
            "Skip simulations that already have output. "
            "Set False to force re-run all simulations. "
            "Can be overridden at call time via model.run(skip_existing=...)."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _resolve_nested_configs(cls, data: dict) -> dict:
        """Expand named grid references in footprint configs before field validation."""
        if not isinstance(data, dict):
            return data
        grids_raw = data.get("grids") or {}
        fp_raw = data.get("footprints") or {}
        if fp_raw:
            resolved = {}
            for name, cfg in fp_raw.items():
                if isinstance(cfg, dict):
                    cfg = dict(cfg)
                    grid_ref = cfg.get("grid")
                    if isinstance(grid_ref, str):
                        if grid_ref not in grids_raw:
                            raise ValueError(
                                f"Footprint '{name}' references unknown grid '{grid_ref}'"
                            )
                        cfg["grid"] = grids_raw[grid_ref]
                    elif grid_ref is None:
                        raise ValueError(f"Footprint '{name}' is missing a 'grid' key.")
                resolved[name] = cfg
            data = {**data, "footprints": resolved}
        return data

    @model_validator(mode="after")
    def _validate_mets(self) -> Self:
        """Ensure each configured meteorology stream has a unique name."""
        if not self.mets:
            raise ValueError(
                "ModelConfig.mets must contain at least one meteorology configuration"
            )
        bad_keys = [k for k in self.mets if not k.isalnum()]
        if bad_keys:
            raise ValueError(
                f"Met keys must be alphanumeric (no underscores or special chars), got: {bad_keys}"
            )
        return self

    def to_yaml(self, path: str | Path) -> None:
        """Write the model config to a YAML file.

        Parameters
        ----------
        path : str or Path
            Destination path. Parent directories are created if absent.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json", exclude=_EXCLUDE_FROM_YAML)
        for key in ("mets", "grids", "footprints", "execution"):
            if not data.get(key):
                del data[key]  # delete key if empty to keep YAML clean
        with path.open("w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Load a model config from a YAML file.

        Parameters
        ----------
        path : str or Path
            Path to the config YAML file.

        Returns
        -------
        ModelConfig
            Validated config instance. Unknown keys are ignored with a warning.
        """
        path = Path(path)
        with path.open() as f:
            raw: dict = yaml.safe_load(f)
        known = set(cls.model_fields)
        unknown = set(raw) - known
        if unknown:
            warnings.warn(
                f"Unknown config keys ignored: {sorted(unknown)}", stacklevel=2
            )
        return cls.model_validate({k: v for k, v in raw.items() if k in known})


CONFIG_DOC_MODELS: tuple[type[BaseModel], ...] = (
    Bounds,
    Grid,
    MetConfig,
    FootprintConfig,
    ModelParams,
    TransportParams,
    ErrorParams,
    ModelConfig,
)


def iter_documented_config_fields(
    *models: type[BaseModel],
    include_internal: bool = False,
) -> Iterator[tuple[type[BaseModel], str, Any]]:
    """Yield config fields in declaration order for docs or UI generation."""
    if not models:
        models = CONFIG_DOC_MODELS
    for model in models:
        for name, field in model.model_fields.items():
            visibility = _field_meta(field).get("visibility", "public")
            if visibility == "internal" and not include_internal:
                continue
            yield model, name, field


def _collect_target_entries(
    params: BaseModel,
    model: type[BaseModel],
    *,
    target: str,
) -> dict[str, Any]:
    """Collect config fields whose metadata routes them to one output target."""
    entries: dict[str, Any] = {}
    default_target = getattr(model, "DEFAULT_TARGET", None)
    for name, field in model.model_fields.items():
        meta = _field_meta(field)
        field_target = meta.get("target", default_target)
        if field_target != target:
            continue
        key = meta.get("namelist", name)
        entries[key] = getattr(params, name)
    return entries


def build_setup_entries(params: STILTParams) -> dict[str, Any]:
    """Collect fields that belong in HYSPLIT ``SETUP.CFG``."""
    entries: dict[str, Any] = {}
    entries.update(
        _collect_target_entries(
            params,
            ModelParams,
            target="setup",
        )
    )
    entries.update(
        _collect_target_entries(
            params,
            TransportParams,
            target="setup",
        )
    )
    return entries


def build_control_entries(params: STILTParams) -> dict[str, Any]:
    """Collect fields that belong in HYSPLIT ``CONTROL``."""
    return _collect_target_entries(
        params,
        TransportParams,
        target="control",
    )


def _config_or_kwargs(
    config: T | None,
    kwargs: dict,
    cls: type[T],
) -> T | None:
    """Resolve a config-or-kwargs pair, returning *config* or building one from *kwargs*.

    Raises :class:`TypeError` if both are provided.  Returns ``None`` when
    neither is given (caller decides the fallback).
    """
    if config is not None and kwargs:
        raise TypeError(
            f"Cannot pass both a {cls.__name__} instance and keyword arguments."
        )
    if kwargs:
        return cls(**kwargs)
    return config
