from abc import ABC
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from stilt.receptors import Receptor


class Resolution(BaseModel):
    xres: float
    yres: float

    def __str__(self) -> str:
        return f"{self.xres}x{self.yres}"


class SystemParams(BaseModel):
    stilt_wd: Path
    output_wd: Path | None = None
    lib_loc: Path | int | None = None

    @model_validator(mode="after")
    def _set_system_defaults(self) -> Self:
        """Set default values for system parameters."""

        if self.output_wd is None:
            self.output_wd = self.stilt_wd / "out"

        return self


class FootprintParams(BaseModel):
    hnf_plume: bool = True
    projection: str = "+proj=longlat"
    smooth_factor: float = 1.0
    time_integrate: bool = False
    xmn: float | None = None
    xmx: float | None = None
    xres: float | list[float] | None = None
    ymn: float | None = None
    ymx: float | None = None
    yres: float | list[float] | None = None

    @model_validator(mode="after")
    def _set_footprint_defaults(self) -> Self:
        """Set default values for footprint parameters."""
        if self.yres is None:
            self.yres = self.xres
        return self

    @model_validator(mode="after")
    def _validate_footprint_params(self) -> Self:
        """Validate footprint parameters."""

        if not isinstance(self.xres, type(self.yres)):
            raise ValueError("xres and yres must both be of the same type.")

        def length(res):
            if res is None:
                return 0
            if isinstance(res, list):
                return len(res)
            return 1

        xlen = length(self.xres)
        ylen = length(self.yres)

        if xlen != ylen:
            raise ValueError("xres and yres must have the same length.")

        return self

    @property
    def resolutions(self) -> list[Resolution] | None:
        """Get the x and y resolutions as a list of tuples."""
        if self.xres is None:
            return None
        if not isinstance(self.xres, list):
            self.xres = [self.xres]
            self.yres = [self.yres]
        return [
            Resolution(xres=xres, yres=yres)
            for xres, yres in zip(self.xres, self.yres, strict=False)
        ]


class MetParams(BaseModel):
    met_path: Path
    met_file_format: str
    met_file_tres: str
    met_subgrid_buffer: float = 0.1
    met_subgrid_enable: bool = False
    met_subgrid_levels: int | None = None
    n_met_min: int = 1


class ModelParams(BaseModel):
    n_hours: int = -24
    numpar: int = 1000
    rm_dat: bool = True
    run_foot: bool = True
    run_trajec: bool = True
    simulation_id: str | list[str] | None = None
    timeout: int = 3600
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
        ]
    )

    @model_validator(mode="after")
    def _validate_run_flags(self) -> Self:
        """Ensure at least one of `run_trajec` or `run_foot` is True."""
        if not self.run_trajec and not self.run_foot:
            raise ValueError("Nothing to do: set `run_trajec` or `run_foot` to True")
        return self


class TransportParams(BaseModel):
    capemin: float = -1.0
    cmass: int = 0
    conage: int = 48
    cpack: int = 1
    delt: int = 1
    dxf: int = 1
    dyf: int = 1
    dzf: float = 0.01
    efile: str = ""
    emisshrs: float = 0.01
    frhmax: float = 3.0
    frhs: float = 1.0
    frme: float = 0.1
    frmr: float = 0.0
    frts: float = 0.1
    frvs: float = 0.1
    hscale: int = 10800
    ichem: int = 8
    idsp: int = 2
    initd: int = 0
    k10m: int = 1
    kagl: int = 1
    kbls: int = 1
    kblt: int = 5
    kdef: int = 0
    khinp: int = 0
    khmax: int = 9999
    kmix0: int = 250
    kmixd: int = 3
    kmsl: int = 0
    kpuff: int = 0
    krand: int = 4
    krnd: int = 6
    kspl: int = 1
    kwet: int = 1
    kzmix: int = 0
    maxdim: int = 1
    maxpar: int | None = None
    mgmin: int = 10
    mhrs: int = 9999
    nbptyp: int = 1
    ncycl: int = 0
    ndump: int = 0
    ninit: int = 1
    nstr: int = 0
    nturb: int = 0
    nver: int = 0
    outdt: int = 0
    p10f: int = 1
    pinbc: str = ""
    pinpf: str = ""
    poutf: str = ""
    qcycle: int = 0
    rhb: float = 80.0
    rht: float = 60.0
    splitf: int = 1
    tkerd: float = 0.18
    tkern: float = 0.18
    tlfrac: float = 0.1
    tout: float = 0.0
    tratio: float = 0.75
    tvmix: float = 1.0
    veght: float = 0.5
    vscale: int = 200
    vscaleu: int = 200
    vscales: int = -1
    w_option: int = 0
    wbbh: int = 0
    wbwf: int = 0
    wbwr: int = 0
    wvert: bool = False
    z_top: float = 25000.0
    zicontroltf: int = 0
    ziscale: int | list[int] = 0


class ErrorParams(BaseModel):
    siguverr: float | None = None
    tluverr: float | None = None
    zcoruverr: float | None = None
    horcoruverr: float | None = None
    sigzierr: float | None = None
    tlzierr: float | None = None
    horcorzierr: float | None = None

    XYERR_PARAMS: ClassVar[tuple[str, ...]] = (
        "siguverr",
        "tluverr",
        "zcoruverr",
        "horcoruverr",
    )
    ZIERR_PARAMS: ClassVar[tuple[str, ...]] = ("sigzierr", "tlzierr", "horcorzierr")

    @model_validator(mode="after")
    def _validate_error_params(self) -> Self:
        """
        Validate error parameters to ensure they are either all set or all None
        """
        xy_params = self.xyerr_params()
        zi_params = self.zierr_params()

        for name, params in [("XY", xy_params), ("ZI", zi_params)]:
            is_na = [pd.isna(v) for v in params.values()]
            if any(is_na) and not all(is_na):
                raise ValueError(
                    f"Inconsistent {name} error parameters: all must be set or all must be None"
                )

        return self

    def xyerr_params(self) -> dict[str, float | None]:
        """
        Get the XY error parameters as a dictionary.
        """
        return {param: getattr(self, param) for param in self.XYERR_PARAMS}

    def zierr_params(self) -> dict[str, float | None]:
        """
        Get the ZI error parameters as a dictionary.
        """
        return {param: getattr(self, param) for param in self.ZIERR_PARAMS}

    @property
    def winderrtf(self) -> int:
        """
        Determine the winderrtf flag based on the presence of error parameters.

        Returns
        -------
        int
            Wind error control flag.
                0 : No error parameters are set
                1 : ZI error parameters are set
                2 : XY error parameters are set
                3 : Both XY and ZI error parameters are set
        """
        xyerr = all(self.xyerr_params().values())
        zierr = all(self.zierr_params().values())

        return 2 * xyerr + zierr


class UserFuncParams(BaseModel):
    before_footprint: Callable | Path | None = None

    @field_validator("before_footprint", mode="before")
    @classmethod
    def _load_before_footprint(cls, v: Any) -> Any:
        """Ensure before_footprint is a callable or None."""
        if isinstance(v, str | Path):
            # Load the function from the specified path
            p = Path(v)

            if p.suffix.lower().endswith("r"):
                # Pass the R path
                return v
            elif p.suffix.lower().endswith("py"):
                # Load the Python function
                raise NotImplementedError(
                    "Loading Python functions from file is not implemented yet."
                )
            else:
                raise ValueError(f"Unsupported file type: {p.suffix}")
        return v


class BaseConfig(
    ABC,
    SystemParams,
    FootprintParams,
    MetParams,
    ModelParams,
    TransportParams,
    ErrorParams,
    UserFuncParams,
):
    """
    STILT Configuration

    This class consolidates all configuration parameters for the STILT model,
    including system settings, footprint parameters, meteorological data,
    model specifics, transport settings, error handling, and user-defined
    functions.
    """

    # Allows Pydantic to work with custom classes like Receptor
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    def _load_yaml_params(path: str | Path) -> dict[str, Any]:
        """
        Load a YAML config file and return its contents as a dictionary.
        """
        with Path(path).open() as f:
            config = yaml.safe_load(f)

        # Flatten the config dictionary
        params = {}
        for key, value in config.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    params[f"{subkey}"] = subvalue
            else:
                params[key] = value

        return params

    @classmethod
    def from_path(cls, path: str | Path) -> Self:
        """
        Load STILT configuration from a YAML file.
        """
        params = cls._load_yaml_params(path)
        return cls(**params)

    @model_validator(mode="after")
    def _validate_base_config(self) -> Self:
        """Perform validation that depends on multiple fields."""

        # Check if there's anything to run
        if not self.run_trajec and not self.run_foot:
            raise ValueError("Nothing to do: set run_trajec or run_foot to True")

        # Check for grid parameters if running footprint or subgrid met
        if self.run_foot or self.met_subgrid_enable:
            required_grid_params = ["xmn", "xmx", "xres", "ymn", "ymx"]
            if any(getattr(self, arg) is None for arg in required_grid_params):
                raise ValueError(
                    "xmn, xmx, xres, ymn, and ymx must be specified when "
                    "met_subgrid_enable or run_foot is True"
                )

        return self

    @model_validator(mode="after")
    def _set_config_defaults(self) -> Self:
        """Set default values for configuration parameters."""

        # Set default for maxpar if not provided
        if self.maxpar is None:
            self.maxpar = self.numpar

        return self

    def system_params(self) -> dict[str, Any]:
        return {attr: getattr(self, attr) for attr in SystemParams.model_fields}

    def footprint_params(self) -> dict[str, Any]:
        return {attr: getattr(self, attr) for attr in FootprintParams.model_fields}

    def met_params(self) -> dict[str, Any]:
        return {attr: getattr(self, attr) for attr in MetParams.model_fields}

    def model_params(self) -> dict[str, Any]:
        return {attr: getattr(self, attr) for attr in ModelParams.model_fields}

    def transport_params(self) -> dict[str, Any]:
        return {attr: getattr(self, attr) for attr in TransportParams.model_fields}

    def error_params(self) -> dict[str, Any]:
        return {attr: getattr(self, attr) for attr in ErrorParams.model_fields}

    def user_funcs(self) -> dict[str, Any]:
        return {attr: getattr(self, attr) for attr in UserFuncParams.model_fields}


class SimulationConfig(BaseConfig):
    receptor: Receptor

    @classmethod
    def from_path(cls, path: str | Path) -> Self:
        # Open simulation config like a model config
        model_config = ModelConfig.from_path(path)
        # Then extract the receptor
        receptor = model_config.receptors[0]
        return cls(receptor=receptor, **model_config.model_dump())

    @field_validator("simulation_id", mode="after")
    @classmethod
    def _validate_simulation_id(cls, simulation_id) -> str:
        if not simulation_id:
            simulation_id = cls.receptor.id
        elif not isinstance(simulation_id, str):
            raise TypeError("simulation_id must be a string")
        return simulation_id

    def to_model_config(self) -> "ModelConfig":
        config = self.model_dump()
        receptor = config.pop("receptor")
        return ModelConfig(receptors=[receptor], **config)


class ModelConfig(BaseConfig):
    receptors: list[Receptor]

    @classmethod
    def from_path(cls, path: str | Path) -> Self:
        params = cls._load_yaml_params(path)
        if "stilt_wd" not in params:
            params["stilt_wd"] = Path(path).parent
        return cls(**params)

    @model_validator(mode="before")
    @classmethod
    def _load_receptors(cls, data) -> Self:
        """
        Validates and loads receptors. If a path is provided, it loads
        receptors from the corresponding CSV file.
        """
        receptors = data.get("receptors")
        if isinstance(receptors, str | Path):
            # If the input is a path, load from the file.
            receptor_path = Path(receptors)
            if not receptor_path.is_absolute():
                receptor_path = Path(data.get("stilt_wd")) / receptor_path
            data["receptors"] = Receptor.load_receptors_from_csv(receptor_path)
        return data

    @model_validator(mode="after")
    def _validate_model_config(self) -> Self:
        """Validate the model configuration."""

        # Check if simulation_id is set
        if isinstance(self.simulation_id, str) and len(self.receptors) > 1:
            raise ValueError(
                "Simulation ID must be specified for each receptor or be left blank."
            )

        return self

    def to_file(self):
        # Write out receptor information to csv
        # Write out config
        raise NotImplementedError

    def build_simulation_configs(self) -> list[SimulationConfig]:
        """
        Build a list of SimulationConfig objects, one for each receptor.
        """
        raise NotImplementedError
        config = self.model_dump()
        receptors = config.pop("receptors")
        simulation_id = config.pop("simulation_id")
        if isinstance(simulation_id, list):
            # TODO
            pass

        return [SimulationConfig(receptor=receptor, **config) for receptor in receptors]
