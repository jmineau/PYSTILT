import datetime as dt
import re
from collections import UserDict
from pathlib import Path

import pandas as pd
import xarray as xr
from typing_extensions import Self

from stilt.config import SimulationConfig
from stilt.receptors import Receptor
from stilt.trajectory import Output


class Footprint(Output):
    """STILT footprint."""

    # Maybe in the future we will inherit or replicate BaseGrid functionality
    # super(BaseGrid).__init__(data=self.data, crs=self.crs)
    # thinking now that this would be implmented in a subclass to keep this class cleaner

    def __init__(
        self,
        simulation_id: str,
        receptor: Receptor,
        data: xr.DataArray,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        xres: float,
        yres: float,
        projection: str = "+proj=longlat",
        smooth_factor: float = 1.0,
        time_integrate: bool = False,
    ):
        super().__init__(simulation_id=simulation_id, receptor=receptor, data=data)
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.xres = xres
        self.yres = yres
        self.projection = projection
        self.smooth_factor = smooth_factor
        self.time_integrate = time_integrate

    @property
    def resolution(self) -> str:
        return f"{self.xres}x{self.yres}"

    @property
    def id(self) -> str:
        return f"{self.simulation_id}_{self.resolution}_foot"

    @property
    def time_range(self) -> tuple[dt.datetime, dt.datetime]:
        """
        Get time range of footprint data.

        Returns
        -------
        tuple[pd.Timestamp, pd.Timestamp]
            Time range of footprint data.
        """
        times = sorted(self.data.time.values)
        start = pd.Timestamp(times[0]).to_pydatetime()
        stop = pd.Timestamp(times[-1]) + pd.Timedelta(hours=1)
        return start, stop.to_pydatetime()

    @classmethod
    def from_path(cls, path: str | Path, **kwargs) -> Self:
        """
        Create Footprint object from netCDF file.

        Parameters
        ----------
        path : str | Path
            Path to netCDF file.
        **kwargs : dict
            Additional keyword arguments for xr.open_dataset.

        Returns
        -------
        Footprint
            Footprint object.
        """
        # Resolve the file path
        path = Path(path).resolve()

        # Build the configuration from the file path
        config = SimulationConfig.from_path(
            path.parent / f"{path.parent.name}_config.yaml"
        )

        # Read the netCDF file, parsing the receptor
        data = cls.read_netcdf(path, parse_receptor=True, **kwargs)

        # Assert that the footprint receptor matches the config receptor
        if data.attrs["receptor"] != config.receptor:
            raise ValueError(
                f"Receptor mismatch: {data.attrs['receptor']} != {config.receptor}"
            )

        # Redefine the attributes of the data
        attrs = {
            "time_created": dt.datetime.fromisoformat(data.attrs.pop("time_created"))
        }
        data.attrs = attrs

        return cls(
            simulation_id=config.simulation_id,
            receptor=config.receptor,
            data=data.foot,
            xmin=config.xmn,
            xmax=config.xmx,
            ymin=config.ymn,
            ymax=config.ymx,
            xres=config.xres,
            yres=config.yres,
            projection=config.projection,
            smooth_factor=config.smooth_factor,
            time_integrate=config.time_integrate,
        )

    @staticmethod
    def get_res_from_file(file: str | Path) -> str:
        """
        Extract resolution from the file name.

        Parameters
        ----------
        file : str | Path
            Path to the footprint netCDF file.

        Returns
        -------
        str
            resolution.
        """
        file = Path(file).resolve()
        match = re.search(r"_(\d+\.?\d*x\d+\.?\d*)_foot\.nc$", file.name)
        if match:
            return match.group(1)
        raise ValueError(
            f"Unable to extract resolution from file name: {file.name}"
        )

    @classmethod
    def calculate(
        cls,
        particles: pd.DataFrame,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        xres: float,
        yres: float,
        projection: str = "+proj=longlat",
        smooth_factor: float = 1.0,
        time_integrate: bool = False,
        file: str | Path | None = None,
    ) -> Self:
        raise NotImplementedError

    @staticmethod
    def read_netcdf(
        file: str | Path, parse_receptor: bool = True, **kwargs
    ) -> xr.Dataset:
        """
        Read netCDF file and return xarray Dataset.

        Parameters
        ----------
        file : str | Path
            Path to netCDF file.
        parse_receptor : bool, optional
            Whether to parse receptor coordinates. Default is True.
        **kwargs : dict
            Additional keyword arguments for xr.open_dataset.

        Returns
        -------
        xr.Dataset
            Footprint data as an xarray Dataset.
        """
        ds = xr.open_dataset(Path(file).resolve(), **kwargs)

        if parse_receptor:
            receptor = Receptor.build(
                time=ds.attrs.pop("r_time"),
                longitude=ds.attrs.pop("r_long"),
                latitude=ds.attrs.pop("r_lati"),
                height=ds.attrs.pop("r_zagl"),
            )
            ds.attrs["receptor"] = receptor
            # ds = ds.assign_coords({'receptor': receptor.id})

        return ds

    @staticmethod
    def _integrate_over_time(
        data: xr.DataArray,
        start: dt.datetime | None = None,
        end: dt.datetime | None = None,
    ) -> xr.DataArray:
        """
        Integrate footprint dataarray over time.
        """
        return data.sel(time=slice(start, end)).sum("time")

    def integrate_over_time(
        self, start: dt.datetime | None = None, end: dt.datetime | None = None
    ) -> xr.DataArray:
        """
        Integrate footprint over time.

        Parameters
        ----------
        start : datetime, optional
            Start time of integration. The default is None.
        end : datetime, optional
            End time of integration. The default is None.

        Returns
        -------
        xr.DataArray
            Time-integrated footprint
        """
        return self._integrate_over_time(self.data, start=start, end=end)


class FootprintCollection(UserDict):
    def __init__(self, simulation):
        super().__init__()
        self.simulation = simulation

        self.resolutions = simulation.config.resolutions

    def __getitem__(self, key):
        # If the footprint for the given resolution is not loaded, load it
        if key not in self.data:
            path = self.simulation.paths["footprints"].get(key)
            if path and path.exists():
                xres, yres = map(float, key.split("x"))
                self.data[key] = Footprint(
                    simulation_id=self.simulation.id,
                    receptor=self.simulation.receptor,
                    data=Footprint.read_netcdf(path, parse_receptor=False).foot,
                    xmin=self.simulation.config.xmn,
                    xmax=self.simulation.config.xmx,
                    ymin=self.simulation.config.ymn,
                    ymax=self.simulation.config.ymx,
                    xres=xres,
                    yres=yres,
                    projection=self.simulation.config.projection,
                    smooth_factor=self.simulation.config.smooth_factor,
                    time_integrate=self.simulation.config.time_integrate,
                )
            else:
                raise KeyError(f"Footprint for resolution '{key}' not found.")
        return self.data[key]

    def get(self, resolution: str) -> Footprint | None:
        """
        Get footprint for a specific resolution.

        Parameters
        ----------
        resolution : str
            Resolution string (e.g., '1x1').

        Returns
        -------
        Footprint | None
            Footprint object if available, else None.
        """
        try:
            return self[resolution]
        except KeyError:
            return None

    def __repr__(self):
        return f"FootprintCollection(simulation_id={self.simulation.id}, resolutions={self.resolutions})"
