"""
The oxidation transform: weigh each particle's ``foot`` by the tracer that survives a reaction with an oxidant such as OH.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, field_validator

from stilt.sampling import sample_field, vertical_dim
from stilt.transforms.lifetime import survival

if TYPE_CHECKING:
    from stilt.releases import Release

# The specific gas constant of dry air, in J kg⁻¹ K⁻¹.
_R_DRY_AIR = 287.05


def _temperature(particles: pd.DataFrame) -> np.ndarray:
    """Return the air temperature at each row, in K: ``temz``, else from ``pres`` and ``dens``."""
    if "temz" in particles.columns:
        return particles["temz"].to_numpy(dtype=float)
    if {"pres", "dens"} <= set(particles.columns):
        pres = particles["pres"].to_numpy(dtype=float) * 100.0
        return pres / (particles["dens"].to_numpy(dtype=float) * _R_DRY_AIR)
    raise ValueError(
        "Particle DataFrame has neither 'temz' nor 'pres' and 'dens', one of "
        "which oxidation needs for the temperature."
    )


@functools.lru_cache(maxsize=4)
def _read_field(path: str, variable: str | None, mtime: float) -> xr.DataArray:
    """Read an oxidant field into memory; *mtime* makes a changed file read again."""
    with xr.open_dataset(path) as dataset:
        if variable is None:
            names = list(dataset.data_vars)
            if len(names) != 1:
                raise ValueError(
                    f"oxidant file {path!r} has {len(names)} variables "
                    f"({names}); name one as variable."
                )
            variable = str(names[0])
        if variable not in dataset.data_vars:
            raise ValueError(
                f"oxidant file {path!r} has no variable {variable!r}; it has "
                f"{list(dataset.data_vars)}."
            )
        return dataset[variable].load()


class Oxidation(BaseModel):
    r"""
    Transform that weighs each particle's ``foot`` by the tracer that survives a reaction with an oxidant.

    Use it for a species lost mainly to one reaction whose rate you know,
    such as CO, ethane, or methane with OH. The rate at each row is
    ``k(T) [X]``, with the rate constant ``k = a exp(-e_over_r / T)`` and
    the oxidant's number density ``[X]``, and each row's ``foot`` is
    multiplied by :func:`survival` of it. Give ``[X]`` as a constant or as a
    gridded field in a netCDF file, sampled at each row's longitude,
    latitude, time and ``coordinate``.

    The loss must not depend on the tracer itself. Methane's lifetime
    against OH is about nine years, so over a few days of trajectory it
    changes ``foot`` by less than 0.1 %; the transform matters for shorter
    lived species and long trajectories.

    Notes
    -----
    The temperature is the particles' ``temz`` column when they have it.
    Otherwise it is :math:`p / (\rho R_d)` from ``pres`` and ``dens``, which
    are in the default ``varsiwant``, so particles already run need no
    rerun. That is the virtual temperature, above the temperature by about
    :math:`0.6 \, q T` in moist air. On a HYSPLIT run from Salt Lake City
    in January (HRRR, 72,000 rows) it was 0.5 to 0.9 K above HYSPLIT's
    ``temz`` (5th to 95th percentile), about 2 % in a rate with
    ``e_over_r`` of 1775 K. Add ``temz`` to ``varsiwant`` when that matters.

    A field with a ``time`` dimension is matched to each row's nearest
    time, so a climatology must carry the dates of the run.

    Examples
    --------
    Methane and OH at a constant number density, with JPL's rate constant:

    >>> import pandas as pd
    >>> from stilt.transforms import Oxidation
    >>> p = pd.DataFrame(
    ...     {
    ...         "particle": [1, 1],
    ...         "age": [-1440.0, -2880.0],
    ...         "lon": [-112.0, -113.0],
    ...         "lat": [40.0, 40.0],
    ...         "temz": [280.0, 280.0],
    ...         "foot": [1.0, 1.0],
    ...     }
    ... )
    >>> ch4 = Oxidation(a=2.45e-12, e_over_r=1775, oxidant=1e6)
    >>> ch4.apply(p)["foot"].round(5).tolist()
    [0.99963, 0.99925]
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["oxidation"] = "oxidation"
    a: float = Field(
        gt=0,
        description=(
            "Pre-exponential factor of the rate constant, in cm³ molecule⁻¹ "
            "s⁻¹ (the A of the JPL and IUPAC tables)."
        ),
    )
    e_over_r: float = Field(
        0.0,
        description=(
            "Activation temperature of the rate constant, in K (the E/R of "
            "the JPL and IUPAC tables). 0 for a rate that does not depend on "
            "temperature."
        ),
    )
    oxidant: float | str = Field(
        description=(
            "The oxidant's number density, in molecules cm⁻³: a constant, or "
            "a netCDF file of a field on a regular ``lat``/``lon`` grid with "
            "an optional vertical dimension and ``time``. A relative path is "
            "relative to the project root."
        ),
    )
    variable: str | None = Field(
        None,
        description=(
            "Variable of the oxidant file to read. Unset reads the file's "
            "only variable."
        ),
    )
    coordinate: str = Field(
        "pres",
        description=(
            "Particle column matched to the field's vertical dimension, in its "
            "units: ``pres`` (hPa) by default, or ``zagl`` (m above ground)."
        ),
    )

    @field_validator("oxidant", mode="before")
    @classmethod
    def _number_or_file(cls, value: Any) -> Any:
        """Read a number written as text, such as YAML's ``1e6``, as a number."""
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                if not value.strip():
                    raise ValueError(
                        "oxidant must be a number or a file name."
                    ) from None
                return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value < 0:
                raise ValueError("oxidant must be zero or positive.")
            return float(value)
        return value

    def _path(self, directory: str | Path | None) -> str:
        """Return the oxidant file's path; a relative one starts from *directory*."""
        assert isinstance(self.oxidant, str)
        path = Path(self.oxidant)
        if directory is not None and not path.is_absolute():
            path = Path(directory) / path
        return str(path)

    def field(self, directory: str | Path | None = None) -> xr.DataArray:
        """
        Return the oxidant field, read from its file.

        The file is read into memory once per process and again only when it
        changes. A relative ``oxidant`` path starts from *directory*.

        Raises
        ------
        ValueError
            If ``oxidant`` is a constant, or the file has no such variable.
        FileNotFoundError
            If the file does not exist.
        """
        if not isinstance(self.oxidant, str):
            raise ValueError("oxidation's oxidant is a constant, not a file.")
        path = self._path(directory)
        return _read_field(path, self.variable, os.stat(path).st_mtime)

    def rate(
        self, particles: pd.DataFrame, directory: str | Path | None = None
    ) -> np.ndarray:
        """
        Return the loss rate at each row, in s⁻¹.

        A row outside the oxidant field, or in a missing cell, has no loss.
        A relative ``oxidant`` path starts from *directory*.
        """
        temperature = _temperature(particles)
        k = self.a * np.exp(-self.e_over_r / temperature)
        if not isinstance(self.oxidant, str):
            return k * self.oxidant
        field = self.field(directory)
        z = None
        if vertical_dim(field) is not None:
            if self.coordinate not in particles.columns:
                raise ValueError(
                    f"Particle DataFrame has no {self.coordinate!r} column, "
                    "which oxidation matches to the field's vertical dimension."
                )
            z = particles[self.coordinate].to_numpy(dtype=float)
        times = None
        if "time" in field.dims:
            if "time" not in particles.columns:
                raise ValueError(
                    "The oxidant field varies in time but the particles have "
                    "no 'time' column."
                )
            times = particles["time"].to_numpy()
        number_density = sample_field(
            field,
            particles["lon"].to_numpy(dtype=float),
            particles["lat"].to_numpy(dtype=float),
            z=z,
            times=times,
            fill_value=0.0,
        )
        return k * number_density

    def check(
        self, releases: Sequence[str], directory: str | Path | None = None
    ) -> list[tuple[str, str | None, str]]:
        """
        Return what is wrong with the oxidant file, which ``stilt check`` reports.

        Each finding is ``(check, None, message)``: the file does not exist,
        cannot be read, or has no such variable (``oxidant_file``). A
        constant oxidant has nothing to check. A relative ``oxidant`` path
        starts from *directory*.
        """
        if not isinstance(self.oxidant, str):
            return []
        try:
            self.field(directory)
        except FileNotFoundError as error:
            return [
                (
                    "oxidant_file",
                    None,
                    f"oxidant {self.oxidant!r} does not exist: {error.filename}",
                )
            ]
        except (OSError, ValueError) as error:
            return [("oxidant_file", None, str(error))]
        return []

    def apply(
        self,
        particles: pd.DataFrame,
        release: Release | None = None,
        directory: str | Path | None = None,
    ) -> pd.DataFrame:
        """Return the particles with ``foot`` scaled by the surviving fraction."""
        fraction = survival(particles, self.rate(particles, directory))
        out = particles.copy()
        out["foot"] = out["foot"] * fraction.to_numpy()
        return out
