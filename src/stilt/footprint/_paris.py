"""
PARIS-format footprints for openghg, behind ``foot.stilt.to_paris``.

The PARIS format is the NetCDF layout that openghg reads with
``source_format="paris"``. This module turns one footprint into a dataset in
that layout: the total footprint ``srr``, the footprint of each of the last
23 hours ``srr_time_resolved``, the older rest ``srr_residual``, and the
share of particles that left the footprint grid through each edge,
``particle_locations_n``, ``_e``, ``_s`` and ``_w``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from stilt.footprint.config import FootprintConfig
from stilt.receptors import parse_receptor_id

# Units of the footprint in a PARIS file. PYSTILT's ppm per (µmol m⁻² s⁻¹)
# is the same quantity with the same numbers.
_UNITS = "(mol/mol)/(mol/m2/s)"

# Hours back that ``srr_time_resolved`` resolves, 0 to 23.
_H_BACK = np.arange(24)

# Edges of the height layers of ``particle_locations``, in meters above
# ground: 1 km layers from the ground to 20 km, as in the NAME PARIS files.
_HEIGHT_EDGES = np.arange(0.0, 20001.0, 1000.0)

# The edges of the footprint grid a particle can leave through, and the
# grid axis it is counted along there.
_EDGES = {"n": "longitude", "e": "latitude", "s": "longitude", "w": "latitude"}

_EDGE_NAMES = {"n": "north", "e": "east", "s": "south", "w": "west"}


def to_paris(
    foot: xr.DataArray,
    particles: pd.DataFrame | None = None,
    *,
    exits: pd.DataFrame | None = None,
    n_particles: int | None = None,
) -> xr.Dataset:
    """Build the PARIS dataset; see ``FootprintAccessor.to_paris``."""
    layer_dim, layer_start = _layer_starts(foot)
    for dim in ("lat", "lon"):
        if dim not in foot.dims:
            raise ValueError(
                "The PARIS format needs a longitude/latitude grid; this footprint "
                f"has dims {tuple(foot.dims)}."
            )
    receptor_time = _receptor_time(foot)
    _check_hourly(foot, layer_start)

    hours_back = -(layer_start - receptor_time) / pd.Timedelta(hours=1)
    if (hours_back < 1).any():
        raise ValueError(
            "A layer of this footprint ends after the receptor time; the PARIS "
            "export takes backward footprints only."
        )
    hours_back = np.rint(hours_back.to_numpy()).astype(int)

    values = foot.transpose(layer_dim, "lat", "lon").to_numpy()
    resolved = np.zeros((*values.shape[1:], len(_H_BACK)), dtype=values.dtype)
    for layer, back in enumerate(hours_back):
        if back < len(_H_BACK):
            resolved[..., back] += values[layer]
    residual = values[hours_back >= len(_H_BACK)].sum(axis=0)
    total = values.sum(axis=0)

    lat = foot["lat"].to_numpy()
    lon = foot["lon"].to_numpy()
    spatial = ("time", "latitude", "longitude")
    ds = xr.Dataset(
        {
            "srr": (
                spatial,
                total[None],
                {"long_name": "source_receptor_relationship", "units": _UNITS},
            ),
            "srr_time_resolved": (
                (*spatial, "resolution"),
                resolved[None],
                {
                    "long_name": "time_resolved_source_receptor_relationship",
                    "units": _UNITS,
                },
            ),
            "srr_residual": (
                spatial,
                residual[None],
                {
                    "long_name": "residual_source_receptor_relationship",
                    "definition": "srr - sum(srr_time_resolved)",
                    "units": _UNITS,
                },
            ),
        },
        coords={
            "time": (
                "time",
                [np.datetime64(receptor_time, "ns")],
                {
                    "standard_name": "time",
                    "long_name": "time",
                    "comment": "release time of the particles (the receptor time), UTC",
                },
            ),
            "latitude": (
                "latitude",
                lat,
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
            "longitude": (
                "longitude",
                lon,
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
            "resolution": (
                "resolution",
                _H_BACK.astype(np.float32),
                {"units": "hours", "long_name": "hours back from release time"},
            ),
        },
    )

    exits, n_particles = _exits(foot, particles, exits, n_particles)
    if exits is not None and n_particles is not None:
        ds = ds.assign(_particle_locations(exits, n_particles, lat, lon))

    ds.attrs = {
        "Conventions": "CF-1.8",
        "model": "STILT",
        "source": "PYSTILT footprint",
        "receptor": str(foot["receptor"].item()),
        "history": "Converted from a PYSTILT footprint",
    }
    return ds


def _layer_starts(foot: xr.DataArray) -> tuple[str, pd.DatetimeIndex]:
    """
    Return the layer dimension and the start of each layer, from the ``time`` coordinate.

    The layer dimension is the one ``time`` runs along, whatever its name.
    """
    if "receptor" in foot.dims:
        raise ValueError(
            "The footprint holds several receptors. Select one receptor first: "
            "foot.sel(receptor=...)."
        )
    if "time" not in foot.coords or foot["time"].ndim != 1:
        raise ValueError(
            "The footprint needs a time coordinate with one entry per layer (the "
            "layer's start). Select one receptor first: foot.sel(receptor=...)."
        )
    dim = str(foot["time"].dims[0])
    return dim, pd.DatetimeIndex(foot["time"].to_numpy())


def _receptor_time(foot: xr.DataArray) -> pd.Timestamp:
    """Return the receptor time, from the footprint's ``receptor`` coordinate (its id)."""
    if "receptor" not in foot.coords or foot["receptor"].ndim != 0:
        raise ValueError(
            "The footprint needs its receptor id as a scalar receptor coordinate, "
            "as sim.footprint and one receptor of project.footprints() have."
        )
    time, _ = parse_receptor_id(str(foot["receptor"].item()))
    return pd.Timestamp(time)


def _check_hourly(foot: xr.DataArray, layer_start: pd.DatetimeIndex) -> None:
    """
    Refuse a footprint whose layers are not one hour long.

    The layer length is the footprint setting ``tres`` when the footprint
    carries its settings, and the spacing of the layers otherwise.
    """
    hour = pd.Timedelta(hours=1)
    settings = foot.attrs.get("stilt_footprint")
    if settings is not None:
        step = pd.Timedelta(minutes=foot.stilt.config.tres_minutes)
    elif len(layer_start) > 1:
        steps = np.unique(np.diff(np.sort(layer_start.to_numpy())))
        if len(steps) != 1:
            raise ValueError(
                "The footprint's layers are not evenly spaced in time; the PARIS "
                "export needs whole hourly layers."
            )
        step = pd.Timedelta(steps[0])
    else:
        raise ValueError(
            "The footprint has one layer and no settings, so its layer length "
            "is unknown. Pass the footprint as PYSTILT gives it (sim.footprint)."
        )
    if step != hour:
        raise ValueError(
            f"The footprint's layers are {FootprintConfig(tres=step).tres} long; openghg reads "
            "hourly layers (H_back is in hours). Make the footprint with tres: 1h."
        )


def _exits(
    foot: xr.DataArray,
    particles: pd.DataFrame | None,
    exits: pd.DataFrame | None,
    n_particles: int | None,
) -> tuple[pd.DataFrame | None, int | None]:
    """Return the grid exits and the number of particles, from what the caller gave."""
    if particles is not None:
        if n_particles is None:
            n_particles = int(particles["particle"].nunique())
        if exits is None:
            exits = particles.stilt.grid_exits(foot.stilt.grid)
    if exits is not None and n_particles is None:
        raise ValueError(
            "Give n_particles, the number of particles released, with exits."
        )
    return exits, n_particles


def _particle_locations(
    exits: pd.DataFrame, n_particles: int, lat: np.ndarray, lon: np.ndarray
) -> dict[str, Any]:
    """
    Return ``particle_locations_n``, ``_e``, ``_s`` and ``_w`` from a table of grid exits.

    A particle counts in the grid cell along the edge it left through (by
    its longitude on the north and south edges, its latitude on the east and
    west edges) and in the height layer of its ``zagl``. A particle above
    the top layer counts in the top layer.
    """
    missing = {"particle", "edge", "lon", "lat", "zagl"} - set(exits.columns)
    if missing:
        raise ValueError(f"The grid exits have no column {', '.join(sorted(missing))}.")
    if exits["particle"].duplicated().any():
        raise ValueError(
            "The grid exits list a particle twice; give each particle's first exit."
        )
    unknown = set(exits["edge"]) - set(_EDGES)
    if unknown:
        raise ValueError(
            f"Unknown grid edges {sorted(unknown)}; use 'n', 'e', 's' or 'w'."
        )
    if n_particles < 1:
        raise ValueError("n_particles must be at least 1.")

    heights = (_HEIGHT_EDGES[:-1] + _HEIGHT_EDGES[1:]) / 2
    layer = (
        np.searchsorted(
            _HEIGHT_EDGES, exits["zagl"].to_numpy(dtype=float), side="right"
        )
        - 1
    )
    layer = np.clip(layer, 0, len(heights) - 1)

    out: dict[str, Any] = {
        "height": (
            "height",
            heights,
            {"long_name": "height at layer midpoints", "units": "m", "positive": "up"},
        )
    }
    for edge, axis in _EDGES.items():
        centers = lon if axis == "longitude" else lat
        column = "lon" if axis == "longitude" else "lat"
        rows = (exits["edge"] == edge).to_numpy()
        cell = _cell_index(exits.loc[rows, column].to_numpy(dtype=float), centers, axis)
        counts = np.zeros((len(heights), len(centers)))
        np.add.at(counts, (layer[rows], cell), 1.0)
        out[f"particle_locations_{edge}"] = (
            ("time", "height", axis),
            (counts / n_particles)[None],
            {
                "units": "1",
                "long_name": (
                    "Fraction of total particles leaving domain "
                    f"({_EDGE_NAMES[edge]} side)"
                ),
            },
        )
    return out


def _cell_index(values: np.ndarray, centers: np.ndarray, axis: str) -> np.ndarray:
    """
    Return the grid cell along *axis* that each coordinate falls in.

    Longitudes count modulo 360 from the grid's west edge, so a grid across
    180 degrees works. A coordinate just outside the grid counts in the
    nearest cell.
    """
    if len(centers) < 2:
        raise ValueError(f"The footprint grid needs two or more cells along {axis}.")
    step = float(centers[1] - centers[0])
    offset = values - (centers[0] - step / 2)
    if axis == "longitude":
        offset = offset % 360.0
        offset[offset >= 180.0 + len(centers) * step / 2] -= 360.0
    index = np.floor(offset / step).astype(int)
    return np.clip(index, 0, len(centers) - 1)
