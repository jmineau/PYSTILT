"""
Footprint maps for a web page: the stage that draws each footprint as an image.

:class:`MapStage` reads each simulation's footprint and writes one JSON file
per release beside it in the output directory, holding the footprint summed
over time as a PNG image and what a web map needs to place it:

.. code-block:: yaml

   stages:
     footprint:
       class: stilt.footprint.FootprintStage
       grid: {xmin: -113.0, xmax: -110.5, ymin: 40.0, ymax: 42.0, xres: 0.01, yres: 0.01}
     map:
       class: stilt.footprint.MapStage

The image's rows are spaced as on a web map (web Mercator), so a map that
stretches it between its corners, such as Leaflet's ``imageOverlay``, draws
each cell in its place. Every image is drawn on one fixed log color scale
(``vmin`` to ``vmax``), so two hours or two days compare by eye; cells below
the scale are transparent.
"""

from __future__ import annotations

import base64
import datetime as dt
import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, model_validator
from runtraj.stages import Stage

from .io import read_footprint

if TYPE_CHECKING:
    from runtraj.releases import Release
    from runtraj.simulation import Simulation
    from upath import UPath


class MapConfig(BaseModel):
    """
    Map settings: the color scale every footprint image is drawn on.

    The scale is fixed, not fitted to each footprint, so that images from
    different hours and days compare by eye. The defaults suit a tower's
    footprint in ppm per (μmol m⁻² s⁻¹) summed over a day of particle travel.
    """

    model_config = ConfigDict(extra="forbid")

    vmin: float = Field(
        1e-5,
        gt=0,
        description=(
            "The bottom of the log color scale, in the footprint's units. "
            "Cells below it are transparent."
        ),
    )
    vmax: float = Field(
        1e-1,
        gt=0,
        description="The top of the log color scale. Cells above it take the top color.",
    )
    colormap: str = Field("viridis", description="The name of a matplotlib colormap.")

    @model_validator(mode="after")
    def _check(self) -> MapConfig:
        """Check that the scale goes up."""
        if self.vmin >= self.vmax:
            raise ValueError(
                f"vmin ({self.vmin:g}) must be below vmax ({self.vmax:g})."
            )
        return self


class MapStage(Stage[MapConfig]):
    """
    The stage that draws each simulation's footprint as an image for a web map.

    Name it as the ``class`` of a block under ``stages:`` in
    ``config.yaml``, after the footprint's block. Its file for a release is
    ``<release id>.json`` in the stage's settings folder, written after the
    footprint is (:func:`map_entry` lists what it holds). An empty footprint
    gets a file too, with ``image`` and ``bounds`` null, so the release is
    finished. A web page that knows its sites and hours names each file from
    the release id, and treats a missing one as not drawn yet.

    Needs matplotlib (the ``visualization`` extra).

    Attributes
    ----------
    name : str
        ``map``, what its results are called.
    config_class : type
        :class:`MapConfig`, which checks the block's settings.
    suffix : str
        ``.json``.
    reads : tuple of str
        ``("footprint",)``: a map is drawn from the footprint.

    Its settings are the color scale, every field of :class:`MapConfig`,
    and it finds nothing to check before a run: both are
    :class:`runtraj.stages.Stage`'s defaults.
    """

    name = "map"
    config_class = MapConfig
    suffix = ".json"
    reads = ("footprint",)

    def resolve(
        self,
        config: MapConfig,
        model: str,
        model_settings: Mapping[str, Any],
        directory: Path | None,
    ) -> MapConfig:
        """
        Return the map settings as written, after checking that matplotlib knows the colormap.

        Raises
        ------
        ValueError
            If the colormap is not one of matplotlib's.
        """
        import matplotlib

        if config.colormap not in matplotlib.colormaps:
            raise ValueError(
                f"the map's colormap {config.colormap!r} is not one of "
                "matplotlib's, such as 'viridis' or 'magma'."
            )
        return config

    def run(self, config: MapConfig, sim: Simulation, path: Path | UPath) -> None:
        """
        Draw a simulation's footprint and write its JSON file at *path*.

        Raises
        ------
        ValueError
            If the footprint is on a projected grid (:func:`bounds`).
        """
        footprint = sim.path("footprint")
        if footprint is None:  # the worker runs a stage after the ones it reads
            raise FileNotFoundError(f"{sim} has no footprint to draw.")
        foot = read_footprint(footprint)
        entry = map_entry(sim.release, foot, config)
        path.write_text(json.dumps(entry))


def mercator_y(lat: Any) -> Any:
    """
    Return web Mercator's y for latitudes in degrees, in units of the Earth's radius.

    Examples
    --------
    >>> round(float(mercator_y(45.0)), 4)
    0.8814
    """
    return np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def _check_lonlat(foot: xr.DataArray) -> None:
    """Raise a ValueError if *foot* is not on a longitude/latitude grid."""
    if not {"lat", "lon"} <= set(foot.dims):
        raise ValueError(
            "the map stage draws footprints on a longitude/latitude grid, "
            f"and this footprint is on a projected one (dimensions {tuple(foot.dims)})."
        )


def bounds(foot: xr.DataArray) -> list[list[float]]:
    """
    Return the footprint grid's outer corners, ``[[south, west], [north, east]]``, the order Leaflet takes.

    The grid's ``lat`` and ``lon`` are cell centers, so the corners are half
    a cell beyond the outer ones. A grid across 180° keeps the longitudes it
    is written with, so its east edge is past 180 (``190`` for a grid from
    170 to -170), which a web map draws in the right place.

    Raises
    ------
    ValueError
        If the footprint is on a projected grid (``y`` and ``x`` rather than
        ``lat`` and ``lon``): the image's rows follow web Mercator in
        latitude, so the map stage does not draw a projected footprint.
    """
    _check_lonlat(foot)
    lat, lon = foot["lat"].values, foot["lon"].values
    dlat = abs(float(lat[1] - lat[0])) if lat.size > 1 else 0.0
    dlon = abs(float(lon[1] - lon[0])) if lon.size > 1 else 0.0
    return [
        [float(lat.min()) - dlat / 2, float(lon.min()) - dlon / 2],
        [float(lat.max()) + dlat / 2, float(lon.max()) + dlon / 2],
    ]


def _bands(lats: np.ndarray, south: float, north: float, rows: int) -> np.ndarray:
    """
    Return the grid row (index into *lats*, north first) each of *rows* image rows shows.

    The image rows are equal bands of web Mercator's y between *north* and
    *south*; each takes the grid row nearest its middle.
    """
    edges = np.linspace(mercator_y(north), mercator_y(south), rows + 1)
    middles = np.degrees(
        2 * np.arctan(np.exp((edges[:-1] + edges[1:]) / 2)) - np.pi / 2
    )
    return np.abs(lats[None, :] - middles[:, None]).argmin(axis=1)


def image(foot: xr.DataArray, config: MapConfig | None = None) -> np.ndarray:
    """
    Return the footprint as an RGBA image, north at the top, rows in web Mercator.

    The footprint is summed over ``time`` when it has that dimension. Each
    image row is a band of equal height on a web map, and takes the value
    of the grid row whose latitude is nearest its middle. The image has one
    row per grid row, unless that would leave a grid row out (the bands are
    taller than a cell where a wide grid is far from the equator); then it
    has as many rows as it takes for every grid row to show. Cells are
    drawn more opaque the higher they are on the scale, and cells below
    ``vmin`` are transparent.

    Parameters
    ----------
    foot : xarray.DataArray
        A footprint.
    config : MapConfig, optional
        The color scale; the defaults when not given.

    Raises
    ------
    ValueError
        If the footprint is on a projected grid (:func:`bounds`).
    """
    import matplotlib

    _check_lonlat(foot)
    config = config or MapConfig()
    total = foot.sum("time") if "time" in foot.dims else foot
    total = total.transpose("lat", "lon").sortby("lat", ascending=False)
    values = np.asarray(total.values, dtype=float)
    (south, _), (north, _) = bounds(total)
    lats = total["lat"].values
    rows = values.shape[0]
    nearest = _bands(lats, south, north, rows)
    if len(set(nearest)) < rows:
        # Every cell is at least as tall (in Mercator y) as the smallest one,
        # so bands that high cannot step over a cell.
        cells = np.diff(mercator_y(np.linspace(north, south, rows + 1)))
        height = (mercator_y(north) - mercator_y(south)) / abs(cells).min()
        nearest = _bands(lats, south, north, int(np.ceil(height)))
    values = values[nearest]

    low, high = np.log10(config.vmin), np.log10(config.vmax)
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = (np.log10(values) - low) / (high - low)
    shown = np.isfinite(scaled) & (scaled >= 0)
    rgba = matplotlib.colormaps[config.colormap](np.clip(np.nan_to_num(scaled), 0, 1))
    rgba[..., 3] = np.where(shown, 0.4 + 0.5 * np.clip(scaled, 0, 1), 0.0)
    return (rgba * 255).round().astype(np.uint8)


def png(foot: xr.DataArray, config: MapConfig | None = None) -> bytes:
    """Return :func:`image` as PNG bytes."""
    import matplotlib.image

    buffer = io.BytesIO()
    matplotlib.image.imsave(buffer, image(foot, config), format="png")
    return buffer.getvalue()


def utc(time: dt.datetime) -> str:
    """
    Return *time* as text with its zone, UTC, ending in ``Z``.

    A time with no zone is UTC, as PYSTILT's are. JavaScript reads text
    without a zone as the viewer's local time, so the zone is written out.

    Examples
    --------
    >>> utc(dt.datetime(2026, 10, 29, 10))
    '2026-10-29T10:00:00Z'
    """
    if time.tzinfo is not None:
        time = time.astimezone(dt.UTC)
    return f"{time:%Y-%m-%dT%H:%M:%S}Z"


def map_entry(
    release: Release, foot: xr.DataArray | None, config: MapConfig | None = None
) -> dict[str, Any]:
    """
    Return what a map file holds for *release*'s footprint *foot*.

    Keys: ``release`` (the release id), ``time`` (the release time, UTC),
    ``location`` (the release's kind and points, as ``releases.csv`` gives
    them), ``bounds`` (:func:`bounds`), ``image`` (the PNG as a data URL,
    which a web map hands to ``imageOverlay`` as it is), ``vmin``, ``vmax``,
    ``colormap`` and ``units``. ``image`` and ``bounds`` are null for an
    empty footprint (``foot`` is ``None``).

    Raises
    ------
    ValueError
        If the footprint is on a projected grid (:func:`bounds`).
    """
    config = config or MapConfig()
    location = {k: v for k, v in release.to_dict().items() if k != "time"}
    entry: dict[str, Any] = {
        "release": str(release.id),
        "time": utc(release.time),
        "location": location,
        "bounds": None,
        "image": None,
        **config.model_dump(mode="json"),
        "units": "",
    }
    if foot is not None:
        encoded = base64.b64encode(png(foot, config)).decode("ascii")
        entry.update(
            bounds=bounds(foot),
            image=f"data:image/png;base64,{encoded}",
            units=foot.attrs.get("units", ""),
        )
    return entry


#: The import path of the map stage, the ``class`` of its ``stages:`` block.
MAP_STAGE = "stilt.footprint.MapStage"

__all__ = [
    "MAP_STAGE",
    "MapConfig",
    "MapStage",
    "bounds",
    "image",
    "map_entry",
    "mercator_y",
    "png",
    "utc",
]
