"""
The columns of a table of soundings, which every reader returns.

A reader turns a product file into a :class:`pandas.DataFrame` with one row
per sounding. :data:`SOUNDING_SCHEMA` lists its columns and what each
means, and :func:`check_soundings` checks a table against it, so a new
reader can be tested the same way as the others.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

#: When a reader returns a column: ``always``; ``kernel``, for every product
#: with an averaging kernel (all but GGG ``.oof`` files); or ``optional``,
#: when the product has it.
When = Literal["always", "kernel", "optional"]

#: Every column of a table of soundings, as ``{name: (when, meaning)}``.
#: Vertical arrays run from the surface upward, pressures are in hPa,
#: altitudes in meters above sea level, and azimuths in degrees clockwise
#: from north toward the instrument or the sun.
SOUNDING_SCHEMA: dict[str, tuple[When, str]] = {
    "sounding_id": (
        "always",
        "A string that identifies the sounding within the product.",
    ),
    "time": ("always", "UTC time of the sounding, timezone-naive."),
    "longitude": ("always", "Pixel center or station longitude, degrees."),
    "latitude": ("always", "Pixel center or station latitude, degrees."),
    "surface_altitude": ("always", "Surface (or station) altitude, m above sea level."),
    "surface_pressure": ("always", "Surface pressure the retrieval used, hPa."),
    "value": ("always", "The column-average dry-air mole fraction."),
    "uncertainty": ("always", "The one-sigma error of value."),
    "species": ("always", "The gas, such as xch4 or xco2."),
    "units": ("always", "The units of value, ppb or ppm."),
    "good": ("always", "The product's recommended quality screen, as a boolean."),
    "ak_pressure": (
        "kernel",
        "Pressures of the averaging kernel, hPa, one array per row; the layer "
        "midpoints for a layer product (TROPOMI).",
    ),
    "ak": ("kernel", "The column averaging kernel at ak_pressure, one array per row."),
    "pressure_levels": (
        "kernel",
        "The retrieval's own pressure grid, hPa, one array per row.",
    ),
    "zenith": (
        "optional",
        "Zenith angle of the line of sight toward the instrument or the sun, degrees.",
    ),
    "azimuth": ("optional", "Azimuth of the line of sight, degrees."),
    "solar_zenith": ("optional", "Solar zenith angle, degrees."),
    "solar_azimuth": ("optional", "Solar azimuth angle, degrees."),
    "altitude_levels": (
        "optional",
        "Heights of pressure_levels, m above sea level, one array per row.",
    ),
    "apriori": (
        "optional",
        "The prior profile as a mole fraction in units, one array per row: on "
        "the kernel's levels (TROPOMI, OCO-2) or on pressure_levels (GGG).",
    ),
    "apriori_column": ("optional", "The prior column value (OCO-2)."),
    "longitude_bounds": ("optional", "Pixel corner longitudes, one array per row."),
    "latitude_bounds": ("optional", "Pixel corner latitudes, one array per row."),
}


def check_soundings(soundings: pd.DataFrame, *, kernel: bool = True) -> None:
    """
    Raise unless a table of soundings has the columns every reader returns.

    Parameters
    ----------
    soundings : pandas.DataFrame
        A reader's table.
    kernel : bool, default True
        Also require the averaging-kernel columns, which every product with
        a kernel has (all but GGG ``.oof`` files).

    Raises
    ------
    ValueError
        Naming the missing columns.
    """
    wanted = [
        name
        for name, (when, _) in SOUNDING_SCHEMA.items()
        if when == "always" or (kernel and when == "kernel")
    ]
    missing = [name for name in wanted if name not in soundings.columns]
    if missing:
        raise ValueError(
            f"The soundings lack the columns {missing} (see SOUNDING_SCHEMA)."
        )


__all__ = ["SOUNDING_SCHEMA", "check_soundings"]
