"""Tests for the forward-run plume outline and the background beside it."""

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from stilt.observations import (
    Plume,
    PlumeBackground,
    plume_background,
    plume_polygon,
)
from stilt.observations.plumes import density_polygon, kernel_density

SITE = (-111.9, 40.75)


def _plume_particles(n=4000, seed=0):
    """A plume blown east-north-east from the site: a tilted elongated cloud."""
    rng = np.random.default_rng(seed)
    along = rng.uniform(0.0, 1.0, n)  # degrees downwind
    across = rng.normal(0.0, 0.03 + 0.05 * along, n)
    lon = SITE[0] + along
    lat = SITE[1] + 0.3 * along + across
    return lon, lat


def _swath(step=0.02):
    """A rectangular field of soundings around the site with a plume signal."""
    lons = np.arange(SITE[0] - 1.0, SITE[0] + 2.0, step)
    lats = np.arange(SITE[1] - 1.0, SITE[1] + 1.5, step)
    lon, lat = (a.ravel() for a in np.meshgrid(lons, lats))
    return lon, lat


def test_kernel_density_peaks_at_the_points_and_is_normalised():
    d = kernel_density([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], bandwidth=(0.1, 0.1), n=41)
    assert d.dims == ("lat", "lon")
    assert d.max() == 1.0
    peak = d.isel(d.argmax(...))
    assert float(peak["lon"]) == pytest.approx(0.0, abs=1e-9)
    assert float(peak["lat"]) == pytest.approx(0.0, abs=1e-9)
    # Grid spans the points plus one bandwidth each side.
    assert float(d["lon"].min()) == pytest.approx(-0.1)
    assert float(d["lon"].max()) == pytest.approx(0.1)
    # A quarter of the bandwidth is the kernel's standard deviation.
    at_sd = float(d.sel(lon=0.025, lat=0.0, method="nearest"))
    assert at_sd == pytest.approx(np.exp(-0.5), rel=0.05)


def test_kernel_density_rejects_bad_inputs():
    with pytest.raises(ValueError, match="same length"):
        kernel_density([0.0, 1.0], [0.0])
    with pytest.raises(ValueError, match="finite"):
        kernel_density([np.nan], [np.nan])
    with pytest.raises(ValueError, match="bandwidth"):
        kernel_density([0.0], [0.0], bandwidth=(0.0, 0.1))
    with pytest.raises(ValueError, match="n must"):
        kernel_density([0.0], [0.0], n=1)


def test_density_polygon_keeps_the_largest_piece():
    lon = np.linspace(-1, 1, 21)
    lat = np.linspace(-1, 1, 21)
    z = np.zeros((21, 21))
    z[8:13, 2:6] = 1.0  # small blob, 5 x 4 cells
    z[3:18, 10:19] = 1.0  # big blob, 15 x 9 cells
    import xarray as xr

    density = xr.DataArray(z, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    poly = density_polygon(density, 0.5)
    assert isinstance(poly, Polygon)
    assert poly.area == pytest.approx(15 * 9 * 0.1 * 0.1)
    assert poly.contains(
        Polygon([(0.05, -0.6), (0.75, -0.6), (0.75, 0.6), (0.05, 0.6)])
    )
    assert not poly.intersects(
        Polygon([(-0.8, -0.2), (-0.5, -0.2), (-0.5, 0.2), (-0.8, 0.2)])
    )
    with pytest.raises(ValueError, match="threshold"):
        density_polygon(density, 0.0)
    with pytest.raises(ValueError, match="no density"):
        density_polygon(density * 0.1, 0.5)


def test_plume_polygon_follows_the_particles():
    lon, lat = _plume_particles()
    plume = plume_polygon(lon, lat)
    assert isinstance(plume, Plume)
    assert plume.threshold == 0.1
    assert plume.density.max() == 1.0
    # Downwind axis is inside; well off-axis points are outside.
    inside = plume.contains([SITE[0] + 0.5], [SITE[1] + 0.15])
    outside = plume.contains([SITE[0] + 0.5], [SITE[1] + 0.9])
    upwind = plume.contains([SITE[0] - 0.5], [SITE[1]])
    assert inside[0] and not outside[0] and not upwind[0]
    # Most particles are inside the outline at the default threshold.
    assert plume.contains(lon, lat).mean() > 0.85
    # A tighter threshold gives a smaller plume.
    tight = plume_polygon(lon, lat, threshold=0.5)
    assert tight.polygon.area < plume.polygon.area
    assert plume.polygon.contains(tight.polygon.representative_point())


def test_plume_background_takes_the_median_beside_the_plume():
    plon, plat = _plume_particles()
    plume = plume_polygon(plon, plat)
    lon, lat = _swath()
    in_plume = plume.contains(lon, lat)
    rng = np.random.default_rng(1)
    value = 1900.0 + rng.normal(0.0, 2.0, lon.size)
    value[in_plume] += 30.0
    unc = np.full(lon.size, 5.0)

    bg = plume_background(lon, lat, value, plume, uncertainties=unc, trim=None)
    assert isinstance(bg, PlumeBackground)
    assert bg.in_plume.sum() == in_plume.sum()
    assert not (bg.used & bg.in_plume).any()
    assert bg.n == bg.used.sum() > 0
    assert bg.value == pytest.approx(np.median(value[bg.used]))
    assert bg.value == pytest.approx(1900.0, abs=1.0)
    spread = np.std(value[bg.used], ddof=1)
    assert bg.uncertainty == pytest.approx(np.hypot(spread, 5.0))
    assert list(bg.sides.index) == ["north", "south", "east", "west"]
    assert bg.sides["n"].sum() >= bg.n  # a corner sounding can count twice
    assert (bg.sides["n"] > 0).all()
    assert bg.sides["median"].between(1898, 1902).all()

    # One side alone.
    north = plume_background(lon, lat, value, plume, side="north", trim=None)
    assert north.n == bg.sides.loc["north", "n"]
    assert north.value == bg.sides.loc["north", "median"]
    assert lat[north.used].min() >= lat[north.in_plume].max()  # above the plume box
    assert np.isnan(north.sides["retrieval_std"]).all()  # no uncertainties given
    assert north.uncertainty == pytest.approx(np.std(value[north.used], ddof=1))

    # A bare polygon is accepted too.
    same = plume_background(lon, lat, value, plume.polygon, trim=None)
    assert same.value == bg.value


def test_plume_background_trim_drops_the_high_tail():
    plon, plat = _plume_particles()
    plume = plume_polygon(plon, plat)
    lon, lat = _swath()
    in_plume = plume.contains(lon, lat)
    value = np.full(lon.size, 1900.0)
    value[in_plume] = 1930.0
    # Enhanced air just past the outline that the plume missed.
    miss = ~in_plume & (lon > SITE[0] + 1.0) & (lon < SITE[0] + 1.2)
    value[miss] = 1950.0

    kept = plume_background(lon, lat, value, plume, trim=None)
    trimmed = plume_background(lon, lat, value, plume, trim=0.9)
    assert (value[kept.used] == 1950.0).any()
    assert not (value[trimmed.used] == 1950.0).any()
    assert trimmed.value == 1900.0


def test_plume_background_width_and_pad_grow_the_selection():
    plon, plat = _plume_particles()
    plume = plume_polygon(plon, plat)
    lon, lat = _swath()
    value = np.full(lon.size, 1900.0)
    narrow = plume_background(lon, lat, value, plume, width=0.1, trim=None)
    wide = plume_background(lon, lat, value, plume, width=0.5, trim=None)
    assert wide.n > narrow.n
    padded = plume_background(lon, lat, value, plume, side="north", pad=0.5, trim=None)
    tight = plume_background(lon, lat, value, plume, side="north", pad=0.0, trim=None)
    assert lat[padded.used].min() > lat[tight.used].min()


def test_plume_background_errors():
    plon, plat = _plume_particles()
    plume = plume_polygon(plon, plat)
    lon, lat = _swath()
    value = np.full(lon.size, 1900.0)
    with pytest.raises(ValueError, match="same length"):
        plume_background(lon, lat, value[:-1], plume)
    with pytest.raises(ValueError, match="uncertainties"):
        plume_background(lon, lat, value, plume, uncertainties=value[:-1])
    with pytest.raises(ValueError, match="side"):
        plume_background(lon, lat, value, plume, side="up")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="width"):
        plume_background(lon, lat, value, plume, width=0.0)
    with pytest.raises(ValueError, match="trim"):
        plume_background(lon, lat, value, plume, trim=1.5)
    far = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    with pytest.raises(ValueError, match="inside the plume"):
        plume_background(lon, lat, value, far)
    # NaN values never count as in-plume or as background.
    nan = value.copy()
    nan[:] = np.nan
    with pytest.raises(ValueError, match="inside the plume"):
        plume_background(lon, lat, nan, plume)


def test_plume_recipe_from_particle_tables():
    """The documented recipe: pool overpass-time rows across forward runs."""
    overpass = (pd.Timestamp("2023-10-19 19:40"), pd.Timestamp("2023-10-19 19:44"))
    rows = []
    for k, release in enumerate(
        pd.date_range("2023-10-19 10:00", "2023-10-19 20:00", freq="30min")
    ):
        lon, lat = _plume_particles(n=50, seed=k)
        minutes = np.arange(0, 12 * 60, 2)
        table = pd.DataFrame(
            {
                "indx": np.repeat(np.arange(1, 51), minutes.size),
                "time": np.tile(minutes, 50),
                "long": np.repeat(lon, minutes.size),
                "lati": np.repeat(lat, minutes.size),
            }
        )
        table["datetime"] = release + pd.to_timedelta(table["time"], unit="min")
        rows.append(table[table["datetime"].between(overpass[0], overpass[1])])
    particles = pd.concat(rows, ignore_index=True)
    assert particles["datetime"].between(*overpass).all()
    plume = plume_polygon(particles["long"], particles["lati"])
    assert plume.contains([SITE[0] + 0.5], [SITE[1] + 0.15])[0]
