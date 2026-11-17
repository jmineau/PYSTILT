"""Tests for stilt.sampling and the enhancement of footprints and particles."""

from pathlib import Path
from unittest.mock import patch

import arlmet
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from stilt import sampling
from stilt.footprint.config import FootprintConfig
from stilt.sampling import sample_field, sample_met
from stilt.spatial import Grid, horizontal_dims
from stilt.transport.hysplit import HysplitConfig

from .fixtures.footprints import as_footprint


def _flux(values=None, lons=(-112.0, -111.0, -110.0), lats=(40.0, 41.0)):
    values = (
        np.arange(len(lats) * len(lons), dtype=float).reshape(len(lats), len(lons))
        if values is None
        else values
    )
    return xr.DataArray(
        values, dims=["lat", "lon"], coords={"lat": list(lats), "lon": list(lons)}
    )


def test_horizontal_dims_lonlat_and_projected():
    assert horizontal_dims(_flux()) == ("lat", "lon")
    projected = xr.DataArray(
        np.zeros((2, 2)), dims=["y", "x"], coords={"y": [0, 1], "x": [0, 1]}
    )
    assert horizontal_dims(projected) == ("y", "x")
    with pytest.raises(ValueError, match="Expected"):
        horizontal_dims(xr.DataArray(np.zeros((2, 2)), dims=["a", "b"]))


def test_sample_field_with_zero_fill_nearest_cell_and_outside_is_zero():
    flux = _flux()  # rows: lat 40 -> [0,1,2], lat 41 -> [3,4,5]

    sampled = sample_field(
        flux,
        x=[-112.0, -111.4, -110.6, -109.4, -112.6, -111.0],
        y=[40.0, 40.4, 40.6, 41.0, 40.0, 41.6],
        fill_value=0.0,
    )

    # nearest centres: (-112,40)=0, (-111,40)=1, (-111,41)=4, outside east, outside west, outside north
    assert sampled.tolist() == [0.0, 1.0, 4.0, 0.0, 0.0, 0.0]


def test_sample_field_with_zero_fill_edges_extend_half_a_cell():
    flux = _flux()
    inside = sample_field(flux, x=[-112.49, -109.51], y=[39.51, 41.49], fill_value=0.0)
    outside = sample_field(flux, x=[-112.51, -109.49], y=[39.49, 41.51], fill_value=0.0)
    assert inside.tolist() == [0.0, 5.0]
    assert outside.tolist() == [0.0, 0.0]


def test_sample_field_with_zero_fill_descending_latitude_axis():
    flux = _flux(values=np.array([[3.0, 4.0, 5.0], [0.0, 1.0, 2.0]]), lats=(41.0, 40.0))
    assert sample_field(
        flux, x=[-111.0, -111.0], y=[40.0, 41.0], fill_value=0.0
    ).tolist() == [1.0, 4.0]


def test_sample_field_with_zero_fill_nan_cells_count_as_zero():
    values = np.arange(6, dtype=float).reshape(2, 3)
    values[0, 1] = np.nan
    assert sample_field(
        _flux(values), x=[-111.0], y=[40.0], fill_value=0.0
    ).tolist() == [0.0]


def test_sample_field_leaves_missing_cells_missing_by_default():
    values = np.array([[np.nan, 1.0, 2.0], [3.0, 4.0, 5.0]])
    sampled = sample_field(_flux(values), x=[-112.0, -111.0, -100.0], y=[40.0] * 3)
    assert np.isnan(sampled[0]) and sampled[1] == 1.0 and np.isnan(sampled[2])


def test_sample_field_with_zero_fill_time_varying_uses_nearest_time():
    times = pd.to_datetime(["2023-01-01 00:00", "2023-01-01 06:00"])
    flux = xr.DataArray(
        np.stack([np.full((2, 3), 1.0), np.full((2, 3), 10.0)]),
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": [40.0, 41.0], "lon": [-112.0, -111.0, -110.0]},
    )

    sampled = sample_field(
        flux,
        x=[-111.0, -111.0, -111.0],
        y=[40.0, 40.0, 40.0],
        times=["2023-01-01 02:00", "2023-01-01 05:00", "2023-01-02 00:00"],
        fill_value=0.0,
    )

    assert sampled.tolist() == [1.0, 10.0, 10.0]
    with pytest.raises(ValueError, match="pass times"):
        sample_field(flux, x=[-111.0], y=[40.0], fill_value=0.0)


def test_sample_field_with_zero_fill_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        sample_field(_flux(), x=[0.0, 1.0], y=[0.0], fill_value=0.0)


def _bounded_flux(starts, ends, values=None):
    """Return a flux uniform in space whose ``time`` is a CF-bounds IntervalIndex."""
    values = np.arange(1.0, len(starts) + 1) if values is None else np.asarray(values)
    bounds = pd.IntervalIndex.from_arrays(starts, ends, closed="left")
    return xr.DataArray(
        values[:, None, None] * np.ones((len(starts), 2, 3)),
        dims=("time", "lat", "lon"),
        coords={"time": bounds, "lat": [40.0, 41.0], "lon": [-112.0, -111.0, -110.0]},
    )


def test_sample_field_step_starts_reads_cf_time_bounds_when_present():
    starts = pd.to_datetime(
        ["2023-01-01 00:00", "2023-01-01 01:00", "2023-01-01 02:00"]
    )
    ends = starts + pd.Timedelta("1h")
    flux = _bounded_flux(starts, ends)

    times = [
        "2022-12-31 23:00",  # before the first bound
        "2023-01-01 00:30",  # inside step 0
        "2023-01-01 01:30",  # inside step 1
        "2023-01-01 02:59",  # inside step 2
        "2023-01-01 03:00",  # at the last bound's end: outside, unlike the no-bounds rule
    ]
    sampled = sample_field(
        flux,
        x=[-111.0] * 5,
        y=[40.0] * 5,
        times=times,
        fill_value=0.0,
        step_starts=True,
    )
    assert sampled.tolist() == [0.0, 1.0, 2.0, 3.0, 0.0]


def test_sample_field_step_starts_reads_bounds_even_when_the_label_disagrees():
    # the bounds, not the (here middle-of-step) label, decide which step a time falls in
    starts = pd.to_datetime(["2023-01-01 00:00", "2023-01-01 01:00"])
    ends = starts + pd.Timedelta("1h")
    flux = _bounded_flux(starts, ends)
    assert flux.indexes["time"][0].left == starts[0]  # the interval carries both edges

    sampled = sample_field(
        flux,
        x=[-111.0, -111.0],
        y=[40.0, 40.0],
        times=["2023-01-01 00:45", "2023-01-01 01:45"],
        fill_value=0.0,
        step_starts=True,
    )
    assert sampled.tolist() == [1.0, 2.0]


def test_sample_field_step_starts_a_gap_between_bounds_gets_zero():
    starts = pd.to_datetime(["2023-01-01 00:00", "2023-01-01 02:00"])  # a missing hour
    ends = starts + pd.Timedelta("1h")
    flux = _bounded_flux(starts, ends)

    sampled = sample_field(
        flux,
        x=[-111.0],
        y=[40.0],
        times=["2023-01-01 01:30"],
        fill_value=0.0,
        step_starts=True,
    )
    assert sampled.tolist() == [0.0]


def test_sample_field_step_starts_refuses_overlapping_bounds():
    starts = pd.to_datetime(["2023-01-01 00:00", "2023-01-01 00:30"])
    ends = starts + pd.Timedelta("1h")  # the two steps overlap by 30 minutes
    flux = _bounded_flux(starts, ends)

    with pytest.raises(ValueError, match="overlap"):
        sample_field(
            flux,
            x=[-111.0],
            y=[40.0],
            times=["2023-01-01 00:45"],
            fill_value=0.0,
            step_starts=True,
        )


def _particles():
    # two particles; particle 2's second row is outside the flux field
    return pd.DataFrame(
        {
            "particle": [1, 1, 2, 2, 3],
            "lon": [-112.0, -111.0, -110.0, -100.0, -111.0],
            "lat": [40.0, 40.0, 41.0, 41.0, 41.0],
            "foot": [2.0, 3.0, 1.0, 5.0, 0.0],
            "time": pd.to_datetime(["2023-01-01"] * 5),
        }
    )


def test_particles_enhancement_sums_foot_times_flux_per_particle():
    per_particle = _particles().stilt.enhancement(_flux())

    assert per_particle.index.tolist() == [1, 2, 3]
    assert per_particle.tolist() == [2.0 * 0 + 3.0 * 1, 1.0 * 5 + 0.0, 0.0]
    assert per_particle.name == "enhancement"


def test_particles_enhancement_time_varying_needs_time():
    flux = _flux().expand_dims(time=pd.to_datetime(["2023-01-01"]))
    assert _particles().stilt.enhancement(flux).tolist() == [3.0, 5.0, 0.0]
    with pytest.raises(ValueError, match="time"):
        _particles().drop(columns="time").stilt.enhancement(flux)


def _footprint(point_receptor, values):
    grid = Grid(xmin=-112.5, xmax=-109.5, ymin=39.5, ymax=41.5, xres=1.0, yres=1.0)
    times = pd.to_datetime(["2023-01-01 12:00", "2023-01-01 11:00"])
    data = xr.DataArray(
        np.asarray(values, dtype=float),
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": [40.0, 41.0], "lon": [-112.0, -111.0, -110.0]},
    )
    return as_footprint(data, point_receptor, FootprintConfig(grid=grid), "test")


def test_footprint_enhancement_is_foot_times_flux_per_time_step(point_receptor):
    foot = _footprint(point_receptor, [np.ones((2, 3)), 2 * np.ones((2, 3))])

    enhancement = foot.stilt.enhancement(_flux())  # flux cells 0..5 sum to 15

    assert enhancement.dims == ("time",)
    assert enhancement.to_numpy().tolist() == [15.0, 30.0]
    assert float(enhancement.sum()) == 45.0


def test_footprint_enhancement_does_not_depend_on_dimension_order(point_receptor):
    foot = _footprint(point_receptor, [np.ones((2, 3)), 2 * np.ones((2, 3))])
    flux = _flux()
    expected = foot.stilt.enhancement(flux)

    swapped = foot.transpose("time", "lon", "lat")
    assert swapped.stilt.enhancement(flux).to_numpy().tolist() == [15.0, 30.0]
    xr.testing.assert_equal(swapped.stilt.enhancement(flux), expected)


def test_footprint_enhancement_of_a_time_summed_footprint_is_the_total(
    point_receptor,
):
    foot = _footprint(point_receptor, [np.ones((2, 3)), 2 * np.ones((2, 3))])
    total = foot.sum("time", keep_attrs=True)

    assert float(total.stilt.enhancement(_flux())) == 45.0
    flux_in_time = _flux().expand_dims(time=pd.to_datetime(["2023-01-01 11:00"]))
    with pytest.raises(ValueError, match="no time dimension"):
        total.stilt.enhancement(flux_in_time)


def test_footprint_enhancement_samples_flux_on_its_own_grid(point_receptor):
    foot = _footprint(point_receptor, [np.ones((2, 3)), np.zeros((2, 3))])
    # a single-cell axis has no spacing to bound, so it covers every point
    uniform = xr.DataArray(
        [[7.0]], dims=["lat", "lon"], coords={"lat": [40.5], "lon": [-111.0]}
    )
    assert float(foot.stilt.enhancement(uniform)[0]) == 7.0 * 6

    # two 2-degree cells reach only one degree past their centres
    narrow = xr.DataArray(
        [[7.0, 7.0]],
        dims=["lat", "lon"],
        coords={"lat": [40.5], "lon": [-114.5, -112.5]},
    )
    assert float(foot.stilt.enhancement(narrow)[0]) == 7.0 * 2  # only the -112 column


def test_footprint_enhancement_time_varying_flux(point_receptor):
    foot = _footprint(point_receptor, [np.ones((2, 3)), np.ones((2, 3))])
    flux = xr.DataArray(
        np.stack([np.ones((2, 3)), 10 * np.ones((2, 3))]),
        dims=["time", "lat", "lon"],
        coords={
            "time": pd.to_datetime(["2023-01-01 11:00", "2023-01-01 12:00"]),
            "lat": [40.0, 41.0],
            "lon": [-112.0, -111.0, -110.0],
        },
    )

    assert foot.stilt.enhancement(flux).to_numpy().tolist() == [60.0, 6.0]


def test_footprint_enhancement_refuses_a_finer_flux(point_receptor):
    foot = _footprint(point_receptor, [np.ones((2, 3)), np.ones((2, 3))])
    lons = np.arange(-112.0, -109.9, 0.25)
    fine = _flux(np.ones((2, lons.size)), lons=tuple(lons))
    with pytest.raises(ValueError, match="finer than the footprint's"):
        foot.stilt.enhancement(fine)


def test_footprint_enhancement_warns_when_cells_of_one_size_are_offset(
    point_receptor, recwarn
):
    import warnings

    foot = _footprint(point_receptor, [np.ones((2, 3)), np.ones((2, 3))])
    with pytest.warns(UserWarning, match="0.50 of a cell"):
        foot.stilt.enhancement(_flux(lons=(-112.5, -111.5, -110.5, -109.5)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        foot.stilt.enhancement(_flux())  # aligned
        foot.stilt.enhancement(
            _flux(np.ones((1, 2)), lons=(-112.3, -110.3), lats=(40.0,))
        )  # coarser


# ---------------------------------------------------------------------------
# sample_met: met columns at particle positions
# ---------------------------------------------------------------------------

#: Met times of the synthetic file, an hour apart.
_T0 = pd.Timestamp("2021-01-15 00:00")
_T1 = pd.Timestamp("2021-01-15 01:00")
#: Heights of its levels above ground; level 0 is the surface.
_LEVELS = [0.0, 50.0, 500.0, 2000.0]


def _met_file(path: Path) -> Path:
    """
    Write an ARL file with two times of uniform fields, and return its path.

    A 1 degree lat-lon grid from 20 E, 10 S, with terrain-following levels
    at the heights in ``_LEVELS``. Every field is uniform across the grid.
    From the first time to the second, PBLH goes from 500 to 800 m and each
    level's PRES rises by 10 hPa; the rest stay the same. SPHU stays above
    1/254, the smallest value ARL packing keeps in a uniform field.
    """
    from arlmet import File, Projection
    from arlmet import Grid as ArlGrid
    from arlmet.vertical import TerrainAxis

    grid = ArlGrid(
        projection=Projection(
            pole_lat=90.0,
            pole_lon=0.0,
            tangent_lat=1.0,
            tangent_lon=1.0,
            grid_size=0.0,
            orientation=0.0,
            cone_angle=0.0,
            sync_x=1.0,
            sync_y=1.0,
            sync_lat=-10.0,
            sync_lon=20.0,
        ),
        nx=20,
        ny=20,
    )
    ones = np.ones((20, 20), np.float32)
    with File(
        path, mode="w", source="TEST", grid=grid, vertical_axis=TerrainAxis(_LEVELS)
    ) as f:
        for time, pblh, dp in ((_T0, 500.0, 0.0), (_T1, 800.0, 10.0)):
            rs = f.create_recordset(time)
            rs.create_datarecord("SHGT", level=0, forecast=0, data=ones * 1500)
            rs.create_datarecord("PBLH", level=0, forecast=0, data=ones * pblh)
            for level, (pres, temp, sphu) in enumerate(
                [(850.0, 280.0, 0.010), (800.0, 275.0, 0.008), (650.0, 265.0, 0.006)],
                start=1,
            ):
                rs.create_datarecord(
                    "PRES", level=level, forecast=0, data=ones * (pres + dp)
                )
                rs.create_datarecord("TEMP", level=level, forecast=0, data=ones * temp)
                rs.create_datarecord("SPHU", level=level, forecast=0, data=ones * sphu)
    return path


def _particles_at(times, zagl):
    """Return particles at 25.5 E, 5 S, at the given times and heights."""
    return pd.DataFrame(
        {
            "particle": np.arange(1, len(times) + 1),
            "lon": 25.5,
            "lat": -5.0,
            "zagl": zagl,
            "time": pd.to_datetime(times),
        }
    )


def test_sample_met_interpolates_linearly_in_time_and_height(tmp_path):
    """The file's levels are 46.25, 462.5 and 1850 m above its 1500 m ground (Z* (1 - 1500 / 20000))."""
    met = _met_file(tmp_path / "met.arl")
    particles = _particles_at(
        [_T0, _T0 + pd.Timedelta(minutes=20), _T1], [100.0, 200.0, 300.0]
    )

    sampled = sample_met(particles, [met])

    np.testing.assert_allclose(sampled["mlht"], [500.0, 600.0, 800.0], rtol=1e-6)
    np.testing.assert_allclose(sampled["zsfc"], 1500.0, rtol=1e-6)
    # Between the first two levels every field is linear in height; a third of
    # the way to the next hour for the second row.
    up = (np.array([100.0, 200.0, 300.0]) - 46.25) / (462.5 - 46.25)
    pres = 850.0 - 50.0 * up + np.array([0.0, 10.0 / 3.0, 10.0])
    np.testing.assert_allclose(sampled["pres"], pres, rtol=1e-6)
    temp = 280.0 - 5.0 * up
    sphu = 0.010 - 0.002 * up
    dens = 100.0 * pres / (287.04 * temp * (1.0 + 0.61 * sphu))
    np.testing.assert_allclose(sampled["dens"], dens, rtol=1e-6)
    pd.testing.assert_frame_equal(sampled[particles.columns], particles)


def test_sample_met_samples_only_the_columns_asked_for(tmp_path):
    met = _met_file(tmp_path / "met.arl")
    particles = _particles_at([_T0], [100.0])

    sampled = sample_met(particles, [met], columns=["zsfc"])

    assert list(sampled.columns) == [*particles.columns, "zsfc"]


def test_sample_met_is_nan_below_the_ground_and_off_the_grid(tmp_path):
    """A column starts at the ground, so a height of zero or more has a value."""
    met = _met_file(tmp_path / "met.arl")
    particles = _particles_at([_T0, _T0, _T0], [10.0, -5.0, 100.0])
    particles.loc[2, "lon"] = 60.0  # east of the grid

    sampled = sample_met(particles, [met], columns=["pres", "zsfc"])

    assert np.isfinite(sampled["pres"].iloc[0])
    assert np.isnan(sampled["pres"].iloc[[1, 2]]).all()
    assert sampled["zsfc"].iloc[0] == pytest.approx(1500.0)
    assert np.isnan(sampled["zsfc"].iloc[2])


def test_sample_mets_default_top_is_a_runs_default_top():
    """sampling.py does not import HYSPLIT's package, so a test keeps the two equal."""
    assert HysplitConfig.model_fields["z_top"].default == sampling._DEFAULT_Z_TOP


@pytest.mark.parametrize("z_top", [sampling._DEFAULT_Z_TOP, 10000.0, None])
def test_a_terrain_following_file_is_read_with_the_runs_top(tmp_path, z_top):
    met = _met_file(tmp_path / "met.arl")
    particles = _particles_at([_T0], [100.0])

    with patch("arlmet.sample_points", wraps=arlmet.sample_points) as sample_points:
        sampled = sample_met(particles, [met], columns=["pres"], z_top=z_top)

    assert sample_points.call_args.kwargs["model_top"] == z_top
    assert np.isfinite(sampled["pres"]).all()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda p: p.assign(pres=1.0), "already have"),
        (lambda p: p.drop(columns="time"), "'time' column"),
        (lambda p: p.assign(time=_T1 + pd.Timedelta(minutes=1)), "outside the met"),
        (lambda p: p.assign(time=_T0 - pd.Timedelta(minutes=1)), "outside the met"),
    ],
    ids=["column-present", "no-time", "after-the-met", "before-the-met"],
)
def test_sample_met_rejects(tmp_path, change, message):
    met = _met_file(tmp_path / "met.arl")
    particles = change(_particles_at([_T0], [100.0]))

    with pytest.raises(ValueError, match=message):
        sample_met(particles, [met], columns=["pres"])


def test_sample_met_rejects_a_column_it_cannot_sample(tmp_path):
    met = _met_file(tmp_path / "met.arl")

    with pytest.raises(ValueError, match="cannot sample"):
        sample_met(_particles_at([_T0], [100.0]), [met], columns=["foot"])


def _richardson_column(heat: float, theta_top: float | None = None):
    """
    One column on 10 m levels: neutral air to 1000 m, stable above, and 5 m/s of shear above the lowest level.

    Without the temperature excess, the bulk Richardson number from the
    lowest level (10 m) is g (θ − θ₁)(z − 10) / (θ₁ · 25), which passes 0.25
    at about 1019 m.
    """
    from stilt.sampling import _richardson_mixed_layer

    zz = np.arange(10.0, 3001.0, 10.0)[None, :]
    theta = 300.0 + 0.01 * np.clip(zz - 1000.0, 0.0, None)
    if theta_top is not None:
        theta = np.full_like(zz, theta_top)
    uu = np.where(zz > 10.0, 5.0, 0.0)
    return _richardson_mixed_layer(
        zz,
        theta,
        uu,
        np.zeros_like(zz),
        np.ones_like(zz),
        np.array([0.3]),
        np.array([heat]),
        ks=6,
    )


def test_richardson_mixed_layer_ends_where_the_bulk_richardson_number_passes_a_quarter():
    stable = _richardson_column(heat=-20.0)

    g = 9.80616
    expected = (
        1010 + np.sqrt(1010**2 + 4 * (0.25 * 300 * 25 / (g * 0.01) - 10000))
    ) / 2
    assert stable[0] == pytest.approx(expected, rel=0.005)


def test_richardson_mixed_layer_is_deeper_in_unstable_air():
    # Upward heat flux gives the rising air a temperature excess.
    assert _richardson_column(heat=200.0)[0] > _richardson_column(heat=-20.0)[0] + 20


def test_richardson_mixed_layer_without_a_crossing_is_the_second_highest_level():
    assert _richardson_column(heat=-20.0, theta_top=300.0)[0] == pytest.approx(2990.0)
