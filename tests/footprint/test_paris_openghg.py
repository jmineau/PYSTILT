"""
The PARIS export read back by openghg's own parser.

openghg is not a dependency (it needs Python 3.13 or older and pins old pandas
and numpy), so these tests skip unless it is installed: ``pip install
openghg`` in an environment with Python 3.13 or older, and PYSTILT with its
test tools.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from .test_paris import LAT, LON, RECEPTOR_TIME, exits_table, layered

pytest.importorskip("openghg")

# openghg's own deprecation and decoding warnings are not ours to fix.
pytestmark = pytest.mark.filterwarnings(
    "ignore::DeprecationWarning", "ignore::FutureWarning"
)


@pytest.fixture
def paris_file(tmp_path):
    """Two receptors an hour apart, 30 hourly layers each, written as one PARIS file."""
    exits = exits_table([(1, "n", -111.2, 43.0, 150.0), (2, "w", -113.0, 41.0, 2500.0)])
    datasets = []
    for k in range(2):
        foot = layered(range(1, 31), lat=LAT, lon=LON)
        shift = pd.Timedelta(hours=k)
        foot = foot.assign_coords(
            time=foot.time + shift,
            receptor=f"{RECEPTOR_TIME + shift:%Y%m%d%H%M}_-111.85_40.77_10",
        )
        datasets.append(foot.stilt.to_paris(exits=exits, n_particles=4))
    ds = xr.concat(datasets, dim="time", combine_attrs="drop_conflicts")
    path = tmp_path / "WBB-10magl_STILT_HRRR_TEST_co2_202401.nc"
    ds.to_netcdf(path)
    return path


def parse(path):
    """Read a PARIS file with openghg's parser, as ``standardise_footprint`` does."""
    from openghg.standardise.footprints import parse_paris

    out = parse_paris(
        filepath=path,
        site="wbb",
        domain="stilt_test",
        model="STILT",
        inlet="10m",
        met_model="hrrr",
        species="co2",
        time_resolved=True,
    )
    (entry,) = out.values()
    return entry["data"], entry["metadata"]


def test_openghg_reads_the_file(paris_file):
    data, metadata = parse(paris_file)

    assert metadata["time_resolved"] == "True"
    assert metadata["start_date"].startswith("2024-01-15 18:00")
    assert data.H_back.values.tolist() == list(range(24))
    for name in ("fp", "fp_time_resolved", "fp_residual"):
        assert name in data
    first = data.isel(time=0, lat=0, lon=0)
    # Each synthetic layer's value is its hours back.
    assert first.fp_time_resolved.values.tolist() == list(range(24))
    assert float(first.fp_residual) == sum(range(24, 31))
    assert float(first.fp) == sum(range(1, 31))
    assert float(data.particle_locations_n.isel(time=0).sum()) == pytest.approx(0.25)
    assert float(data.particle_locations_w.isel(time=1).sum()) == pytest.approx(0.25)


def test_openghg_validates_the_data(paris_file):
    from openghg.store import Footprints

    data, _ = parse(paris_file)
    Footprints.validate_data(data, time_resolved=True, source_format="PARIS")


def test_openghg_pairs_h_back_1_with_the_hour_before_the_footprint_time(paris_file):
    """Openghg's flux windows: H_back = k is the flux hour that starts k hours before."""
    from openghg.analyse import _modelled_obs

    data, _ = parse(paris_file)
    rolling = getattr(_modelled_obs, "_make_hf_flux_rolling_avg_array", None)
    if rolling is None:
        pytest.skip("this openghg has no _make_hf_flux_rolling_avg_array")

    times = pd.date_range("2024-01-13 00:00", "2024-01-16 00:00", freq="h")
    hour = (times - times[0]) / pd.Timedelta(hours=1)
    flux = xr.DataArray(
        np.broadcast_to(np.asarray(hour)[:, None, None], (len(times), 3, 4)).copy(),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": data.lat.values, "lon": data.lon.values},
    )
    windows = rolling(flux, data.fp_time_resolved)
    at_receptor = windows.sel(time=RECEPTOR_TIME).isel(lat=0, lon=0)
    expected = (RECEPTOR_TIME - times[0]) / pd.Timedelta(hours=1)
    # The flux paired with H_back = 1 is the hour from 17:00 to 18:00, which
    # PYSTILT's layer -1 covers.
    assert float(at_receptor.sel(H_back=1)) == expected - 1
    assert float(at_receptor.sel(H_back=0)) == expected
