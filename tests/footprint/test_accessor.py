"""Tests for a footprint as a DataArray and its .stilt accessor."""

import pytest
import xarray as xr

from stilt.receptors import PointReceptor

from ..fixtures.footprints import as_footprint, make_footprint

# ---------------------------------------------------------------------------
# A footprint is a DataArray (#107)

# ---------------------------------------------------------------------------


def test_receptor_coordinate_survives_arithmetic_and_reductions():
    """The receptor id is a coordinate, which xarray keeps where it drops attributes."""
    foot = make_footprint(n_times=2)
    rid = "202301011200_-111.85_40.77_5"
    for result in (foot * 2, foot.sum("time"), foot.isel(time=0), foot + foot):
        assert result["receptor"].item() == rid


def test_accessor_says_what_is_missing_without_attributes():
    """Older xarray drops attributes in arithmetic; that gives a clear error, not a wrong answer."""
    foot = make_footprint()
    bare = foot.copy()
    bare.attrs = {}
    with pytest.raises(ValueError, match="keep_attrs"):
        _ = bare.stilt.grid
    with xr.set_options(keep_attrs=True):
        assert (foot * 2).stilt.grid == foot.stilt.grid


def test_footprints_stack_along_the_receptor_coordinate():
    """xr.concat labels a stack of footprints by receptor with no extra work."""
    a = make_footprint()
    other = PointReceptor(
        time=a.stilt.receptor.time, longitude=-112.0, latitude=40.0, altitude=5.0
    )
    b = as_footprint(a * 2, other, a.stilt.config, "slv")
    stack = xr.concat([a, b], dim="receptor")
    assert stack["receptor"].values.tolist() == [
        str(a.stilt.receptor.id),
        str(other.id),
    ]
    assert stack.sizes["receptor"] == 2
