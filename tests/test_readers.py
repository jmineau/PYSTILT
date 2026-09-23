"""Tests for stilt.observations.readers against sample product files."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from netCDF4 import Dataset

import stilt
from stilt.observations import (
    pressure_altitudes,
    read_ggg_netcdf,
    read_ggg_oof,
    read_oco2,
    read_tccon,
    read_tropomi_ch4,
    slant_points,
)
from stilt.transforms import averaging_kernel_table

DATA = Path(__file__).parent / "data" / "products"
TROPOMI = (
    DATA
    / "S5P_OFFL_L2__CH4____20231019T192749_20231019T210918_31176_03_020500_20231021T113213.nc"
)
BLENDED = (
    DATA
    / "S5P_BLND_L2__CH4____20180430T001950_20180430T020120_02818_03_020400_20230613T235247.nc"
)
TCCON = DATA / "if20120823_20121201.public.qc.nc"
OCO2 = DATA / "oco2_LtCO2_231019_B11210Ar_synthetic.nc4"
OOF = DATA / "ha20220602.vav.ada.aia.oof"
GGG_PRIVATE = DATA / "xa20140709_20140709.private.nc"

REQUIRED = [
    "sounding_id",
    "time",
    "longitude",
    "latitude",
    "surface_altitude",
    "surface_pressure",
    "value",
    "uncertainty",
    "species",
    "units",
    "good",
    "ak_pressure",
    "ak",
    "pressure_levels",
]


def _check_common(df: pd.DataFrame) -> None:
    for col in REQUIRED:
        assert col in df.columns, col
    assert df["sounding_id"].is_unique
    assert pd.api.types.is_datetime64_dtype(df["time"])
    assert df["time"].dt.tz is None
    assert df["good"].dtype == bool
    row = df.iloc[0]
    assert len(row.ak_pressure) == len(row.ak)
    # surface first: pressures fall with index
    assert np.all(np.diff(row.pressure_levels) <= 0)
    assert np.all(np.diff(row.ak_pressure) <= 0)
    if "azimuth" in df.columns:
        assert ((df.azimuth >= 0) & (df.azimuth < 360)).all()


# -- TROPOMI operational -------------------------------------------------------


@pytest.fixture(scope="module")
def tropomi():
    return read_tropomi_ch4(TROPOMI)


def test_tropomi_columns_and_orientation(tropomi):
    _check_common(tropomi)
    assert (tropomi.species == "xch4").all() and (tropomi.units == "ppb").all()
    assert len(tropomi) > 50  # 10 x 10 pixels minus masked ones
    assert tropomi.time.dt.strftime("%Y-%m-%d").eq("2023-10-19").all()
    good = tropomi[tropomi.good].iloc[0]
    assert 600 < good.surface_pressure < 1000  # hPa, not Pa
    assert len(good.pressure_levels) == 13 and len(good.ak) == 12
    assert good.pressure_levels[0] == pytest.approx(good.surface_pressure)
    assert good.pressure_levels[-1] == pytest.approx(0.0, abs=0.5)  # psfc - 12 dp
    assert good.ak_pressure[0] == pytest.approx(
        good.surface_pressure
        - 0.5 * (good.pressure_levels[0] - good.pressure_levels[1])
    )
    # TROPOMI's kernel is low in the top layer and near one in the troposphere
    assert good.ak[-1] < 0.85 < good.ak[0]
    assert good.altitude_levels[0] == pytest.approx(good.surface_altitude, abs=1.0)
    assert np.all(np.diff(good.altitude_levels) > 0)
    assert 1700 < np.nanmean(good.apriori) < 2100  # ppb
    assert 1700 < good.value < 2100
    assert (tropomi.good == (tropomi.qa_value >= 0.5)).all()
    assert tropomi.sounding_id.str.startswith("31176_").all()


def test_tropomi_box_filter(tropomi):
    lon0, lat0 = tropomi.longitude.median(), tropomi.latitude.median()
    sub = read_tropomi_ch4(
        TROPOMI,
        lon_range=(lon0 - 0.05, lon0 + 0.05),
        lat_range=(lat0 - 0.05, lat0 + 0.05),
    )
    assert 0 < len(sub) < len(tropomi)
    assert sub.longitude.between(lon0 - 0.05, lon0 + 0.05).all()
    empty = read_tropomi_ch4(TROPOMI, lon_range=(0.0, 1.0))
    assert len(empty) == 0 and "value" in empty.columns


def test_tropomi_recipe_to_receptors_and_kernels(tropomi):
    df = tropomi[tropomi.good].head(3)
    receptors = [
        stilt.Receptor.from_points(
            r.time,
            slant_points(
                r.longitude,
                r.latitude,
                r.altitude_levels[r.altitude_levels < r.surface_altitude + 3000],
                zenith=r.zenith,
                azimuth=r.azimuth,
            ),
            altitude_ref="msl",
        )
        for r in df.itertuples()
    ]
    assert all(
        rc.altitudes[0] == pytest.approx(a, abs=1.0)
        for rc, a in zip(receptors, df.surface_altitude, strict=True)
    )
    table = averaging_kernel_table(receptors, levels=df.ak_pressure, values=df.ak)
    assert set(table.columns) >= {"receptor", "level", "value"}
    assert len(table) == 3 * 12


# -- TROPOMI blended -----------------------------------------------------------


def test_tropomi_blended_flat_layout():
    df = read_tropomi_ch4(BLENDED)
    _check_common(df)
    assert len(df) == 40
    assert "zenith" not in df.columns  # the blended files carry no viewing angles
    assert df.sounding_id.str.startswith("02818_").all()
    good = df[df.good].iloc[0]
    assert good.value == pytest.approx(good.value)  # the blended value is finite
    assert good.value != good.xch4_bias_corrected or good.value != good.xch4_uncorrected
    assert len(good.pressure_levels) == 13


# -- TCCON ---------------------------------------------------------------------


def test_tccon_xco2_and_xch4():
    df = read_tccon(TCCON)
    _check_common(df)
    assert len(df) == 24
    assert (df.species == "xco2").all() and (df.units == "ppm").all()
    assert df.good.all()  # a qc file holds only good data
    row = df.iloc[0]
    assert row.surface_altitude == pytest.approx(270.0)  # zobs km -> m
    assert 950 < row.surface_pressure < 1050
    assert len(row.ak) == 51 and row.ak_pressure[0] > 1000
    assert row.pressure_levels[0] > 1000  # atm -> hPa
    assert row.altitude_levels[0] == 0.0 and row.altitude_levels[-1] == 70_000.0
    assert 380 < row.value < 420 and 0 < row.uncertainty < 5
    assert row.zenith == row.solar_zenith and 0 <= row.azimuth < 360
    assert row.sounding_id.startswith("indianapolis01_2012")
    ch4 = read_tccon(TCCON, "xch4")
    assert (ch4.units == "ppm").all() and 1.6 < ch4.value.iloc[0] < 2.0
    assert len(ch4.iloc[0].apriori) == 51


def test_tccon_time_range_and_errors():
    df = read_tccon(TCCON)
    t0, t1 = df.time.iloc[5], df.time.iloc[10]
    sub = read_tccon(TCCON, time_range=(t0, t1))
    assert len(sub) == 6
    with pytest.raises(ValueError, match="column variable"):
        read_tccon(TCCON, "co2")
    with pytest.raises(ValueError, match="no 'xhf'"):
        read_tccon(TCCON, "xhf")


def test_tccon_slant_recipe():
    df = read_tccon(TCCON).head(2)
    for r in df.itertuples():
        alts = pressure_altitudes(
            r.pressure_levels,
            surface_pressure=r.surface_pressure,
            surface_altitude=r.surface_altitude,
            top=r.surface_altitude + 5000.0,
        )
        # the prior grid's lowest level (1.008 atm) is below ground and dropped,
        # so the station itself is the anchor to prepend
        assert alts[0] > r.surface_altitude
        alts = np.concatenate(([r.surface_altitude], alts))
        points = slant_points(
            r.longitude, r.latitude, alts, zenith=r.zenith, azimuth=r.azimuth
        )
        assert len(points) == len(alts) and points[0][2] == r.surface_altitude


# -- OCO-2 (synthetic layout) --------------------------------------------------


def test_oco2_lite_layout():
    df = read_oco2(OCO2)
    _check_common(df)
    assert len(df) == 6
    assert (df.species == "xco2").all() and (df.units == "ppm").all()
    assert df.good.tolist() == [True, True, False, True, False, True]
    assert np.isnan(df.value.iloc[2])  # the product's fill value
    row = df.iloc[0]
    assert len(row.pressure_levels) == 20
    assert row.pressure_levels[0] == pytest.approx(
        row.surface_pressure
    )  # level 20 first
    assert row.ak[0] == pytest.approx(1.1) and row.ak[-1] == pytest.approx(0.3)
    assert len(row.pressure_weight) == 20 and row.apriori_column == pytest.approx(417.0)
    assert row.sounding_id == "2023101919480000"
    assert row.time == pd.Timestamp("2023-10-19T19:48:00")
    sub = read_oco2(OCO2, lat_range=(40.75, 40.85))
    assert len(sub) == 3


def test_oco2_recipe_with_pressure_altitudes():
    df = read_oco2(OCO2)
    r = df[df.good].iloc[0]
    alts = pressure_altitudes(
        r.pressure_levels,
        surface_pressure=r.surface_pressure,
        surface_altitude=r.surface_altitude,
        top=r.surface_altitude + 3000.0,
    )
    assert alts[0] == pytest.approx(r.surface_altitude)
    receptor = stilt.Receptor.from_points(
        r.time,
        slant_points(r.longitude, r.latitude, alts, zenith=r.zenith, azimuth=r.azimuth),
        altitude_ref="msl",
    )
    assert receptor.altitudes[0] == pytest.approx(r.surface_altitude)


# -- GGG2020: EM27/SUN .oof and private netCDF ---------------------------------


def test_ggg_oof_columns_units_and_time():
    df = read_ggg_oof(OOF)
    assert len(df) == 12
    assert (
        df.sounding_id.is_unique and df.sounding_id.iloc[0] == "ha20220602s0e00a.0001"
    )
    assert pd.api.types.is_datetime64_dtype(df.time) and df.time.dt.tz is None
    # year 2022, day 153, 14.840 UT hours -> 2022-06-02 14:50:24
    assert df.time.iloc[0] == pd.Timestamp("2022-06-02 14:50:24")
    assert (df.species == "xch4").all() and (df.units == "ppm").all()
    assert df.good.dtype == bool and (df.good == (df.flag == 0)).all()
    row = df.iloc[0]
    assert row.latitude == pytest.approx(40.766) and row.longitude == pytest.approx(
        -111.847
    )
    assert row.surface_altitude == pytest.approx(1470.0)  # zobs km -> m
    assert row.surface_pressure == pytest.approx(853.3)
    assert row.zenith == pytest.approx(59.92) and row.azimuth == pytest.approx(85.53)
    assert row.solar_zenith == row.zenith and 0 <= row.azimuth < 360
    assert row.value == pytest.approx(1.8698) and row.uncertainty == pytest.approx(
        0.0020
    )
    # no kernel or prior in a .oof: the columns are absent, not faked
    for col in ("ak", "ak_pressure", "pressure_levels", "altitude_levels"):
        assert col not in df.columns
    # the other column variables ride along under their own names
    assert 400 < df.xco2.iloc[0] < 440 and df.xco2_error.iloc[0] > 0
    assert 1.0 < df.zmin.iloc[0] < 2.0
    co = read_ggg_oof(OOF, "xco")
    assert (co.units == "ppb").all() and 50 < co.value.iloc[0] < 200
    luft = read_ggg_oof(OOF, "xluft")
    assert (luft.units == "").all() and 0.99 < luft.value.iloc[0] < 1.01


def test_ggg_oof_errors():
    with pytest.raises(ValueError, match="column variable"):
        read_ggg_oof(OOF, "ch4")
    with pytest.raises(ValueError, match="no 'xhf' column"):
        read_ggg_oof(OOF, "xhf")


def test_ggg_oof_slant_recipe_windows():
    """The EM27 recipe: one receptor per averaging window from the solar angles."""
    df = read_ggg_oof(OOF)
    windows = (
        df[df.good]
        .set_index("time")[
            ["longitude", "latitude", "surface_altitude", "zenith", "azimuth"]
        ]
        .resample("10min")
        .mean()
        .dropna()
    )
    assert len(windows) >= 1
    w = windows.iloc[0]
    alts = np.linspace(w.surface_altitude, w.surface_altitude + 3000.0, 20)
    points = slant_points(
        w.longitude, w.latitude, alts, zenith=w.zenith, azimuth=w.azimuth
    )
    receptor = stilt.Receptor.from_points(windows.index[0], points, altitude_ref="msl")
    assert len(receptor) == 20
    # morning at ~15 UT in Salt Lake City: the sun is east, so the path leans east
    assert receptor.longitudes[-1] > receptor.longitudes[0]


def test_ggg_private_netcdf_expands_kernels_and_shares_priors():
    df = read_ggg_netcdf(GGG_PRIVATE, "xch4")
    _check_common(df)
    assert len(df) == 4
    assert df.sounding_id.iloc[0] == "xa20140709s0e00a.0001"  # private files keep names
    assert (df.species == "xch4").all() and (df.units == "ppm").all()
    assert (df.good == (df.flag == 0)).all() if "flag" in df.columns else True
    row = df.iloc[0]
    assert row.surface_altitude == pytest.approx(240.0, abs=20)  # zobs km -> m
    assert 900 < row.surface_pressure < 1050
    assert 79 < row.zenith < 81
    # kernel: 51 levels on the site's median-pressure grid, surface first
    assert (
        len(row.ak) == 51 and len(row.ak_pressure) == 51 and row.ak_pressure[0] > 1000
    )
    with Dataset(GGG_PRIVATE) as ds:
        table = np.asarray(ds["ak_xch4"][:], dtype=float)
        bins = np.asarray(ds["ak_slant_xch4_bin"][:], dtype=float)
        am = float(ds["o2_7885_am_o2"][0])
    slant = row.value * am
    assert bins[0] < slant < bins[-1] and not row.ak_extrapolated
    # each level is the linear interpolation of that level's table row at the slant xgas
    expected = np.array([np.interp(slant, bins, table[k]) for k in range(51)])
    assert row.ak == pytest.approx(expected)
    assert (table.min(axis=1) <= row.ak + 1e-9).all() and (
        row.ak - 1e-9 <= table.max(axis=1)
    ).all()
    # priors: the four spectra share one prior_index, so one prior row
    assert all(
        np.array_equal(r.pressure_levels, row.pressure_levels) for r in df.itertuples()
    )
    assert row.pressure_levels[0] > 900 and np.all(
        np.diff(row.pressure_levels) < 0
    )  # atm -> hPa
    assert row.altitude_levels[0] == 0.0 and row.altitude_levels[-1] == 70_000.0
    assert len(row.apriori) == 51 and 1.5 < row.apriori[0] < 2.1  # 'parts' -> ppm
    co2 = read_ggg_netcdf(GGG_PRIVATE, "xco2")
    assert (co2.units == "ppm").all() and 380 < co2.value.iloc[0] < 420
    assert 380 < co2.iloc[0].apriori[0] < 420


def test_read_tccon_is_the_public_ggg_layout():
    a = read_tccon(TCCON, "xch4")
    b = read_ggg_netcdf(TCCON, "xch4")
    pd.testing.assert_frame_equal(a, b)
    assert not a.ak_extrapolated.any()  # public kernels are stored per spectrum
    row = a.iloc[0]
    # the public prior_ch4 is in ppb; apriori is in the species' ppm
    assert 1.5 < row.apriori[0] < 2.1
