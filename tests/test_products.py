"""Tests for stilt.observations.products against sample product files."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import stilt
from stilt.observations import (
    pressure_altitudes,
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
