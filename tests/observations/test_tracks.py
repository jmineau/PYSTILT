"""Tests for receptors_from_track on synthetic tracks."""

import numpy as np
import pandas as pd
import pytest

import stilt
from stilt.observations import TRACK_SCHEMA, check_track, receptors_from_track
from stilt.receptors import PointReceptor


def _drive(minutes=10, start="2024-07-01 18:00", lon0=-111.90, lon1=-111.83):
    """A car driving east, measured every second, with a measured column."""
    n = 60 * minutes
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "time": pd.date_range(start, periods=n, freq="1s"),
            "longitude": np.linspace(lon0, lon1, n),
            "latitude": 40.76,
            "altitude": 3.0,
            "ch4": 2.0 + rng.random(n) / 10,
        }
    )


def test_schema_names_the_required_columns():
    assert list(TRACK_SCHEMA) == ["time", "longitude", "latitude", "altitude"]


def test_check_track_names_the_missing_columns():
    check_track(_drive())
    with pytest.raises(ValueError, match=r"\['latitude', 'altitude'\]"):
        check_track(_drive().drop(columns=["latitude", "altitude"]))


def test_groups_of_five_minutes_by_default():
    track = _drive()
    receptors, table = receptors_from_track(track)

    assert all(isinstance(r, PointReceptor) for r in receptors)
    assert len(receptors) == 2
    first = receptors[0]
    # Mean of 18:00:00 to 18:04:59 is 18:02:29.5, rounded to the minute.
    assert first.time == pd.Timestamp("2024-07-01 18:02")
    assert first.longitude == round(track["longitude"][:300].mean(), 5)
    assert first.latitude == 40.76
    assert first.altitude == 3.0
    assert first.altitude_ref == "agl"
    assert (table["receptor"][:300] == first.id).all()
    assert (table["receptor"][300:] == receptors[1].id).all()


def test_table_keeps_every_row_its_order_index_and_columns():
    track = _drive().sample(frac=1, random_state=1)  # rows out of time order
    track.index = track.index + 1000
    receptors, table = receptors_from_track(track, every="5min")

    assert table.index.equals(track.index)
    assert list(table.columns) == [*track.columns, "receptor"]
    pd.testing.assert_frame_equal(table.drop(columns="receptor"), track)
    assert "receptor" not in track.columns  # the input is not changed
    # Receptors come in time order, whatever the row order.
    assert [r.time for r in receptors] == sorted(r.time for r in receptors)
    assert set(table["receptor"]) == {r.id for r in receptors}


def test_measured_values_average_per_receptor():
    track = _drive()
    receptors, table = receptors_from_track(track, every="5min")
    measured = table.groupby("receptor")["ch4"].mean()
    assert measured[receptors[0].id] == pytest.approx(track["ch4"][:300].mean())


def test_groups_start_on_the_clock():
    # A track from 18:03 to 18:12 splits at 18:05 and 18:10.
    _, table = receptors_from_track(_drive(start="2024-07-01 18:03"), every="5min")
    assert table["receptor"].value_counts(sort=False).tolist() == [120, 300, 180]


def test_spacing_groups_by_distance_along_the_track():
    track = _drive()
    receptors, table = receptors_from_track(track, spacing=1000)

    step = stilt.spatial.haversine_km(
        track.longitude[:-1].to_numpy(),
        track.latitude[:-1].to_numpy(),
        track.longitude[1:].to_numpy(),
        track.latitude[1:].to_numpy(),
    )
    along = np.concatenate([[0.0], np.cumsum(step)]) * 1000
    expected = np.floor(along / 1000)
    assert len(receptors) == len(np.unique(expected))
    codes = pd.factorize(table["receptor"])[0]
    assert (codes == expected).all()


def test_a_platform_at_rest_with_spacing_stays_in_one_group():
    track = _drive(minutes=20, lon1=-111.90)  # parked, measuring
    receptors, _ = receptors_from_track(track, spacing=500)
    assert len(receptors) == 1


@pytest.mark.parametrize("by", [{"every": "1h"}, {"spacing": 100_000.0}])
def test_a_pause_longer_than_gap_ends_a_group(by):
    before = _drive(minutes=5)
    after = _drive(minutes=5, start="2024-07-01 18:20")  # 15 minutes later
    track = pd.concat([before, after], ignore_index=True)

    receptors, table = receptors_from_track(track, **by)
    assert len(receptors) == 2
    assert table["receptor"].nunique() == 2
    assert table["receptor"][0] != table["receptor"][len(track) - 1]

    receptors, _ = receptors_from_track(track, gap="30min", **by)
    assert len(receptors) == 1


def test_mean_longitude_across_180():
    track = pd.DataFrame(
        {
            "time": pd.date_range("2024-07-01 18:00", periods=4, freq="1min"),
            "longitude": [179.9, 179.95, -179.95, -179.9],
            "latitude": 10.0,
            "altitude": 9000.0,
        }
    )
    (receptor,), _ = receptors_from_track(track, every="1h", altitude_ref="msl")
    assert receptor.longitude in (180.0, -180.0)
    assert receptor.id.endswith("msl")

    shifted = track.assign(longitude=[179.8, 179.9, 179.95, -179.95])
    (receptor,), _ = receptors_from_track(shifted, every="1h", altitude_ref="msl")
    assert receptor.longitude == pytest.approx(179.925)


def test_time_zone_aware_times_become_utc():
    track = _drive()
    track["time"] = track["time"].dt.tz_localize("America/Denver")
    receptors, _ = receptors_from_track(track)
    assert receptors[0].time == pd.Timestamp("2024-07-02 00:02")


def test_groups_with_one_id_share_the_receptor():
    track = _drive(minutes=1, lon1=-111.90).iloc[:40]  # 40 s at rest
    receptors, table = receptors_from_track(track, every="20s")
    # Two groups, with mean times 18:00:09.5 and 18:00:29.5, both rounded to
    # 18:00, at one position.
    assert len(receptors) == 1
    assert (table["receptor"] == receptors[0].id).all()


def test_receptors_and_table_go_into_a_project(tmp_path):
    receptors, table = receptors_from_track(_drive(), every="5min")
    project = stilt.Project.init(tmp_path / "project", starter=True)
    project.add_receptors(receptors)
    path = project.add_table("track", table)

    held = pd.read_parquet(path)
    assert len(held) == len(table)
    assert set(held["receptor"]) == set(project.receptors["receptor"])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"every": "5min", "spacing": 500}, "not both"),
        ({"every": "0min"}, "every must be positive"),
        ({"spacing": -1}, "spacing must be positive"),
        ({"gap": "0s"}, "gap must be positive"),
    ],
)
def test_bad_arguments_raise(change, message):
    with pytest.raises(ValueError, match=message):
        receptors_from_track(_drive(), **change)


def test_missing_values_and_empty_tracks_raise():
    track = _drive()
    track.loc[5, "latitude"] = np.nan
    with pytest.raises(ValueError, match=r"missing values in \['latitude'\]"):
        receptors_from_track(track)
    with pytest.raises(ValueError, match="no rows"):
        receptors_from_track(_drive().iloc[:0])
    with pytest.raises(ValueError, match="TRACK_SCHEMA"):
        receptors_from_track(_drive().drop(columns="time"))
