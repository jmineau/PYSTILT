"""Tests for select_observations_spatial."""

from stilt.observations import Observation
from stilt.observations.selection import select_observations_spatial


def _obs(lon: float, lat: float, obs_id: str | None = None) -> Observation:
    return Observation(
        sensor="oco2",
        species="xco2",
        time="2023-01-01 21:00:00",
        longitude=lon,
        latitude=lat,
        observation_id=obs_id,
    )


# A dense regular grid of observations over a domain centred on SLC.
_DOMAIN_LONS = [-112.5, -112.0, -111.5, -111.0, -110.5]
_DOMAIN_LATS = [40.0, 40.5, 41.0, 41.5, 42.0]
_ALL_OBS = [
    _obs(lon, lat, f"obs-{i}-{j}")
    for i, lon in enumerate(_DOMAIN_LONS)
    for j, lat in enumerate(_DOMAIN_LATS)
]  # 25 observations total

_SITE_LON = -111.85
_SITE_LAT = 40.77


def test_returns_subset_of_input():
    result = select_observations_spatial(
        _ALL_OBS,
        site_longitude=_SITE_LON,
        site_latitude=_SITE_LAT,
        near_field_dlon=0.5,
        near_field_dlat=0.5,
        near_field_cols=3,
        near_field_rows=3,
        background_cols=2,
        background_rows=2,
        domain_lon_range=(-112.5, -110.5),
        domain_lat_range=(40.0, 42.0),
    )

    assert all(obs in _ALL_OBS for obs in result)
    assert len(result) > 0


def test_result_sorted_by_latitude():
    result = select_observations_spatial(
        _ALL_OBS,
        site_longitude=_SITE_LON,
        site_latitude=_SITE_LAT,
        near_field_dlon=0.5,
        near_field_dlat=0.5,
        near_field_cols=3,
        near_field_rows=3,
        background_cols=2,
        background_rows=2,
        domain_lon_range=(-112.5, -110.5),
        domain_lat_range=(40.0, 42.0),
    )

    lats = [o.latitude for o in result]
    assert lats == sorted(lats)


def test_no_duplicates():
    result = select_observations_spatial(
        _ALL_OBS,
        site_longitude=_SITE_LON,
        site_latitude=_SITE_LAT,
        near_field_dlon=0.5,
        near_field_dlat=0.5,
        near_field_cols=3,
        near_field_rows=3,
        background_cols=2,
        background_rows=2,
        domain_lon_range=(-112.5, -110.5),
        domain_lat_range=(40.0, 42.0),
    )

    ids = [id(o) for o in result]
    assert len(ids) == len(set(ids))


def test_near_field_favours_site_adjacent_observations():
    # Dense near-field grid, no background — should favour obs near the site.
    result = select_observations_spatial(
        _ALL_OBS,
        site_longitude=_SITE_LON,
        site_latitude=_SITE_LAT,
        near_field_dlon=0.2,
        near_field_dlat=0.2,
        near_field_cols=3,
        near_field_rows=3,
        background_cols=0,
        background_rows=0,
        domain_lon_range=(-112.5, -110.5),
        domain_lat_range=(40.0, 42.0),
    )

    # All selected observations should be within ≈1° of the site.
    for obs in result:
        assert abs(obs.longitude - _SITE_LON) < 1.0
        assert abs(obs.latitude - _SITE_LAT) < 1.0


def test_empty_observations_returns_empty():
    result = select_observations_spatial(
        [],
        site_longitude=_SITE_LON,
        site_latitude=_SITE_LAT,
        near_field_dlon=0.5,
        near_field_dlat=0.5,
        near_field_cols=3,
        near_field_rows=3,
        background_cols=2,
        background_rows=2,
        domain_lon_range=(-112.5, -110.5),
        domain_lat_range=(40.0, 42.0),
    )

    assert result == []


def test_single_observation_always_selected():
    single = _obs(-111.85, 40.77, "only")
    result = select_observations_spatial(
        [single],
        site_longitude=_SITE_LON,
        site_latitude=_SITE_LAT,
        near_field_dlon=0.5,
        near_field_dlat=0.5,
        near_field_cols=3,
        near_field_rows=3,
        background_cols=2,
        background_rows=2,
        domain_lon_range=(-112.5, -110.5),
        domain_lat_range=(40.0, 42.0),
    )

    assert result == [single]
