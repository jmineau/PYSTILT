"""A release must not be split when its group rows straddle a pandas read chunk."""

import pandas as pd
import pytest

from stilt.releases import MultiPointRelease, read_releases


def test_mixed_int_and_str_group_does_not_split_a_release(tmp_path):
    # Enough rows that pandas' low_memory reader types the column in more than one chunk,
    # with a numeric-id multipoint release sitting across the boundary and string ids after.
    n_filler = 300_000
    filler = pd.DataFrame(
        {
            "group": range(n_filler),
            "time": "2024-06-01 20:00:00",
            "longitude": -111.9,
            "latitude": 40.7,
            "altitude": 4.0,
        }
    )
    # distinct locations per filler row (point releases need no uniqueness, but keep it sane)
    filler["latitude"] = 40.0 + filler.index * 1e-6
    lons = [-111.90 + 0.001 * i for i in range(38)]
    multi = pd.DataFrame(
        {
            "group": 999_999_999,
            "time": "2024-06-01 21:00:00",
            "longitude": lons,
            "latitude": 40.70,
            "altitude": 4.0,
        }
    )
    tail = pd.DataFrame(
        {
            "group": ["dwell_a", "dwell_b"],
            "time": "2024-06-01 22:00:00",
            "longitude": [-111.8, -111.7],
            "latitude": [40.6, 40.5],
            "altitude": 4.0,
        }
    )
    # place the multipoint release so it straddles pandas' 2**18-row chunk
    head, rest = filler.iloc[: 2**18 - 19], filler.iloc[2**18 - 19 :]
    df = pd.concat([head, multi, rest, tail], ignore_index=True)
    f = tmp_path / "releases.csv"
    df.to_csv(f, index=False)

    # sanity: the naive reader really does split it, so this test guards the fix
    with pytest.warns(pd.errors.DtypeWarning, match="mixed types"):
        naive = pd.read_csv(f)
    assert naive.group.nunique() > df.group.astype(str).nunique()

    recs = read_releases(f)
    multis = [r for r in recs if isinstance(r, MultiPointRelease)]
    assert len(multis) == 1
    assert len(multis[0].coords()) == 38
    assert len(recs) == df.group.astype(str).nunique()
