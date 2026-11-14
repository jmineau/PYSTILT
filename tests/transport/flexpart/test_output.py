"""Tests for reading FLEXPART 11's particle output."""

import datetime as dt
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from stilt.particles import add_foot, check_particles
from stilt.transport.flexpart import read_partoutput

RECEPTOR_TIME = dt.datetime(2023, 6, 1, 18)

#: Five particles of a real FLEXPART 11 run on GFS (data/make_sample.py says how it was made).
SAMPLE = Path(__file__).parent / "data" / "partoutput_20230601180000_init.nc"


def write_partoutput(path, times, fields, start="2023-06-01 18:00", grid=None):
    """
    Write a file as FLEXPART 11 does (``writeheader_partoutput``, ``write_to_file``).

    *times* are seconds from *start*; *fields* maps a variable to a
    (time, particle) array with -1 where FLEXPART fills. FLEXPART's Fortran
    (time, particle) is (particle, time) in the file, as here.
    """
    n_particles = next(iter(fields.values())).shape[1]
    with netCDF4.Dataset(path, "w", format="NETCDF4") as nc:
        nc.createDimension("time", None)
        nc.createDimension("particle", None)
        particle = nc.createVariable("particle", "i4", ("particle",))
        particle.long_name = "particle index"
        time = nc.createVariable("time", "i4", ("time",))
        time.units = f"seconds since {start}"
        time.calendar = "proleptic_gregorian"
        time[:] = times
        particle[:] = np.arange(1, n_particles + 1)
        for name, values in fields.items():
            var = nc.createVariable(
                name,
                "f4",
                ("particle", "time"),
                fill_value=-1.0,
                zlib=True,
                complevel=1,
                chunksizes=(n_particles, 1),
            )
            var.units = "meters"
            var[:] = np.asarray(values).T
        if grid is not None:
            nc.createDimension("longitude", 2)
            nc.createDimension("latitude", 2)
            var = nc.createVariable(grid, "f4", ("latitude", "longitude", "time"))
            var[:] = np.ones((2, 2, len(times)))


def test_reads_flexpart_output(tmp_path):
    particles = read_partoutput(SAMPLE, RECEPTOR_TIME)

    check_particles(particles)
    assert len(particles) == 5 * 24
    assert particles["age"].iloc[0] == -60
    assert particles["age"].iloc[-1] == -1440
    assert particles["particle"].iloc[:5].tolist() == [1, 2, 3, 4, 5]
    first = particles.iloc[0]
    # Released at 40.766 N, 111.848 W, 25 m above ground, an hour before.
    assert first["lon"] == pytest.approx(-111.88, abs=0.05)
    assert first["lat"] == pytest.approx(40.74, abs=0.05)
    assert 0 < first["zagl"] < 100
    assert first["pres"] == pytest.approx(806, abs=5)
    assert {"mlht", "dens", "temp", "sphu", "zsfc", "pv"} <= set(particles.columns)


def test_foot_from_flexpart_output():
    particles = read_partoutput(SAMPLE, RECEPTOR_TIME)

    foot = add_foot(particles, [], veght=0.5)["foot"]

    below = particles["zagl"] < 0.5 * particles["mlht"]
    assert below.any() and (~below).any()
    assert (foot[~below] == 0).all()
    # A whole hour below half the mixed layer, divided by the air in it.
    expected = 0.02897 * 3600 / (particles["dens"] * 0.5 * particles["mlht"])
    np.testing.assert_allclose(foot[below], expected[below])


def backward_fields():
    """Three particles over three hours back; particle 3 released an hour late, particle 2 gone at the last."""
    fill = -1.0
    return {
        "lon": np.array(
            [[-112.0, -112.1, fill], [-112.2, fill, -1.0], [-112.4, fill, -1.5]]
        ),
        "lat": np.array([[40.7, 40.8, fill], [40.9, fill, 41.0], [41.1, fill, 41.2]]),
        "z": np.array([[10.0, 20.0, fill], [50.0, fill, 30.0], [80.0, fill, 60.0]]),
        "hmix": np.array(
            [[900.0, 900.0, fill], [800.0, fill, 850.0], [700.0, fill, fill]]
        ),
        "prs": np.array(
            [
                [85000.0, 85100.0, fill],
                [84000.0, fill, 84500.0],
                [83000.0, fill, 83500.0],
            ]
        ),
        "pv": np.array([[0.5, 0.6, fill], [0.7, fill, 0.8], [0.9, fill, 1.0]]),
    }


def test_reads_backward_run_as_particle_table(tmp_path):
    path = tmp_path / "partoutput_20230601180000_init.nc"
    write_partoutput(path, [-3600, -7200, -10800], backward_fields())

    particles = read_partoutput(path, RECEPTOR_TIME)

    check_particles(particles)
    assert list(particles.columns[:5]) == ["particle", "age", "lon", "lat", "zagl"]
    assert particles["particle"].tolist() == [1, 2, 1, 3, 1, 3]
    assert particles["age"].tolist() == [-60, -60, -120, -120, -180, -180]
    assert particles["particle"].dtype == np.int32
    assert particles["age"].dtype == np.int32


def test_longitude_of_minus_one_is_kept(tmp_path):
    # FLEXPART's fill value is -1 for every field, so only z says a row is filled.
    path = tmp_path / "partoutput_20230601180000_init.nc"
    write_partoutput(path, [-3600, -7200, -10800], backward_fields())

    particles = read_partoutput(path, RECEPTOR_TIME)

    row = particles[(particles["particle"] == 3) & (particles["age"] == -120)]
    assert row["lon"].item() == -1.0


def test_renames_fields_hysplit_also_writes(tmp_path):
    path = tmp_path / "partoutput_20230601180000_init.nc"
    write_partoutput(path, [-3600, -7200, -10800], backward_fields())

    particles = read_partoutput(path, RECEPTOR_TIME)

    assert {"mlht", "pres", "pv"} <= set(particles.columns)
    assert not {"z", "hmix", "prs"} & set(particles.columns)
    first = particles.iloc[0]
    assert first["pres"] == pytest.approx(850.0)
    assert first["mlht"] == pytest.approx(900.0)


def test_fill_in_other_fields_becomes_nan(tmp_path):
    path = tmp_path / "partoutput_20230601180000_init.nc"
    write_partoutput(path, [-3600, -7200, -10800], backward_fields())

    particles = read_partoutput(path, RECEPTOR_TIME)

    row = particles[(particles["particle"] == 3) & (particles["age"] == -180)]
    assert np.isnan(row["mlht"].item())


def test_reads_files_of_one_run_in_any_order(tmp_path):
    fields = backward_fields()
    first = tmp_path / "partoutput_20230601180000_init.nc"
    second = tmp_path / "partoutput_20230601160000.nc"
    write_partoutput(first, [-3600], {k: v[:1] for k, v in fields.items()})
    write_partoutput(second, [-7200, -10800], {k: v[1:] for k, v in fields.items()})

    together = read_partoutput([second, first], RECEPTOR_TIME)

    one = tmp_path / "one.nc"
    write_partoutput(one, [-3600, -7200, -10800], fields)
    assert together.equals(read_partoutput(one, RECEPTOR_TIME))


def test_age_counts_from_receptor_time(tmp_path):
    path = tmp_path / "partoutput.nc"
    write_partoutput(
        path, [-3600, -7200, -10800], backward_fields(), start="2023-06-01 19:00"
    )

    particles = read_partoutput(path, RECEPTOR_TIME)

    assert particles["age"].unique().tolist() == [0, -60, -120]


def test_rejects_times_not_whole_minutes(tmp_path):
    path = tmp_path / "partoutput.nc"
    write_partoutput(path, [-90, -7200, -10800], backward_fields())

    with pytest.raises(ValueError, match="whole minutes"):
        read_partoutput(path, RECEPTOR_TIME)


def test_rejects_file_without_positions(tmp_path):
    path = tmp_path / "partoutput.nc"
    fields = backward_fields()
    del fields["z"]
    write_partoutput(path, [-3600, -7200, -10800], fields)

    with pytest.raises(ValueError, match="HEIGHT"):
        read_partoutput(path, RECEPTOR_TIME)


def test_rejects_domain_filling_grid(tmp_path):
    path = tmp_path / "partoutput.nc"
    fields = backward_fields()
    del fields["hmix"]
    write_partoutput(path, [-3600, -7200, -10800], fields, grid="hmix")

    with pytest.raises(ValueError, match="MDOMAINFILL"):
        read_partoutput(path, RECEPTOR_TIME)


def test_rejects_no_files():
    with pytest.raises(ValueError, match="No partoutput"):
        read_partoutput([], RECEPTOR_TIME)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_partoutput(tmp_path / "partoutput.nc", RECEPTOR_TIME)
