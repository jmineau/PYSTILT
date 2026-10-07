"""
Small objects many tests need, built in one place.

A test that needs a met config, a project config, a resolved variant, or a
receptor calls these, so a change to :class:`stilt.transport.hysplit.MetConfig`,
:class:`stilt.config.Variant`, or the receptor types is one edit here, not
one in every test module.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from stilt.config import ProjectConfig, Variant
from stilt.footprint.config import FootprintConfig
from stilt.receptors import PointReceptor
from stilt.transport import ModelInfo, TransportConfig
from stilt.transport.hysplit import HysplitConfig, MetConfig

#: The source id the ARL files these factories write carry.
SOURCE = "HRRR"

_ARL_BYTES: dict[str, bytes] = {}


def write_arl_file(path: Path, source: str = SOURCE) -> Path:
    """
    Write a small ARL file whose header names *source*, and return its path.

    One time step on a 20 x 20 grid: enough for a met's source to be read
    from it (:meth:`stilt.transport.hysplit.MetConfig.source`), not for
    HYSPLIT to run on it.
    """
    if source not in _ARL_BYTES:
        import tempfile

        import numpy as np
        import pandas as pd
        from arlmet import File
        from arlmet.grid import Grid, Projection
        from arlmet.vertical import PressureAxis

        grid = Grid(
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
        with tempfile.TemporaryDirectory() as tmp:
            arl = Path(tmp) / "arl"
            with File(
                arl,
                mode="w",
                source=source,
                grid=grid,
                vertical_axis=PressureAxis(levels=[0.0]),
            ) as f:
                rs = f.create_recordset(pd.Timestamp("2000-01-01"))
                rs.create_datarecord(
                    "PRSS", level=0, forecast=0, data=np.ones((20, 20), np.float32)
                )
            _ARL_BYTES[source] = arl.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_ARL_BYTES[source])
    return path


def make_met_config(
    directory: str | Path,
    *,
    file_format: str = "%Y%m%d_%H",
    file_tres: str = "1h",
    **fields: Any,
) -> MetConfig:
    """
    Return a met config of files named by hour (``%Y%m%d_%H``) under *directory*.

    When *directory* holds no files yet, an ARL file, ``header.arl``, is
    written there, so the met's source can be read as a project reads it;
    its name matches no hour. A directory of real met files is left alone.
    """
    directory = Path(directory)
    if not (directory.is_dir() and any(directory.iterdir())):
        write_arl_file(directory / "header.arl")
    return MetConfig(
        directory=directory, file_format=file_format, file_tres=file_tres, **fields
    )


def make_met_files(directory: Path, time: Any, n_hours: int) -> list[Path]:
    """
    Write hourly met files that cover a run of *n_hours* from *time*, and return them.

    They are named as :func:`make_met_config` expects, one hour beyond the
    run on each side, so :class:`stilt.transport.hysplit.Met` finds them.
    Each is :func:`write_arl_file`'s one step: the transport model cannot
    run on them.
    """
    import pandas as pd

    from stilt.meteorology import run_window

    start, end = run_window(time, n_hours)
    hours = pd.date_range(
        pd.Timestamp(start).floor("h") - pd.Timedelta(hours=1),
        pd.Timestamp(end).ceil("h") + pd.Timedelta(hours=1),
        freq="h",
    )
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for hour in hours:
        files.append(write_arl_file(directory / f"{hour:%Y%m%d_%H}"))
    return files


def make_project_config(tmp_path: Path, **settings: Any) -> ProjectConfig:
    """
    Return a project config with one met, ``hrrr``, and one variant of the defaults.

    The met is :func:`make_met_config` under ``tmp_path / "met"``. *settings*
    are other :class:`~stilt.ProjectConfig` settings, and replace the met
    and the variant when they name ``mets`` or ``variants``.
    """
    settings.setdefault("mets", {"hrrr": make_met_config(tmp_path / "met")})
    settings.setdefault("variants", {"hrrr": {}})
    return ProjectConfig(**settings)


#: A met that reads no file: an archive's, enough for a variant's settings.
MET = MetConfig(directory="/data/hrrr", download="hrrr")


def make_variant(
    name: str = "hrrr",
    *,
    transport: TransportConfig | None = None,
    met_config: MetConfig | None = None,
    footprint: FootprintConfig | None = None,
    realizations: int | None = None,
    **fields: Any,
) -> Variant:
    """
    Return a resolved variant without a project.

    *fields* are HYSPLIT parameters over ``n_hours=-24, numpar=100``, used
    when *transport* is not given. The met is :data:`MET` unless given.
    """
    if transport is None:
        transport = HysplitConfig(**{"n_hours": -24, "numpar": 100, **fields})
    elif fields:
        raise TypeError("Give HYSPLIT fields or a transport config, not both.")
    return Variant(
        name=name,
        met="hrrr",
        met_config=met_config if met_config is not None else MET,
        transport=transport,
        model=ModelInfo(name="hysplit", version="v5.1.0"),
        footprint=footprint,
        realizations=realizations,
    )


def make_receptor(
    time: dt.datetime = dt.datetime(2023, 1, 1, 12),
    longitude: float = -111.85,
    latitude: float = 40.77,
    altitude: float = 5.0,
    **attrs: Any,
) -> PointReceptor:
    """Return a point receptor at the University of Utah's WBB site; *attrs* become its extra columns."""
    return PointReceptor(
        time=time,
        longitude=longitude,
        latitude=latitude,
        altitude=altitude,
        attrs=attrs,
    )
