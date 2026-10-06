"""
The output directory, where particles and footprints are kept.

A project's results live in an output directory that several projects can
share. It has one tree per kind of result, and inside each a folder per set
of settings, in hive form so every tree reads as one dataset::

    <output>/
      particles/
        settings=hrrr-a3f9c2/            variant name + short hash of the transport settings
          _settings.yaml
          date=2024-07-01/<receptor>.parquet
        settings=hrrr-err-0-7be104/
      footprints/
        settings=hrrr-93278c/            variant name + short hash of transport + footprint settings
          _settings.yaml                 names the particles folder it was made from
          date=2024-07-01/<receptor>.parquet
        settings=hrrr-hexes-1b2c3d/      same particles, other footprint settings
      logs/
        settings=hrrr-a3f9c2/date=2024-07-01/<receptor>.log

A folder is found by the hash of its settings, so two projects that run the
same settings share one folder, and a changed setting lands in a new folder
beside the old one instead of overwriting it. ``pyarrow.dataset``, DuckDB,
polars, and R's arrow read ``settings`` and ``date`` as columns of the whole
tree. Particles are one Parquet file per receptor. Footprints are sparse
tables of the non-zero cells, in float32 as STILT-R writes them; an empty
footprint is a file with no rows and its reason in the metadata.

Start from :class:`Output` and a resolved variant (``project.variants``).
Every function takes the kind of result, ``"particles"`` or
``"footprints"``, and the variant::

    out = Output("output")
    out.write_particles(variant, receptor, frame, met_files)
    out.path("particles", variant, receptor.id)  # one receptor's file
    out.present("footprints", variant, receptor_ids)  # which have a footprint
    out.table("footprints", variant)  # many footprints as one table
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import xarray as xr
import yaml

from stilt._atomic import atomic_path
from stilt._paths import location
from stilt.footprint import (
    FOOTPRINT_SCHEMA,
    FootprintConfig,
    write_empty_footprint,
    write_footprint,
)
from stilt.identity import (
    footprint_hash,
    footprint_settings,
    read_footprint_settings,
    read_run_settings,
    settings_hash,
)
from stilt.particles import write_particles
from stilt.receptors import Receptor, parse_receptor_id

if TYPE_CHECKING:
    from upath import UPath

    from stilt._paths import Location
    from stilt.config import Variant

logger = logging.getLogger(__name__)

#: Underscore-prefixed, so dataset readers skip it.
SETTINGS_FILE = "_settings.yaml"
#: A receptor's failure record, under its folder's logs: ``<receptor_id>.failure.yaml``.
FAILURE_SUFFIX = ".failure.yaml"
HASH_CHARS = 6

# -- identity -----------------------------------------------------------------


def _date_dir(receptor_id: str) -> str:
    """Return the ``date=YYYY-MM-DD`` folder of a receptor id, which starts with the receptor time."""
    time, _ = parse_receptor_id(receptor_id)
    return f"date={time:%Y-%m-%d}"


def _names(folder: Location) -> list[str]:
    """
    Return the names in *folder*, or none when it does not exist.

    A filesystem is listed with ``os.scandir``, the fast path. An object
    store is listed through its fsspec filesystem, with its cache of
    listings cleared first, since workers elsewhere add files to it.
    """
    try:
        if isinstance(folder, Path):
            return [entry.name for entry in os.scandir(folder)]
        folder.fs.invalidate_cache(folder.path)
        listed = folder.fs.ls(folder.path, detail=False)
    except FileNotFoundError:
        return []
    return [str(item).rstrip("/").rsplit("/", 1)[-1] for item in listed]


def _list_receptor_files(
    root: Location, suffix: str, among: Iterable[str] | None = None
) -> dict[str, Location]:
    """
    Return ``{receptor_id: path}`` for every ``date=*/<id><suffix>`` under *root*, in date order.

    With *among*, only those receptors are returned, and only their date
    folders are listed. The date folders are listed in a few threads: on a
    network filesystem or an object store the listing waits on the server,
    and a project can have thousands of date folders.
    """
    if among is None:
        wanted = None
        days = sorted(name for name in _names(root) if name.startswith("date="))
    else:
        wanted = set(among)
        days = sorted({_date_dir(r) for r in wanted})

    def names(day: str) -> list[str]:
        return sorted(name for name in _names(root / day) if name.endswith(suffix))

    with ThreadPoolExecutor(max_workers=16) as pool:
        listed = list(pool.map(names, days))
    found = {
        name[: -len(suffix)]: folder / name
        for folder, per_day in zip((root / d for d in days), listed, strict=True)
        for name in per_day
    }
    if wanted is not None:
        found = {rid: path for rid, path in found.items() if rid in wanted}
    return found


def _read_yaml(path: Location) -> dict[str, Any]:
    """Return a YAML file of one mapping as a dict; an empty file is ``{}``."""
    return yaml.safe_load(path.read_text()) or {}


def _write_yaml(path: Location, record: dict[str, Any]) -> None:
    """Write *record* to *path* as YAML in one step, making its folder as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as tmp:
        tmp.write_text(
            yaml.safe_dump(record, default_flow_style=False, sort_keys=False)
        )


def _write_settings(path: Location, record: dict[str, Any]) -> None:
    """Write a ``settings.yaml`` once. An existing file with the same hash is left alone."""
    if path.exists():
        existing = _read_yaml(path)
        if existing.get("hash") != record["hash"]:
            raise FileExistsError(
                f"{path} holds settings with hash {existing.get('hash')}, "
                f"not {record['hash']}."
            )
        return
    # Another worker may write the same settings meanwhile; both files say
    # the same thing, so whichever rename lands last is fine.
    _write_yaml(path, record)


def _pystilt_version() -> str:
    """Return the installed PYSTILT version, as ``stilt.__version__`` does."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("pystilt")
    except PackageNotFoundError:
        return "0+unknown"


def _stamp(digest: str, realization: int | None = None) -> dict[bytes, bytes]:
    """
    Return what every result file records besides its contents.

    Its folder's settings hash, the PYSTILT version, and, for an ensemble,
    its realization number.
    """
    stamp = {
        b"stilt:hash": digest.encode(),
        b"stilt:pystilt": _pystilt_version().encode(),
    }
    if realization is not None:
        stamp[b"stilt:realization"] = str(realization).encode()
    return stamp


# -- the output directory -----------------------------------------------------

#: A kind of result, and the tree it is kept in.
Kind = Literal["particles", "footprints"]
KINDS: tuple[Kind, ...] = ("particles", "footprints")

#: What :meth:`Output.table` returns when no file is read, by kind.
_EMPTY: dict[str, pa.Table] = {
    "particles": pa.table({"receptor": pa.array([], pa.string())}),
    "footprints": FOOTPRINT_SCHEMA.append(pa.field("date", pa.date32())).empty_table(),
}

#: The ``date=YYYY-MM-DD`` folders, read as a ``date32`` column.
_DATE_PARTITIONING = pads.partitioning(
    pa.schema([("date", pa.date32())]), flavor="hive"
)


def _settings_folders(tree: Location) -> list[str]:
    """Return the ``settings=`` values under *tree* that hold a settings file, in name order."""
    return [
        name[len("settings=") :]
        for name in sorted(_names(tree))
        if name.startswith("settings=") and (tree / name / SETTINGS_FILE).exists()
    ]


def _receptor_file(
    folder: Location, receptor_id: str, suffix: str = ".parquet"
) -> Location:
    """Return a receptor's file in *folder*: ``date=YYYY-MM-DD/<receptor_id><suffix>``."""
    return folder / _date_dir(receptor_id) / f"{receptor_id}{suffix}"


def completed(
    particles: frozenset[str], footprints: frozenset[str] | None
) -> frozenset[str]:
    """
    Return the receptors that are complete, from the ones with particles and with a footprint.

    The one definition of done: the particles exist, and the footprint
    too when the variant makes one (*footprints* is ``None`` when it does
    not). An empty footprint is complete.
    """
    return particles if footprints is None else particles & footprints


def _check_realization(variant: Variant, realization: int | None) -> None:
    """Raise unless *realization* is one *variant* runs as."""
    if realization not in variant.realization_numbers:
        raise ValueError(
            f"Variant {variant.name!r} runs as realizations "
            f"{variant.realization_numbers}, not {realization!r}."
        )


def _check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"kind is 'particles' or 'footprints', not {kind!r}.")


class Output:
    """
    An output directory: particles and footprints, by their settings.

    Each kind of result has a tree, and in it a folder per set of settings,
    named ``settings=<variant>-<hash>``. A variant's folder is found by the
    hash of its settings, whatever name it was created under, so projects
    that run the same settings share it. Every method takes the kind of
    result (``"particles"`` or ``"footprints"``) and the variant, resolved
    (``project.variants``).

    Parameters
    ----------
    path : str or Path
        The directory, created on first write. A URL such as
        ``s3://bucket/output`` puts it on an object store, read and written
        through fsspec (install its package, such as ``s3fs``).
    """

    def __init__(self, path: str | Path | UPath) -> None:
        #: The output directory: a Path, or a universal path on an object store.
        self.directory: Location = location(path)
        # Each folder's settings hash, by kind and folder name. Reading one
        # means reading its _settings.yaml and hashing it again, so each is
        # read once; a lookup that misses lists the tree again.
        self._hashes: dict[str, dict[str, str]] = {"particles": {}, "footprints": {}}

    def __repr__(self) -> str:
        return f"Output({str(self.directory)!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Output) and other.directory == self.directory

    def __hash__(self) -> int:
        return hash(str(self.directory))

    # -- folders -----------------------------------------------------------

    def folders(self, kind: Kind) -> dict[str, str]:
        """
        Return every folder of *kind*, as ``{name: settings hash}`` in name order.

        The name is the ``settings=`` value. The hash is of the folder's
        stored settings read back through the current config classes
        (:mod:`stilt.identity`), so a setting added since the folder was
        written, with a default, still matches.
        """
        _check_kind(kind)
        known = self._hashes[kind]
        names = _settings_folders(self.directory / kind)
        for name in names:
            if name not in known:
                known[name] = self._read_hash(kind, name)
        return {name: known[name] for name in names}

    def _read_hash(self, kind: Kind, name: str) -> str:
        """Return the settings hash of one folder, from its ``_settings.yaml``."""
        path = self._dir(kind, name) / SETTINGS_FILE
        record = _read_yaml(path)
        if kind == "particles":
            return settings_hash(read_run_settings(record["settings"]))
        config, geometry_hash = read_footprint_settings(record["settings"], str(path))
        particles = record["particles"]
        if particles not in self._hashes["particles"]:
            self._hashes["particles"][particles] = self._read_hash(
                "particles", particles
            )
        return footprint_hash(
            self._hashes["particles"][particles],
            footprint_settings(config, geometry_hash),
        )

    @staticmethod
    def _variant_hash(kind: Kind, variant: Variant) -> str | None:
        """Return the hash that names *variant*'s folder of *kind*, ``None`` for footprints without a grid."""
        _check_kind(kind)
        return variant.particles_hash if kind == "particles" else variant.footprint_hash

    def _name(self, kind: Kind, variant: Variant) -> str | None:
        """Return the name of *variant*'s folder of *kind*, or ``None`` when it does not exist."""
        digest = self._variant_hash(kind, variant)
        if digest is None:
            return None
        for name, folder_hash in self._hashes[kind].items():
            if folder_hash == digest:
                return name
        # A miss lists the tree again: another worker may have made the folder.
        for name, folder_hash in self.folders(kind).items():
            if folder_hash == digest:
                return name
        return None

    def folder(self, kind: Kind, variant: Variant) -> Location | None:
        """
        Return *variant*'s folder of *kind*, or ``None`` before it exists.

        ``None`` too for the footprints of a variant without a grid.
        """
        name = self._name(kind, variant)
        return None if name is None else self._dir(kind, name)

    def _dir(self, tree: str, name: str) -> Location:
        """Return the folder *name* of a tree: ``<tree>/settings=<name>``."""
        return self.directory / tree / f"settings={name}"

    def _create(self, kind: Kind, variant: Variant) -> str:
        """Return the name of *variant*'s folder of *kind*, writing its settings file on first use."""
        name = self._name(kind, variant)
        if name is not None:
            return name
        if kind == "particles":
            digest = variant.particles_hash
            record: dict[str, Any] = {
                "name": variant.name,
                "hash": digest,
                "pystilt": _pystilt_version(),
                "settings": variant.run_settings,
            }
        else:
            if variant.footprint is None:
                raise ValueError(
                    f"Variant {variant.name!r} makes no footprints (no grid)."
                )
            particles = self._create("particles", variant)
            settings = footprint_settings(variant.footprint, variant.geometry_hash)
            digest = footprint_hash(variant.particles_hash, settings)
            record = {
                "name": variant.name,
                "hash": digest,
                "particles": particles,
                "particles_hash": variant.particles_hash,
                "pystilt": _pystilt_version(),
                "settings": settings,
            }
        name = f"{variant.name}-{digest[:HASH_CHARS]}"
        _write_settings(self._dir(kind, name) / SETTINGS_FILE, record)
        self._hashes[kind][name] = digest
        return name

    def _part(
        self, tree: str, name: str, variant: Variant, realization: int | None
    ) -> Location:
        """
        Return the folder *name* of a tree, inside its ``realization=k`` partition for an ensemble.

        Raises
        ------
        ValueError
            If *realization* is not one the variant runs as.
        """
        _check_realization(variant, realization)
        folder = self._dir(tree, name)
        return folder if realization is None else folder / f"realization={realization}"

    def _found(
        self, kind: Kind, tree: str, variant: Variant, realization: int | None
    ) -> Location | None:
        """Return the existing folder of *variant*'s results of *kind* in *tree*, or ``None``."""
        _check_realization(variant, realization)
        name = self._name(kind, variant)
        return None if name is None else self._part(tree, name, variant, realization)

    def _made(
        self, kind: Kind, tree: str, variant: Variant, realization: int | None
    ) -> Location:
        """Return the folder of *variant*'s results of *kind* in *tree*, making the settings folder on first use."""
        return self._part(tree, self._create(kind, variant), variant, realization)

    # -- one receptor's files ------------------------------------------------

    def path(
        self,
        kind: Kind,
        variant: Variant,
        receptor_id: str,
        realization: int | None = None,
    ) -> Location | None:
        """
        Return the file of one receptor's result, whether or not it exists.

        ``None`` before the variant's folder of *kind* exists. An ensemble's
        results are in a ``realization=k`` partition of its folder.
        """
        folder = self._found(kind, kind, variant, realization)
        return None if folder is None else _receptor_file(folder, receptor_id)

    def log_path(
        self, variant: Variant, receptor_id: str, realization: int | None = None
    ) -> Location | None:
        """Return where the log of a receptor's transport model run is kept, or ``None`` before its folder exists."""
        logs = self._found("particles", "logs", variant, realization)
        return None if logs is None else _receptor_file(logs, receptor_id, ".log")

    def kept_workdir(
        self, variant: Variant, receptor_id: str, realization: int | None = None
    ) -> Location | None:
        """Return where a receptor's failed run's working directory is kept, or ``None`` before its folder exists."""
        kept = self._found("particles", "scratch", variant, realization)
        return None if kept is None else kept / _date_dir(receptor_id) / receptor_id

    # -- many receptors' files ----------------------------------------------

    def present(
        self,
        kind: Kind,
        variant: Variant,
        receptor_ids: Iterable[str] | None = None,
        realization: int | None = None,
    ) -> frozenset[str]:
        """
        Return the receptors that have a file of *kind* for *variant*.

        Only the date folders of *receptor_ids* are listed; without them,
        every date folder is. No file is opened.
        """
        folder = self._found(kind, kind, variant, realization)
        if folder is None:
            return frozenset()
        return frozenset(_list_receptor_files(folder, ".parquet", receptor_ids))

    def complete(
        self,
        variant: Variant,
        receptor_ids: Iterable[str] | None = None,
        realization: int | None = None,
    ) -> frozenset[str]:
        """
        Return the receptors whose simulations under *variant* are complete (:func:`completed`).

        Only the receptors with particles are looked for in the footprints.
        """
        particles = self.present("particles", variant, receptor_ids, realization)
        if variant.footprint is None:
            return completed(particles, None)
        footprints = self.present("footprints", variant, particles, realization)
        return completed(particles, footprints)

    def table(
        self,
        kind: Kind,
        variant: Variant,
        receptor_ids: Iterable[str] | None = None,
        realization: int | None = None,
    ) -> pa.Table:
        """
        Return the files of many receptors as one table.

        Columns are ``receptor``, the stored columns, and ``date`` (the
        receptor date, from the folder, as ``date32``).

        Parameters
        ----------
        kind : {"particles", "footprints"}
            Which results.
        variant : Variant
            Whose results.
        receptor_ids : iterable of str, optional
            The receptors to read, each of which must have a file
            (:meth:`present`). Without it, every file in the folder is read.
        realization : int, optional
            Which realization of an ensemble.
        """
        _check_kind(kind)
        folder = self._found(kind, kind, variant, realization)
        if folder is None:
            return _EMPTY[kind]
        if receptor_ids is None:
            paths = list(_list_receptor_files(folder, ".parquet").values())
        else:
            paths = [_receptor_file(folder, r) for r in dict.fromkeys(receptor_ids)]
        if not paths:
            return _EMPTY[kind]
        if isinstance(folder, Path):
            dataset = pads.dataset(
                [str(p) for p in paths],
                format="parquet",
                partitioning=_DATE_PARTITIONING,
                partition_base_dir=str(folder),
            )
        else:  # an object store, read through its fsspec filesystem
            dataset = pads.dataset(
                [p.path for p in paths if not isinstance(p, Path)],
                filesystem=folder.fs,
                format="parquet",
                partitioning=_DATE_PARTITIONING,
                partition_base_dir=folder.path,
            )
        return dataset.to_table()

    # -- failure records ---------------------------------------------------

    def failure(
        self,
        kind: Kind,
        variant: Variant,
        receptor_id: str,
        realization: int | None = None,
    ) -> dict[str, Any] | None:
        """
        Return why a receptor's result of *kind* failed, as the worker recorded it, or ``None``.

        A record holds ``step``, ``reason``, ``message``, and ``time``, and a
        ``traceback`` for an unexpected error. It is removed when the result
        is written.
        """
        logs = self._found(kind, "logs", variant, realization)
        if logs is None:
            return None
        path = _receptor_file(logs, receptor_id, FAILURE_SUFFIX)
        return _read_yaml(path) if path.exists() else None

    def failures(
        self,
        kind: Kind,
        variant: Variant,
        receptor_ids: Iterable[str] | None = None,
        realization: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Return ``{receptor_id: record}`` for the receptors with a failure record of *kind*.

        With *receptor_ids*, only their date folders are listed. The records
        found are read in a few threads.
        """
        logs = self._found(kind, "logs", variant, realization)
        if logs is None:
            return {}
        paths = _list_receptor_files(logs, FAILURE_SUFFIX, receptor_ids)
        with ThreadPoolExecutor(max_workers=16) as pool:
            records = list(pool.map(_read_yaml, paths.values()))
        return dict(zip(paths, records, strict=True))

    def record_failure(
        self,
        kind: Kind,
        variant: Variant,
        receptor_id: str,
        record: dict[str, Any],
        realization: int | None = None,
    ) -> None:
        """
        Write why a receptor's result of *kind* failed, replacing an earlier record.

        It goes in the logs of the folder whose result failed: a particles
        failure covers every variant on those particles, a footprint failure
        is the variant's own.
        """
        logs = self._made(kind, "logs", variant, realization)
        _write_yaml(_receptor_file(logs, receptor_id, FAILURE_SUFFIX), record)

    def clear_failure(
        self,
        kind: Kind,
        variant: Variant,
        receptor_id: str,
        realization: int | None = None,
    ) -> None:
        """Remove a receptor's failure record of *kind*, once its result is written."""
        logs = self._found(kind, "logs", variant, realization)
        if logs is not None:
            _receptor_file(logs, receptor_id, FAILURE_SUFFIX).unlink(missing_ok=True)

    # -- writing -----------------------------------------------------------

    def write_particles(
        self,
        variant: Variant,
        receptor: Receptor,
        particles: pd.DataFrame,
        met_files: list[Path],
        realization: int | None = None,
    ) -> Location:
        """
        Write a receptor's particles (:func:`stilt.particles.write_particles`).

        The file records the run's settings, so it reads alone, and the
        folder's settings hash, the PYSTILT version, and the realization.
        """
        folder = self._made("particles", "particles", variant, realization)
        return write_particles(
            _receptor_file(folder, str(receptor.id)),
            particles,
            receptor,
            variant.run_settings,
            met_files,
            metadata=_stamp(variant.particles_hash, realization),
        )

    def write_log(
        self,
        variant: Variant,
        receptor_id: str,
        text: str,
        realization: int | None = None,
    ) -> Location:
        """Write the log of a receptor's transport model run."""
        logs = self._made("particles", "logs", variant, realization)
        path = _receptor_file(logs, receptor_id, ".log")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def keep_workdir(
        self,
        variant: Variant,
        receptor_id: str,
        workdir: Path,
        realization: int | None = None,
    ) -> Location:
        """
        Copy a run's working directory to :meth:`kept_workdir`, replacing an earlier copy.

        A failed run's is kept, so CONTROL, SETUP.CFG, and the model's own
        output can be read after the job ends.
        """
        scratch = self._made("particles", "scratch", variant, realization)
        kept = scratch / _date_dir(receptor_id) / receptor_id
        if isinstance(kept, Path):
            shutil.rmtree(kept, ignore_errors=True)
            kept.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(workdir, kept, symlinks=True)
            return kept
        # An object store keeps no links, so the met files the working
        # directory links to are left out.
        if kept.exists():
            kept.fs.rm(kept.path, recursive=True)
        for file in sorted(workdir.rglob("*")):
            if file.is_file() and not file.is_symlink():
                (kept / file.relative_to(workdir).as_posix()).write_bytes(
                    file.read_bytes()
                )
        return kept

    def write_footprint(
        self, variant: Variant, foot: xr.DataArray, realization: int | None = None
    ) -> Location:
        """
        Write one receptor's footprint (:func:`stilt.footprint.write_footprint`).

        The footprint must be on the variant's grid. The file records the
        footprint settings and the folder's hash, so it reads alone.
        """
        path, config, digest = self._footprint_file(
            variant, str(foot.stilt.receptor.id), realization
        )
        return write_footprint(
            path,
            foot,
            config,
            _stamp(digest, realization),
            geometry_hash=variant.geometry_hash,
        )

    def write_empty_footprint(
        self,
        variant: Variant,
        receptor: Receptor,
        reason: str,
        realization: int | None = None,
    ) -> Location:
        """Record that a receptor's footprint is empty (no particle over the grid), with the reason."""
        path, config, digest = self._footprint_file(
            variant, str(receptor.id), realization
        )
        return write_empty_footprint(
            path,
            receptor,
            reason,
            config,
            variant.name,
            _stamp(digest, realization),
            geometry_hash=variant.geometry_hash,
        )

    def _footprint_file(
        self, variant: Variant, receptor_id: str, realization: int | None
    ) -> tuple[Location, FootprintConfig, str]:
        """Return a receptor's footprint file, the settings, and the folder's hash, making the folder on first use."""
        name = self._create("footprints", variant)
        if variant.footprint is None:  # _create has raised already
            raise ValueError(f"Variant {variant.name!r} makes no footprints.")
        folder = self._part("footprints", name, variant, realization)
        path = _receptor_file(folder, receptor_id)
        return path, variant.footprint, self._hashes["footprints"][name]


__all__ = ["KINDS", "Kind", "Output", "completed"]
