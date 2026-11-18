"""
The output directory, where particles and the files of each stage are kept.

A project's results live in an output directory that several projects can
share. It has one tree per kind of result, ``particles`` and one named for
each stage (the footprint's is ``footprint``), and inside each a folder per
set of settings, in hive form so every tree reads as one dataset::

    <output>/
      particles/
        settings=hrrr-a3f9c2/            variant name + short hash of the transport settings
          _settings.yaml
          date=2024-07-01/<release>.parquet
        settings=hrrr-err-7be104/        an ensemble: one folder, a partition per realization
          realization=0/date=2024-07-01/<release>.parquet
      footprint/
        settings=hrrr-93278c/            variant name + short hash of transport + footprint settings
          _settings.yaml                 names the particles folder it was made from
          date=2024-07-01/<release>.parquet
        settings=hrrr-hexes-1b2c3d/      same particles, other footprint settings
      logs/
        settings=hrrr-a3f9c2/date=2024-07-01/<release>.log
        settings=hrrr-a3f9c2/date=2024-07-01/<release>.failure.yaml   a failed run
        settings=hrrr-93278c/date=2024-07-01/<release>.failure.yaml   a failed footprint
      scratch/
        settings=hrrr-a3f9c2/date=2024-07-01/<release>/               a kept workdir

A folder is found by the hash of its settings, so two projects that run the
same settings share one folder, and a changed setting lands in a new folder
beside the old one instead of overwriting it. ``pyarrow.dataset``, DuckDB,
polars, and R's arrow read ``settings`` and ``date`` as columns of the whole
tree. Particles are one Parquet file per release. A stage writes its own
file per release, with the extension it declares
(:attr:`stilt.stages.Stage.suffix`), at the path this module gives it, and
this module never looks inside one. The footprint's are sparse tables of
the non-zero cells; an empty footprint is a file with no rows, marked
``stilt:empty`` in its metadata.

Logs, failure records, and kept workdirs are in trees of their own
(``logs/``, ``scratch/``) rather than beside the results, so that each
results tree holds result files only and reads as one dataset. Whether a
simulation is complete is read from the results trees alone
(:func:`completed`); the logs only explain the ones that are not.

Start from :class:`Output` and a resolved variant (``project.variants``).
Every function takes the kind of result, ``"particles"`` or a stage's name,
and the variant::

    out = Output("output")
    out.write_particles(variant, release, frame, met_files)
    out.path("particles", variant, release.id)  # one release's file
    out.present("footprint", variant, release_ids)  # which have a footprint
    out.make_path("footprint", variant, release.id)  # where a stage writes
"""

from __future__ import annotations

import datetime as dt
import functools
import logging
import os
import shutil
import warnings
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import yaml

from stilt._paths import atomic_path, location, readable
from stilt.identity import (
    canonical,
    file_stamp,
    pystilt_version,
    read_run_settings,
    settings_hash,
    stage_hash,
)
from stilt.particles import write_particles
from stilt.releases import Receptor, parse_release_id
from stilt.stages import get_stage, reads

if TYPE_CHECKING:
    from upath import UPath

    from stilt._paths import Location
    from stilt.config import Variant

logger = logging.getLogger(__name__)

#: Underscore-prefixed, so dataset readers skip it.
SETTINGS_FILE = "_settings.yaml"
#: The suffix of a release's failure record, ``<release_id>.failure.yaml``, in its folder's logs.
FAILURE_SUFFIX = ".failure.yaml"
HASH_CHARS = 6

# -- identity -----------------------------------------------------------------


@functools.cache
def _day_folder(day: str) -> str:
    """Return the ``date=YYYY-MM-DD`` folder of a ``YYYYMMDD`` day, which must be a real date."""
    return f"date={dt.datetime.strptime(day, '%Y%m%d'):%Y-%m-%d}"


def _date_dir(release_id: str) -> str:
    """
    Return the ``date=YYYY-MM-DD`` folder of a release id, which starts with the release time.

    The folder is the id's first eight characters, ``YYYYMMDD``, so a listing
    of hundreds of thousands of ids reads each day once and does not parse
    each id's time. An id that :func:`stilt.releases.parse_release_id`
    rejects raises its error.
    """
    rid = str(release_id)
    stamp = rid[:12]  # YYYYMMDDHHMM
    if (
        len(rid) > 13
        and rid[12] == "_"
        and stamp.isascii()
        and stamp.isdigit()
        and stamp[8:10] < "24"
        and stamp[10:12] < "60"
    ):
        try:
            return _day_folder(stamp[:8])
        except ValueError:
            pass  # not a real date; parse_release_id says so
    time, _ = parse_release_id(rid)
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
    except (FileNotFoundError, NotADirectoryError):
        return []
    return [str(item).rstrip("/").rsplit("/", 1)[-1] for item in listed]


def _list_release_files(
    root: Location, suffix: str, among: Iterable[str] | None = None
) -> dict[str, Location]:
    """
    Return ``{release_id: path}`` for every ``date=*/<id><suffix>`` under *root*, in date order.

    With *among*, only those releases are returned, and only their date
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
        # A name that starts with a dot is a file being written.
        return sorted(
            name
            for name in _names(root / day)
            if name.endswith(suffix) and not name.startswith(".")
        )

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


@functools.cache
def _versions(stage_class: str) -> dict[str, str]:
    """
    Return the versions a stage folder records: PYSTILT's, and the package of the stage's class.

    They say what made the folder and are not part of its hash.
    """
    from importlib.metadata import PackageNotFoundError, packages_distributions, version

    versions = {"pystilt": pystilt_version()}
    package = stage_class.partition(".")[0]
    for distribution in packages_distributions().get(package, ()):
        try:
            versions.setdefault(distribution, version(distribution))
        except PackageNotFoundError:
            continue
    return versions


def _read_stage_settings(
    stage_class: str, stored: Mapping[str, Any], source: str
) -> dict[str, Any]:
    """
    Return the settings a stage folder's ``_settings.yaml`` records, read through the stage's current config class.

    The class reads the record and the stage writes it again
    (:meth:`stilt.stages.Stage.settings`), so a setting added since the
    folder was written, with a default, still matches. A record that cannot
    be read here is returned as written, with a warning naming *source*: a
    stage that is not installed, or settings its class no longer takes, as
    in another project's folder of a shared output directory.
    """
    try:
        stage = get_stage(stage_class)
        return canonical(stage.settings(stage.config_class.model_validate(stored)))
    except Exception as error:
        warnings.warn(
            f"{source}: the settings of a stored {stage_class} could not be "
            f"read ({error}). They are kept as written.",
            stacklevel=2,
        )
        return canonical(stored)


# -- the output directory -----------------------------------------------------

#: A kind of result, and the tree it is kept in: ``particles``, or the name of a stage.
Kind = str

#: The trees of an output directory that hold no results.
_OTHER_TREES = ("logs", "scratch")

#: What :meth:`Output.table` returns when no file is read: for particles, and for a stage.
_NO_PARTICLES = pa.table({"release": pa.array([], pa.string())})
_NO_STAGE_FILES = pa.table(
    {"release": pa.array([], pa.string()), "date": pa.array([], pa.date32())}
)

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


def _release_file(
    folder: Location, release_id: str, suffix: str = ".parquet"
) -> Location:
    """Return a release's file in *folder*: ``date=YYYY-MM-DD/<release_id><suffix>``."""
    return folder / _date_dir(release_id) / f"{release_id}{suffix}"


def completed(particles: frozenset[str], *stages: frozenset[str]) -> frozenset[str]:
    """
    Return the releases that are complete, from the ones with particles and the ones with each stage's file.

    The one definition of done. A simulation is complete when the model's
    output file exists, and so does the file of every stage its variant
    uses. *stages* is one set of releases for each stage the variant uses,
    none for a variant that makes particles only. A stage's file is a file
    whatever it holds, so an empty footprint is complete.
    """
    return particles.intersection(*stages)


def _count_files(folder: Location) -> int:
    """Return how many result files a settings folder holds, in every ``realization=k`` partition."""
    parts = [n for n in _names(folder) if n.startswith("realization=")]
    roots = [folder / n for n in parts] if parts else [folder]
    days = [
        root / name
        for root in roots
        for name in _names(root)
        if name.startswith("date=")
    ]
    with ThreadPoolExecutor(max_workers=16) as pool:
        listed = list(pool.map(_names, days))
    # A file being written has a name that starts with a dot.
    return sum(not name.startswith(".") for names in listed for name in names)


def _short(value: Any) -> str:
    """Return a setting's value in a few characters."""
    text = str(value)
    return text if len(text) <= 40 else text[:37] + "..."


def _diff(
    stored: Mapping[str, Any], current: Mapping[str, Any], prefix: str = ""
) -> list[str]:
    """Return ``key: stored (config: current)`` for each setting that differs, nested blocks by dotted key."""
    found = []
    for key in sorted(set(stored) | set(current)):
        a, b = stored.get(key, "-"), current.get(key, "-")
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            found += _diff(a, b, f"{prefix}{key}.")
        elif a != b:
            found.append(f"{prefix}{key}: {_short(a)} (config: {_short(b)})")
    return found


def _differences(
    kind: Kind, record: Mapping[str, Any], variant: Variant, path: Location
) -> str:
    """Return how a folder's stored settings differ from *variant*'s, the first four, ``; ``-joined."""
    if kind == "particles":
        # What can be read here: the met's files or the build may be missing.
        readable = variant.readable_run_settings
        stored = _like(read_run_settings(record["settings"]), readable)
        current: Mapping[str, Any] = readable
    else:
        if kind not in variant.stage_settings:
            return f"the variant has no {kind} stage"
        current = variant.stage_settings[kind]
        stored = _read_stage_settings(
            str(record.get("class", "")), record["settings"], str(path)
        )
    found = _diff(stored, current)
    if kind != "particles" and not found:
        return "made from other particles"
    if len(found) > 4:
        found = [*found[:4], f"and {len(found) - 4} more"]
    return "; ".join(found)


def _like(stored: Mapping[str, Any], readable: Mapping[str, Any]) -> dict[str, Any]:
    """
    Return *stored* run settings with only the parts *readable* has.

    Without ``met`` when the met's files cannot be read here, and with only
    the model's name when its build cannot (:attr:`Variant.readable_run_settings`).
    """
    like = {k: v for k, v in stored.items() if k != "met" or "met" in readable}
    like["model"] = {k: v for k, v in stored["model"].items() if k in readable["model"]}
    return like


def _check_realization(variant: Variant, realization: int | None) -> None:
    """Raise unless *realization* is one *variant* runs as."""
    if realization not in variant.realization_numbers:
        raise ValueError(
            f"Variant {variant.name!r} runs as realizations "
            f"{variant.realization_numbers}, not {realization!r}."
        )


def _check_kind(kind: str) -> None:
    """Raise for a kind that names one of the trees that hold no results."""
    if kind in _OTHER_TREES:
        raise ValueError(f"kind is 'particles' or a stage's name, not {kind!r}.")


class Output:
    """
    An output directory: particles and the files of each stage, by their settings.

    Each kind of result has a tree, and in it a folder per set of settings,
    named ``settings=<variant>-<hash>``. A variant's folder is found by the
    hash of its settings, whatever name it was created under, so projects
    that run the same settings share it. Every method takes the kind of
    result (``"particles"``, or the name of a stage such as
    ``"footprint"``) and the variant, resolved (``project.variants``).

    Parameters
    ----------
    path : str or Path
        The directory, created on first write. A URL such as
        ``s3://bucket/output`` puts it on an object store, read and written
        through fsspec (install its package, such as ``s3fs``).
    """

    def __init__(self, path: str | Path | UPath) -> None:
        #: The output directory, a Path or a universal path on an object store.
        self.directory: Location = location(path)
        # Each folder's settings hash, by kind and folder name. Reading one
        # means reading its _settings.yaml and hashing it again, so each is
        # read once; a lookup that misses lists the tree again.
        self._hashes: dict[str, dict[str, str]] = {}

    def __repr__(self) -> str:
        return f"Output({str(self.directory)!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Output) and other.directory == self.directory

    def __hash__(self) -> int:
        return hash(str(self.directory))

    # -- folders -----------------------------------------------------------

    def hashes(self, kind: Kind) -> dict[str, str]:
        """
        Return every folder of *kind*, as ``{name: settings hash}`` in name order.

        The name is the ``settings=`` value. The hash is of the folder's
        stored settings read back through the current config classes
        (:mod:`stilt.identity`), so a setting added since the folder was
        written, with a default, still matches.
        """
        _check_kind(kind)
        names = _settings_folders(self.directory / kind)
        return {name: self._folder_hash(kind, name) for name in names}

    def kinds(self) -> list[Kind]:
        """
        Return the kinds of result the directory holds a tree for: ``particles``, then the others in name order.

        Every tree but ``logs`` and ``scratch`` is a kind, whether or not a
        stage of that name is installed here.
        """
        trees = sorted(
            name
            for name in _names(self.directory)
            if name not in ("particles", *_OTHER_TREES)
            and not name.startswith((".", "_"))
        )
        return ["particles", *trees]

    def folders(self, variants: Mapping[str, Variant] | None = None) -> pd.DataFrame:
        """
        Return every settings folder in the output directory, one row each.

        A settings folder holds the results of one set of settings, named
        after the variant that first made them and a short hash of the
        settings (``settings=hrrr-b2399e``).

        Parameters
        ----------
        variants : mapping of str to Variant, optional
            A project's variants (``project.variants``), to say which use
            each folder and how the others differ from them.

        Returns
        -------
        pandas.DataFrame
            ``kind`` (``particles`` or a stage's name), ``folder`` (the
            ``settings=`` value), ``name`` (the variant that made it), and
            ``files`` (the result files it holds, every realization of an
            ensemble counted). With *variants*, also ``variant``, the
            variants that use the folder (empty when none does), and
            ``differs``, how an unused folder's settings differ from those
            of the variant of its name.

        Examples
        --------
        >>> project.output.folders(project.variants)  # doctest: +SKIP
        """
        rows = []
        for kind in self.kinds():
            for folder, digest in self.hashes(kind).items():
                path = self._dir(kind, folder)
                record = _read_yaml(path / SETTINGS_FILE)
                row: dict[str, Any] = {
                    "kind": kind,
                    "folder": folder,
                    "name": record.get("name", ""),
                    "files": _count_files(path),
                }
                if variants is not None:
                    using = [
                        name
                        for name, v in variants.items()
                        if self._lookup_hash(kind, v) == digest
                    ]
                    row["variant"] = ", ".join(using)
                    same = variants.get(row["name"])
                    row["differs"] = (
                        ""
                        if using or same is None
                        else _differences(kind, record, same, path)
                    )
                rows.append(row)
        columns = ["kind", "folder", "name", "files"]
        if variants is not None:
            columns += ["variant", "differs"]
        return pd.DataFrame(rows, columns=columns)

    def _folder_hash(self, kind: Kind, name: str) -> str:
        """Return the settings hash of the folder *name* of *kind*, read once."""
        known = self._hashes.setdefault(kind, {})
        if name not in known:
            known[name] = self._read_hash(kind, name)
        return known[name]

    def _read_hash(self, kind: Kind, name: str) -> str:
        """
        Return the settings hash of one folder, from its ``_settings.yaml``.

        A stage's folder is hashed with the hash of each folder its record
        says it was made from, read the same way. A stage folder whose
        record names no stage class was not written by this layout, and
        keeps the hash it recorded.
        """
        path = self._dir(kind, name) / SETTINGS_FILE
        record = _read_yaml(path)
        if kind == "particles":
            return settings_hash(read_run_settings(record["settings"]))
        made_from = record.get("reads")
        if (
            "class" not in record
            or not isinstance(made_from, Mapping)
            or "particles" not in made_from
        ):
            return str(record.get("hash", ""))
        settings = _read_stage_settings(record["class"], record["settings"], str(path))
        hashes = {}
        for what, folder in made_from.items():
            try:
                hashes[what] = self._folder_hash(what, folder["folder"])
            except FileNotFoundError:
                hashes[what] = folder["hash"]  # its folder is gone: as recorded
        return stage_hash(kind, settings, hashes.pop("particles"), hashes)

    @staticmethod
    def _variant_hash(kind: Kind, variant: Variant) -> str | None:
        """Return the hash that names *variant*'s folder of *kind*, ``None`` for a stage the variant does not use."""
        _check_kind(kind)
        if kind == "particles":
            return variant.particles_hash
        if kind not in variant.stages:
            return None  # without reading the met or the build for the particles' hash
        return variant.stage_hashes[kind]

    def _lookup_hash(self, kind: Kind, variant: Variant) -> str | None:
        """
        Return the hash that finds *variant*'s folder of *kind*, or ``None`` when no folder can be its.

        ``None`` for a stage the variant does not use. When the settings
        cannot all be read here, the particles folders are compared on what
        can be (:meth:`_readable_matches`):

        - The met's files cannot be read: ``None`` when no folder has the
          variant's other settings, since then no folder has its results.
        - Only the build cannot be read, as on a machine without
          ``hycs_std`` that reports status: the folder this variant made
          (its ``name``) with its other settings, or ``None`` when there is
          none.

        Raises
        ------
        FileNotFoundError
            If the met cannot be read and a folder may hold the results, or
            the build cannot be read and the variant made folders with two
            builds.
        """
        try:
            return self._variant_hash(kind, variant)
        except FileNotFoundError:
            found = self._readable_matches(variant)
            if not found:
                return None
            if "met" not in variant.readable_run_settings:
                raise
            # Only the build cannot be read: the folder this variant made.
            mine = [digest for digest, name in found if name == variant.name]
            if len(mine) > 1:
                raise
            if not mine:
                return None
            if kind == "particles":
                return mine[0]
            return variant.stage_hashes_on(mine[0]).get(kind)

    def _readable_matches(self, variant: Variant) -> list[tuple[str, str]]:
        """
        Return ``(hash, name)`` of each particles folder with *variant*'s settings, on what can be read here.

        The settings of :attr:`Variant.readable_run_settings`: without the
        met's when its files cannot be read, and without the build's
        checksums when the model's executable cannot be found. A stage's
        files come from particles with the same hash, so when no particles
        folder can hold the variant's results, no stage's folder can either.
        """
        wanted = variant.readable_run_settings
        digest = settings_hash(wanted)
        found = []
        for folder in _settings_folders(self.directory / "particles"):
            record = _read_yaml(self._dir("particles", folder) / SETTINGS_FILE)
            stored = read_run_settings(record["settings"])
            if settings_hash(_like(stored, wanted)) == digest:
                found.append((settings_hash(stored), record.get("name", "")))
        return found

    def _name(self, kind: Kind, variant: Variant) -> str | None:
        """
        Return the name of *variant*'s folder of *kind*, or ``None`` when it does not exist.

        Raises
        ------
        ValueError
            For ``footprints``, what the footprints' kind was called before
            it took the name of their stage. Any other string is taken for
            a stage the variant does not use.
        """
        if kind == "footprints" and kind not in variant.stages:
            raise ValueError(
                "The footprints' kind is 'footprint', the name of their stage, "
                "not 'footprints'."
            )
        digest = self._lookup_hash(kind, variant)
        if digest is None:
            return None
        for name, folder_hash in self._hashes.get(kind, {}).items():
            if folder_hash == digest:
                return name
        # A miss lists the tree again: another worker may have made the folder.
        for name, folder_hash in self.hashes(kind).items():
            if folder_hash == digest:
                return name
        return None

    def folder(self, kind: Kind, variant: Variant) -> Location | None:
        """
        Return *variant*'s folder of *kind*, or ``None`` before it exists.

        ``None`` too for a stage the variant does not use.
        """
        name = self._name(kind, variant)
        return None if name is None else self._dir(kind, name)

    def _dir(self, tree: str, name: str) -> Location:
        """Return the folder *name* of a tree: ``<tree>/settings=<name>``."""
        return self.directory / tree / f"settings={name}"

    def _create(self, kind: Kind, variant: Variant) -> str:
        """
        Return the name of *variant*'s folder of *kind*, writing its settings file on first use.

        A stage's record names the stage and its class, the folder and hash
        of each thing its files are made from (the particles, then each
        stage it reads), the versions of PYSTILT and of the stage's package,
        and the stage's settings.

        Raises
        ------
        ValueError
            If *kind* is a stage the variant does not use.
        """
        name = self._name(kind, variant)
        if name is not None:
            return name
        if kind == "particles":
            digest = variant.particles_hash
            record: dict[str, Any] = {
                "name": variant.name,
                "hash": digest,
                "pystilt": pystilt_version(),
                "model_version": variant.build.version,
                "settings": variant.run_settings,
            }
        else:
            if kind not in variant.stages:
                raise ValueError(f"Variant {variant.name!r} has no {kind} stage.")
            digest = variant.stage_hashes[kind]
            made_from = {"particles": variant.particles_hash}
            made_from.update(
                {
                    read: variant.stage_hashes[read]
                    for read in reads(variant.stage(kind))
                }
            )
            record = {
                "name": variant.name,
                "hash": digest,
                "stage": kind,
                "class": variant.stage_classes[kind],
                "reads": {
                    what: {"folder": self._create(what, variant), "hash": made_hash}
                    for what, made_hash in made_from.items()
                },
                "versions": _versions(variant.stage_classes[kind]),
                "settings": variant.stage_settings[kind],
            }
        name = f"{variant.name}-{digest[:HASH_CHARS]}"
        _write_settings(self._dir(kind, name) / SETTINGS_FILE, record)
        self._hashes.setdefault(kind, {})[name] = digest
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

    # -- one release's files ------------------------------------------------

    def path(
        self,
        kind: Kind,
        variant: Variant,
        release_id: str,
        realization: int | None = None,
    ) -> Location | None:
        """
        Return the file of one release's result, whether or not it exists.

        ``None`` before the variant's folder of *kind* exists, and for a
        stage the variant does not use. An ensemble's results are in a
        ``realization=k`` partition of its folder.
        """
        folder = self._found(kind, kind, variant, realization)
        if folder is None:
            return None
        return _release_file(folder, release_id, self._suffix(kind, variant))

    def make_path(
        self,
        kind: Kind,
        variant: Variant,
        release_id: str,
        realization: int | None = None,
    ) -> Location:
        """
        Return where one release's result of *kind* is written, making the variant's settings folder on first use.

        What :meth:`path` returns once the folder exists. The worker hands
        it to a stage, which writes its own file there.

        Raises
        ------
        ValueError
            If *kind* is a stage the variant does not use.
        """
        folder = self._made(kind, kind, variant, realization)
        return _release_file(folder, release_id, self._suffix(kind, variant))

    @staticmethod
    def _suffix(kind: Kind, variant: Variant) -> str:
        """Return the extension of *variant*'s files of *kind*: ``.parquet`` for particles, a stage's own for its files."""
        return ".parquet" if kind == "particles" else variant.stage(kind).suffix

    def read(
        self,
        kind: Kind,
        variant: Variant,
        release_id: str,
        realization: int | None = None,
        columns: list[str] | None = None,
    ) -> pa.Table:
        """
        Return one release's result file as a table of its stored columns.

        The file as written, with no ``date`` column: what :meth:`table`
        reads for many releases at once, for one. :meth:`table` builds a
        dataset over its files, which costs more than this on a single
        one, so a method that reads releases one at a time uses this.

        Parameters
        ----------
        kind : str
            Which result: ``"particles"``, or a stage whose files are
            Parquet, as the footprint's are.
        variant : Variant
            Whose result.
        release_id : str
            The release, whose file must exist (:meth:`present`).
        realization : int, optional
            Which realization of an ensemble.
        columns : list of str, optional
            Only these stored columns. Every column by default.

        Raises
        ------
        FileNotFoundError
            If the release has no file, or the variant no folder.
        """
        path = self.path(kind, variant, release_id, realization)
        if path is None:
            raise FileNotFoundError(
                f"No {kind} folder for variant {variant.name!r} in {self.directory}."
            )
        with readable(path) as source:
            return pq.read_table(source, columns=columns, partitioning=None)

    def log_path(
        self, variant: Variant, release_id: str, realization: int | None = None
    ) -> Location | None:
        """Return where the log of a release's transport model run is kept, or ``None`` before its folder exists."""
        logs = self._found("particles", "logs", variant, realization)
        return None if logs is None else _release_file(logs, release_id, ".log")

    def kept_workdir(
        self, variant: Variant, release_id: str, realization: int | None = None
    ) -> Location | None:
        """Return where a release's failed run's working directory is kept, or ``None`` before its folder exists."""
        kept = self._found("particles", "scratch", variant, realization)
        return None if kept is None else kept / _date_dir(release_id) / release_id

    # -- many releases' files ----------------------------------------------

    def present(
        self,
        kind: Kind,
        variant: Variant,
        release_ids: Iterable[str] | None = None,
        realization: int | None = None,
    ) -> frozenset[str]:
        """
        Return the releases that have a file of *kind* for *variant*.

        Only the date folders of *release_ids* are listed; without them,
        every date folder is. No file is opened.
        """
        folder = self._found(kind, kind, variant, realization)
        if folder is None:
            return frozenset()
        suffix = self._suffix(kind, variant)
        return frozenset(_list_release_files(folder, suffix, release_ids))

    def complete(
        self,
        variant: Variant,
        release_ids: Iterable[str] | None = None,
        realization: int | None = None,
    ) -> frozenset[str]:
        """
        Return the releases whose simulations under *variant* are complete (:func:`completed`).

        Only the releases with particles are looked for in each stage's
        folder.
        """
        particles = self.present("particles", variant, release_ids, realization)
        stages = [
            self.present(name, variant, particles, realization)
            for name in variant.stage_order
        ]
        return completed(particles, *stages)

    def table(
        self,
        kind: Kind,
        variant: Variant,
        release_ids: Iterable[str] | None = None,
        realization: int | None = None,
        columns: list[str] | None = None,
    ) -> pa.Table:
        """
        Return the files of many releases as one table.

        Columns are ``release``, the stored columns, and ``date`` (the
        release date, from the folder, as ``date32``).

        Parameters
        ----------
        kind : str
            Which results: ``"particles"``, or a stage whose files are
            Parquet, as the footprint's are.
        variant : Variant
            Whose results.
        release_ids : iterable of str, optional
            The releases to read, each of which must have a file
            (:meth:`present`), in the order the rows come. Without it, every
            file in the folder is read, in date order.
        realization : int, optional
            Which realization of an ensemble.
        columns : list of str, optional
            Only these stored columns, after ``release``. Every column by
            default.

        Raises
        ------
        ValueError
            If a column in *columns* is not in the files.
        """
        _check_kind(kind)
        nothing = _NO_PARTICLES if kind == "particles" else _NO_STAGE_FILES
        folder = self._found(kind, kind, variant, realization)
        if folder is None:
            return nothing
        suffix = self._suffix(kind, variant)
        if release_ids is None:
            paths = list(_list_release_files(folder, suffix).values())
        else:
            paths = [
                _release_file(folder, r, suffix) for r in dict.fromkeys(release_ids)
            ]
        if not paths:
            return nothing
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
        if columns is None:
            return dataset.to_table()
        wanted = ["release", *(c for c in columns if c != "release")]
        unknown = [c for c in wanted if c not in dataset.schema.names]
        if unknown:
            raise ValueError(
                f"No column {unknown[0]!r} in the {kind} files; they hold "
                f"{dataset.schema.names}."
            )
        return dataset.to_table(columns=wanted)

    # -- failure records ---------------------------------------------------

    def failure(
        self,
        kind: Kind,
        variant: Variant,
        release_id: str,
        realization: int | None = None,
    ) -> dict[str, Any] | None:
        """
        Return why a release's result of *kind* failed, as the worker recorded it, or ``None``.

        A record holds ``stage`` (``particles``, or the stage that failed),
        ``reason``, ``message``, and ``time``, and a ``traceback`` for an
        unexpected error. It is removed when the result is written.
        """
        logs = self._found(kind, "logs", variant, realization)
        if logs is None:
            return None
        path = _release_file(logs, release_id, FAILURE_SUFFIX)
        return _read_yaml(path) if path.exists() else None

    def failures(
        self,
        kind: Kind,
        variant: Variant,
        release_ids: Iterable[str] | None = None,
        realization: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Return ``{release_id: record}`` for the releases with a failure record of *kind*.

        With *release_ids*, only their date folders are listed. The records
        found are read in a few threads.
        """
        logs = self._found(kind, "logs", variant, realization)
        if logs is None:
            return {}
        paths = _list_release_files(logs, FAILURE_SUFFIX, release_ids)
        with ThreadPoolExecutor(max_workers=16) as pool:
            records = list(pool.map(_read_yaml, paths.values()))
        return dict(zip(paths, records, strict=True))

    def record_failure(
        self,
        kind: Kind,
        variant: Variant,
        release_id: str,
        record: dict[str, Any],
        realization: int | None = None,
    ) -> None:
        """
        Write why a release's result of *kind* failed, replacing an earlier record.

        It goes in the logs of the folder whose result failed: a particles
        failure covers every variant on those particles, a stage's failure
        every variant with that stage's settings on them.
        """
        logs = self._made(kind, "logs", variant, realization)
        _write_yaml(_release_file(logs, release_id, FAILURE_SUFFIX), record)

    def clear_failure(
        self,
        kind: Kind,
        variant: Variant,
        release_id: str,
        realization: int | None = None,
    ) -> None:
        """Remove a release's failure record of *kind*, once its result is written."""
        logs = self._found(kind, "logs", variant, realization)
        if logs is not None:
            _release_file(logs, release_id, FAILURE_SUFFIX).unlink(missing_ok=True)

    # -- writing -----------------------------------------------------------

    def write_particles(
        self,
        variant: Variant,
        release: Receptor,
        particles: pd.DataFrame,
        met_files: list[Path],
        realization: int | None = None,
    ) -> Location:
        """
        Write a release's particles (:func:`stilt.particles.write_particles`).

        The file records the run's settings, so it reads alone, and the
        folder's settings hash, the PYSTILT version, and the realization.

        Raises
        ------
        ValueError
            If *particles* has no rows. A particle file makes its simulation
            complete, and a run with no particles is a failure
            (``NO_PARTICLE_DATA``).
        """
        if particles.empty:
            raise ValueError(
                f"No particles to write for {release.id}: a particle file "
                "would make the simulation complete."
            )
        folder = self._made("particles", "particles", variant, realization)
        return write_particles(
            _release_file(folder, str(release.id)),
            particles,
            release,
            variant.run_settings,
            met_files,
            metadata=file_stamp(variant.particles_hash, realization),
        )

    def logged(
        self,
        variant: Variant,
        release_ids: Iterable[str] | None = None,
        realization: int | None = None,
    ) -> frozenset[str]:
        """
        Return the releases that have a transport model log for *variant*.

        Only the date folders of *release_ids* are listed. A log with no
        particles and no failure record is a run that started and did not
        finish (``interrupted`` in :meth:`stilt.Project.status`).
        """
        logs = self._found("particles", "logs", variant, realization)
        if logs is None:
            return frozenset()
        return frozenset(_list_release_files(logs, ".log", release_ids))

    def write_log(
        self,
        variant: Variant,
        release_id: str,
        text: str,
        realization: int | None = None,
    ) -> Location:
        """Write the log of a release's transport model run."""
        logs = self._made("particles", "logs", variant, realization)
        path = _release_file(logs, release_id, ".log")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def keep_workdir(
        self,
        variant: Variant,
        release_id: str,
        workdir: Path,
        realization: int | None = None,
    ) -> Location:
        """
        Copy a run's working directory to :meth:`kept_workdir`, replacing an earlier copy.

        A failed run's is kept, so CONTROL, SETUP.CFG, and the model's own
        output can be read after the job ends.
        """
        scratch = self._made("particles", "scratch", variant, realization)
        kept = scratch / _date_dir(release_id) / release_id
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


__all__ = ["Kind", "Output", "completed"]
