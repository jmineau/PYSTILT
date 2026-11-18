"""
What a result was made with, and the hash that names its folder.

Every folder in an output directory holds the results of one set of
settings, recorded in its ``_settings.yaml``. This module writes those
records and hashes them. A run's settings are its transport config without
the fields that change no particle (HYSPLIT's ``directory``), its met as
the met config records it (for HYSPLIT, the weather product and the crop),
the model build (for HYSPLIT, the SHA-256 of ``hycs_std``, not the version
it prints), and whether it is an ensemble (the realizations of one
share a folder, a ``realization=k`` partition each, so the number is not
in the hash). A
stage's settings are the record the stage itself writes
(:meth:`stilt.stages.Stage.settings`), hashed with the particles' hash and
the hash of each stage it reads (:func:`stage_hash`); the footprint's are
its footprint config, with the grid it is computed on, and the hash of the
geometry the grid was derived for.

The hash depends on what the settings mean rather than how they were
written. A stored record is read back through the current config classes,
so a setting added since, with a default, still matches, and a changed
default does not. The met's part is plain values and is kept as written.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from stilt.transport import ModelInfo, TransportConfig, get_model


def canonical(value: Any) -> Any:
    """Return *value* with the spellings that mean the same thing made equal."""
    if isinstance(value, Mapping):
        return {str(k): canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def settings_hash(settings: Mapping[str, Any]) -> str:
    """
    Return the SHA-256 hex digest of *settings*.

    Keys are sorted, whole-number floats equal their integers, and paths are
    strings, so the hash depends on what the settings mean rather than how
    they were written.
    """
    text = json.dumps(canonical(settings), separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()


# -- runs ---------------------------------------------------------------------


def run_settings(
    transport: TransportConfig,
    met: Mapping[str, Any],
    model: ModelInfo,
    ensemble: bool = False,
) -> dict[str, Any]:
    """
    Return the settings that identify a run, in canonical form.

    The transport model's settings (``transport.settings()``) sit at the top
    level, beside ``met`` (what the met config's ``settings()`` returns)
    and ``model`` (the build, without the version it reports). An ensemble records ``ensemble: true`` and its base
    seed; its realization ``k`` runs with ``seed + k`` in the folder's
    ``realization=k`` partition.
    """
    data = dict(transport.settings())
    data["met"] = dict(met)
    data["model"] = model.model_dump(mode="json")
    if ensemble:
        data["ensemble"] = True
    return canonical(data)


def _model_info(stored: Mapping[str, Any]) -> ModelInfo:
    """Return the model build a run settings record names, dropping fields this version does not have."""
    return ModelInfo.model_validate(
        {k: v for k, v in stored["model"].items() if k in ModelInfo.model_fields}
    )


def transport_from_settings(stored: Mapping[str, Any]) -> TransportConfig:
    """
    Return the transport model's config a run settings record holds.

    ``model.name`` says which model's config class reads it. Settings this
    version does not have are dropped. The fields a record leaves out
    (``directory`` for HYSPLIT) come back unset.
    """
    config_class = get_model(_model_info(stored).name).config_class
    fields = config_class.model_fields
    return config_class.model_validate({k: v for k, v in stored.items() if k in fields})


def read_run_settings(stored: Mapping[str, Any]) -> dict[str, Any]:
    """
    Return the run settings a ``_settings.yaml`` records, read through the current classes.

    Settings this version does not have are dropped, so a folder written
    before a setting was removed still loads. A record of a transport model
    that is not installed here is returned as written, with a warning. A folder of one realization,
    as realizations were written before they were partitions, keeps its
    number, so it hashes as it did and no current variant finds it.
    """
    name = _model_info(stored).name
    try:
        get_model(name)
    except (ImportError, ValueError) as error:
        # Another project's folder in a shared output directory, made by a
        # model not installed here: keep its record as written, so its hash
        # and everything else in the directory still read.
        warnings.warn(
            f"Transport model {name!r} of a stored run could not be loaded "
            f"({error}). Its settings are kept as written.",
            stacklevel=2,
        )
        return dict(stored)
    record = run_settings(
        transport_from_settings(stored),
        stored["met"],
        _model_info(stored),
        ensemble=bool(stored.get("ensemble", False)),
    )
    if stored.get("realization") is not None:
        record["realization"] = stored["realization"]
    return record


# -- what a result file records -------------------------------------------------


def pystilt_version() -> str:
    """Return the installed PYSTILT version, as ``stilt.__version__`` does."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("pystilt")
    except PackageNotFoundError:
        return "0+unknown"


def file_stamp(digest: str, realization: int | None = None) -> dict[bytes, bytes]:
    """
    Return what a result file records besides its contents, as Parquet file metadata.

    Its folder's settings hash (*digest*: ``variant.particles_hash``, or
    ``variant.stage_hashes[name]`` for a stage's file), the PYSTILT
    version, and, for an ensemble, its realization number. A particle file
    carries it, and so does a footprint; a stage of your own may write it
    too, so that its files say what made them.
    """
    stamp = {
        b"stilt:hash": digest.encode(),
        b"stilt:pystilt": pystilt_version().encode(),
    }
    if realization is not None:
        stamp[b"stilt:realization"] = str(realization).encode()
    return stamp


# -- stages -------------------------------------------------------------------


def stage_hash(
    name: str,
    settings: Mapping[str, Any],
    particles_hash: str,
    reads: Mapping[str, str] | None = None,
) -> str:
    """
    Return the hash of the results of the stage *name* with *settings*.

    A stage's results are named by what they were made from and with: the
    particles' hash (*particles_hash*), the hash of each stage it reads
    (*reads*, by the stage's name), and the stage's own settings record
    (:meth:`stilt.stages.Stage.settings`). So results made from other
    particles, or from another stage's results under other settings, never
    share a folder.

    Examples
    --------
    >>> stage_hash("footprint", {"tres": "1h"}, "a3f9c2") == settings_hash(
    ...     {"particles": "a3f9c2", "footprint": {"tres": "1h"}}
    ... )
    True
    """
    return settings_hash({"particles": particles_hash, **(reads or {}), name: settings})


__all__ = [
    "canonical",
    "file_stamp",
    "pystilt_version",
    "read_run_settings",
    "run_settings",
    "settings_hash",
    "stage_hash",
    "transport_from_settings",
]
