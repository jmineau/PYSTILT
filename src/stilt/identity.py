"""
What a result was made with, and the hash that names its folder.

Every folder in an output directory holds the results of one set of
settings, recorded in its ``_settings.yaml``. This module writes those
records and hashes them. A run's settings are its transport config without
the fields that change no particle (``exe_dir``, ``data_dir``), its met
without the directories, the model build, and whether it is an ensemble
(the realizations of one share a folder, a ``realization=k`` partition
each, so the number is not in the hash). A
footprint's settings are its footprint config, with the grid it is
computed on, and the hash of the geometry the grid was derived for.

The hash depends on what the settings mean rather than how they were
written. A stored record is read back through the current config classes,
so a setting added since, with a default, still matches, and a changed
default does not.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from stilt.footprint.config import FootprintConfig
from stilt.meteorology import MetConfig
from stilt.transforms import dump_transform, load_transform, transform_kind
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
    met: MetConfig,
    model: ModelInfo,
    ensemble: bool = False,
) -> dict[str, Any]:
    """
    Return the settings that identify a run, in canonical form.

    The transport model's settings (``transport.settings()``) sit at the top
    level, beside ``met`` and ``model``. An ensemble records
    ``ensemble: true`` and its base seed; its realization ``k`` runs with
    ``seed + k`` in the folder's ``realization=k`` partition. A single run
    records ``realization: null``, as every run did before ensembles were
    partitions, so its folder keeps its hash.
    """
    data = dict(transport.settings())
    data["met"] = met.settings()
    record = model.model_dump(mode="json")
    if record.get("data_files") is None:
        # Runs with the model's own data files keep the hash they had
        # before data files were recorded.
        record.pop("data_files", None)
    data["model"] = record
    if ensemble:
        data["ensemble"] = True
    else:
        data["realization"] = None
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
    (``exe_dir`` and ``data_dir`` for HYSPLIT) come back unset.
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
        MetConfig.model_validate(stored["met"]),
        _model_info(stored),
        ensemble=bool(stored.get("ensemble", False)),
    )
    if stored.get("realization") is not None:
        record["realization"] = stored["realization"]
    return record


# -- footprints ---------------------------------------------------------------


def _record_transform(transform: Any) -> dict[str, Any]:
    """Return a transform as recorded, its ``kind`` alone if it has no settings to write."""
    try:
        return dump_transform(transform)
    except TypeError:
        return {"kind": transform_kind(transform)}


def _read_transform(spec: dict[str, Any], source: str) -> Any:
    """Return a recorded transform, or its mapping when it cannot be rebuilt here."""
    try:
        return load_transform(spec)
    except (ImportError, TypeError, ValueError) as exc:
        warnings.warn(
            f"{source}: transform {spec.get('kind')!r} could not be rebuilt "
            f"({exc}). It is kept as its settings and cannot be applied.",
            stacklevel=3,
        )
        return spec


def footprint_settings(
    footprint: FootprintConfig, geometry_hash: str | None
) -> dict[str, Any]:
    """
    Return the settings that identify a footprint, besides its particles, in canonical form.

    The same record goes in a footprint folder's ``_settings.yaml`` and in
    each footprint file and array. A transform that is not a pydantic model
    is recorded by its ``kind`` alone.
    """
    data = footprint.model_dump(mode="json", exclude={"transforms"})
    data["transforms"] = [_record_transform(t) for t in footprint.transforms]
    data["geometry_hash"] = geometry_hash
    return canonical(data)


def read_footprint_settings(
    stored: Mapping[str, Any], source: str = "footprint settings"
) -> tuple[FootprintConfig, str | None]:
    """
    Return the footprint config and geometry hash that :func:`footprint_settings` recorded.

    A transform whose class cannot be imported here is kept as its mapping,
    with a warning naming *source*, so the record still reads.
    """
    data = dict(stored)
    geometry_hash = data.pop("geometry_hash", None)
    specs = data.pop("transforms", [])
    config = FootprintConfig.model_validate(data)
    # model_copy skips validation, so a mapping stays a mapping.
    config = config.model_copy(
        update={"transforms": [_read_transform(s, source) for s in specs]}
    )
    return config, geometry_hash


def footprint_hash(particles_hash: str, settings: Mapping[str, Any]) -> str:
    """Return the hash of footprints with *settings*, made from the particles hashed *particles_hash*."""
    return settings_hash({"particles": particles_hash, "footprint": settings})


__all__ = [
    "footprint_settings",
    "read_footprint_settings",
    "read_run_settings",
    "run_settings",
    "transport_from_settings",
    "settings_hash",
]
