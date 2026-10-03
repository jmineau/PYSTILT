"""
The settings that identify a run, and their hash.

A run is one receptor under one :class:`TransportSettings`: the transport
fields that change its particles, the settings of its meteorology, and the
model that produced them. Two variants with equal settings are one run.
The hash of the settings names the run's folder in the output directory.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field

from .meteorology import MetConfig, MetSettings
from .params import TransportParams

#: Transport fields that change no particle, so they are not part of a run's identity.
UNRECORDED_FIELDS = frozenset({"exe_dir", "data_dir"})

#: Met fields that change no particle: where the files are downloaded from,
#: and how many a run needs before it is allowed to start.
UNRECORDED_MET_FIELDS = frozenset({"download_from", "n_min"})


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


class ModelInfo(BaseModel):
    """The transport model a run was made with."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(default="hysplit", description="Model name.")
    version: str = Field(
        ..., description="Version string of the build, such as ``v5.1.0``."
    )
    data_files: dict[str, str] | None = Field(
        None,
        description=(
            "SHA-256 of each data file the run used in place of the model's "
            "own, by file name. ``None`` when it used the model's own."
        ),
    )


class TransportSettings(TransportParams):
    """
    Everything that decides a run's particles.

    The transport fields of :class:`~stilt.config.TransportParams` (which also
    carry HYSPLIT's namelist writers and validators, so this is the object
    the driver runs with), the :class:`~stilt.config.MetSettings` of its
    meteorology, and the :class:`ModelInfo`. ``exe_dir`` and ``data_dir``
    are carried for running but left out of :meth:`identity` and
    :attr:`hash`: what they point at is recorded instead, as the build's
    version and the checksums of data files that differ from the bundled ones.

    A stored ``settings.yaml`` loads back through this class, so a run is
    found by re-validating and re-hashing what was stored rather than by
    comparing digests: a field added later with a default still matches, a
    changed default does not.
    """

    met: MetSettings = Field(description="Settings of the meteorology the run used.")
    model: ModelInfo = Field(
        description="Model and version that produced the particles."
    )
    realization: int | None = Field(
        None,
        description=(
            "Realization number within an ensemble, or ``None`` for a single "
            "run. Part of the identity, so realizations that HYSPLIT seeds from "
            "the clock (``krand=4``) stay separate runs."
        ),
    )

    @classmethod
    def build(
        cls,
        params: TransportParams,
        met: MetSettings | MetConfig,
        model: ModelInfo | None = None,
        realization: int | None = None,
    ) -> Self:
        """
        Return the settings for *params* run with *met*.

        *model* defaults to HYSPLIT at the version its transport model
        reports for *params* (the build ``params.exe_dir`` points at, or the
        bundled build).
        """
        if model is None:
            from stilt.transport import get_model

            name = "hysplit"
            transport_model = get_model(name)
            model = ModelInfo(
                name=name,
                version=transport_model.version(params),
                data_files=transport_model.data_files(params),
            )
        met_settings = met.settings() if isinstance(met, MetConfig) else met
        return cls(
            **params.model_dump(),
            met=met_settings,
            model=model,
            realization=realization,
        )

    @classmethod
    def from_stored(cls, stored: Mapping[str, Any]) -> Self:
        """
        Return the settings a ``_settings.yaml`` records, ignoring fields this version does not have.

        A setting removed from PYSTILT after a folder was written is dropped
        here, so the folder still loads, and lookup re-hashes what is left.
        A ``config.yaml`` is read strictly instead, so a typo there is an
        error.
        """
        known = {k: v for k, v in stored.items() if k in cls.model_fields}
        if isinstance(known.get("model"), Mapping):
            known["model"] = {
                k: v for k, v in known["model"].items() if k in ModelInfo.model_fields
            }
        return cls.model_validate(known)

    def identity(self) -> dict[str, Any]:
        """
        Return the settings that identify the run, in canonical form.

        This is what ``settings.yaml`` stores and :attr:`hash` digests.
        ``maxpar`` is given as HYSPLIT receives it, so an unset ``maxpar``
        equals ``numpar``.
        """
        exclude: dict[str, Any] = dict.fromkeys(UNRECORDED_FIELDS, True)
        exclude["met"] = set(UNRECORDED_MET_FIELDS)
        data = self.model_dump(mode="json", exclude=exclude)
        if data["model"].get("data_files") is None:
            # Runs with the model's own data files keep the hash they had
            # before data files were recorded.
            data["model"].pop("data_files", None)
        if data["maxpar"] is None:
            data["maxpar"] = self.numpar
        return canonical(data)

    @property
    def hash(self) -> str:
        """SHA-256 hex digest of :meth:`identity`."""
        return settings_hash(self.identity())


__all__ = [
    "UNRECORDED_FIELDS",
    "UNRECORDED_MET_FIELDS",
    "ModelInfo",
    "TransportSettings",
    "canonical",
    "settings_hash",
]
