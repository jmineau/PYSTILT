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
from importlib.resources import files as pkg_files
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field

from .meteorology import MetConfig, MetSettings
from .params import STILTParams

#: Transport fields that change no particle, so they are not part of a run's identity.
UNRECORDED_FIELDS = frozenset({"timeout", "rm_dat", "exe_dir"})

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


def hysplit_version(exe_dir: str | Path | None = None) -> str:
    """
    Return the version of the HYSPLIT build in *exe_dir*, or of the bundled build.

    A build directory names its version in a ``version`` file beside
    ``hycs_std``, as the bundled one does. Two builds with the same settings
    can give different particles, so the version is part of a run's
    identity.

    Raises
    ------
    FileNotFoundError
        If *exe_dir* has no ``version`` file.
    """
    if exe_dir is None:
        path = Path(str(pkg_files("stilt.transport.hysplit") / "bin" / "version"))
    else:
        path = Path(exe_dir) / "version"
        if not path.exists():
            raise FileNotFoundError(
                f"{exe_dir} has no 'version' file. A custom hycs_std build needs "
                "one beside the binary, holding its version string (such as "
                "v5.3.2+t0-rows), so its runs are told apart from other builds'."
            )
    return path.read_text().strip()


class ModelInfo(BaseModel):
    """The transport model a run was made with."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(default="hysplit", description="Model name.")
    version: str = Field(
        ..., description="Version string of the build, such as ``v5.1.0``."
    )


class TransportSettings(STILTParams):
    """
    Everything that decides a run's particles.

    The transport fields of :class:`~stilt.config.STILTParams` (which also
    carry HYSPLIT's namelist writers and validators, so this is the object
    the driver runs with), the :class:`~stilt.config.MetSettings` of its
    meteorology, and the :class:`ModelInfo`. Fields that change no particle
    (``timeout``, ``rm_dat``, ``exe_dir``) are carried for running but left
    out of :meth:`identity` and :attr:`hash`.

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
        params: STILTParams,
        met: MetSettings | MetConfig,
        model: ModelInfo | None = None,
        realization: int | None = None,
    ) -> Self:
        """
        Return the settings for *params* run with *met*.

        *model* defaults to HYSPLIT at the version of the build
        ``params.exe_dir`` points at, or the bundled build.
        """
        if model is None:
            model = ModelInfo(version=hysplit_version(params.exe_dir))
        met_settings = met.settings() if isinstance(met, MetConfig) else met
        return cls(
            **params.model_dump(),
            met=met_settings,
            model=model,
            realization=realization,
        )

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
    "hysplit_version",
    "settings_hash",
]
