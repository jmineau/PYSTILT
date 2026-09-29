"""
A STILT project root and its file layout.

A project is one root — a local directory or an object-store URI — holding::

    config.yaml                    the user's settings (never rewritten)
    receptors.csv                  the user's receptors (only appended to)
    simulations/variants.yaml      PYSTILT's record of every variant that ran
    simulations/by-id/<receptor_id>/<variant>/<receptor_id>_traj.parquet
    simulations/by-id/<receptor_id>/<variant>/<receptor_id>_foot.nc
    simulations/by-id/<receptor_id>/<variant>/<receptor_id>_foot.empty
    simulations/by-id/<receptor_id>/<variant>/stilt.log

Everything is addressed by store key relative to the root. ``config.yaml`` and
``receptors.csv`` together *are* the project: the registered simulation set is
their receptors crossed with the configured variants. The record holds the
fully resolved settings of every variant ever registered, so a variant's
settings cannot change under its name once it has outputs.
"""

from __future__ import annotations

import re
import tempfile
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from stilt.store import Store, is_uri, make_store

if TYPE_CHECKING:
    from stilt.config import ModelConfig
    from stilt.receptors import Receptor

CONFIG_KEY = "config.yaml"
RECEPTORS_KEY = "receptors.csv"
RECORD_KEY = "simulations/variants.yaml"
SIMULATIONS_PREFIX = "simulations/by-id"
SIMULATION_LOG_FILENAME = "stilt.log"
SIMULATION_MET_DIRNAME = "met"


def resolve_directory(
    directory: str | Path | None = None, *, prefix: str = "pystilt_"
) -> Path:
    """Return a resolved directory path, creating a temp root when omitted."""
    if directory is None:
        return Path(tempfile.mkdtemp(prefix=prefix))
    directory = Path(directory)
    if directory.parent == Path("."):
        directory = directory.resolve()
    return directory


def project_slug(root: str) -> str:
    """Derive a DNS-safe / filename-safe slug from a project path or URI."""
    raw = root.rstrip("/")
    if is_uri(raw):
        raw = raw.split("://", 1)[1]
    parts = [part for part in raw.split("/") if part]
    candidate = parts[-1] if parts else "project"
    slug = candidate.lower().replace("_", "-")
    slug = re.sub(r"[^a-z0-9-]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "project"


def simulation_prefix(sim_id: object) -> str:
    """Return the store key prefix for one simulation's outputs (``str(sim_id)``)."""
    return f"{SIMULATIONS_PREFIX}/{sim_id}"


class Project:
    """
    One STILT project root plus its store.

    Parameters
    ----------
    root
        Local directory or object-store URI. A temporary directory is created
        when omitted.
    cache_dir
        Local cache for downloads from a remote store.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        cache_dir: str | Path | None = None,
    ) -> None:
        if root is None or not is_uri(root):
            self.root = str(resolve_directory(root))
            self.is_cloud = False
        else:
            self.root = str(root).rstrip("/")
            self.is_cloud = True
        self.store: Store = make_store(self.root, cache_dir=cache_dir)

    def __repr__(self) -> str:
        return f"Project({self.root!r})"

    def __str__(self) -> str:
        return self.root

    @property
    def name(self) -> str:
        """Human-readable project name (directory basename or URI slug)."""
        return project_slug(self.root) if self.is_cloud else Path(self.root).name

    @property
    def directory(self) -> Path:
        """Local project directory. Raises for cloud projects."""
        if self.is_cloud:
            raise TypeError(f"Cloud project {self.root!r} has no local directory.")
        return Path(self.root)

    @property
    def simulations_dir(self) -> Path:
        """Local ``simulations/by-id`` directory. Raises for cloud projects."""
        return self.directory / SIMULATIONS_PREFIX

    # -- inputs ----------------------------------------------------------------

    @property
    def has_config(self) -> bool:
        """Return whether a config has been written to the project."""
        return self.store.exists(CONFIG_KEY)

    @property
    def has_receptors(self) -> bool:
        """Return whether receptors have been written to the project."""
        return self.store.exists(RECEPTORS_KEY)

    def load_config(self) -> ModelConfig:
        """Load ``config.yaml`` from the store."""
        from stilt.config import ModelConfig

        if not self.has_config:
            raise FileNotFoundError(
                f"No config.yaml found in {self.root}. "
                "Create one with ModelConfig.to_yaml()."
            )
        raw = yaml.safe_load(self.store.read_bytes(CONFIG_KEY).decode()) or {}
        return ModelConfig.model_validate(raw)

    def save_config(self, config: ModelConfig) -> None:
        """Write ``config.yaml`` to the store (only the settings that were given)."""
        self.store.write_bytes(CONFIG_KEY, config.to_yaml().encode())

    def load_receptors(self) -> list[Receptor] | None:
        """Load ``receptors.csv`` from the store, or ``None`` when absent."""
        from stilt.receptors import read_receptors

        if not self.has_receptors:
            return None
        return read_receptors(StringIO(self.store.read_bytes(RECEPTORS_KEY).decode()))

    def add_receptors(
        self, receptors: list[Receptor], *, source: str | Path | None = None
    ) -> list[Receptor]:
        """
        Add *receptors* to ``receptors.csv`` and return the ones that were new.

        A project without a receptors file gets *source* copied byte for byte
        when given, else the receptors written out. An existing file is never
        rewritten: receptors it does not hold yet are appended in its own
        columns (:func:`stilt.receptors.append_receptors_csv`).
        """
        from stilt.receptors import append_receptors_csv, receptors_to_csv

        if not self.has_receptors:
            if source is not None:
                self.store.publish_file(source, RECEPTORS_KEY)
            else:
                self.store.write_bytes(
                    RECEPTORS_KEY, receptors_to_csv(receptors).encode()
                )
            return list(receptors)
        known = {r.id for r in self.load_receptors() or []}
        new = [r for r in receptors if r.id not in known]
        if new:
            text = self.store.read_bytes(RECEPTORS_KEY).decode()
            self.store.write_bytes(
                RECEPTORS_KEY, append_receptors_csv(text, new).encode()
            )
        return new

    # -- record ----------------------------------------------------------------

    def load_record(self) -> dict[str, dict[str, Any]]:
        """
        The record of what has run: ``{"mets": {...}, "variants": {...}}``.

        Each entry is the full dump of a :class:`~stilt.config.MetConfig` or
        :class:`~stilt.config.VariantConfig` as it was when registered. Empty
        when nothing has been registered yet.
        """
        if not self.store.exists(RECORD_KEY):
            return {"mets": {}, "variants": {}}
        raw = yaml.safe_load(self.store.read_bytes(RECORD_KEY).decode()) or {}
        return {"mets": raw.get("mets") or {}, "variants": raw.get("variants") or {}}

    def save_record(self, record: dict[str, dict[str, Any]]) -> None:
        """Write the record back (see :meth:`load_record`)."""
        text = yaml.safe_dump(record, default_flow_style=False, sort_keys=False)
        self.store.write_bytes(RECORD_KEY, text.encode())


__all__ = [
    "CONFIG_KEY",
    "RECEPTORS_KEY",
    "RECORD_KEY",
    "SIMULATIONS_PREFIX",
    "SIMULATION_LOG_FILENAME",
    "SIMULATION_MET_DIRNAME",
    "Project",
    "project_slug",
    "resolve_directory",
    "simulation_prefix",
]
