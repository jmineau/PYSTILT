"""
The files of a STILT project.

A project is a local directory or an object-store URI holding::

    config.yaml                    the user's settings (never rewritten)
    receptors.csv                  the user's receptors (only appended to)
    simulations/variants.yaml      the settings every variant ran with
    simulations/by-id/<receptor_id>/<variant>/<receptor_id>_traj.parquet
    simulations/by-id/<receptor_id>/<variant>/<receptor_id>_foot.nc
    simulations/by-id/<receptor_id>/<variant>/<receptor_id>_foot.empty
    simulations/by-id/<receptor_id>/<variant>/stilt.log

Every file is addressed by its path relative to the root. The simulations a
project defines are the receptors in ``receptors.csv`` crossed with the
variants in ``config.yaml``. ``simulations/variants.yaml`` records the
settings of every variant that has been registered, so those settings cannot
change under the same name once the variant has outputs.
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


def resolve_directory(
    directory: str | Path | None = None, *, prefix: str = "pystilt_"
) -> Path:
    """
    Return *directory* as an absolute path, or a new temporary directory when omitted.

    A relative path is taken from the current directory, so a worker started
    elsewhere still finds the same place.
    """
    if directory is None:
        return Path(tempfile.mkdtemp(prefix=prefix))
    return Path(directory).resolve()


def project_slug(root: str) -> str:
    """Return a lowercase, hyphenated name for a project path or URI, safe for filenames and DNS."""
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
    """Return the key prefix of one simulation's outputs, ``simulations/by-id/<receptor_id>/<variant>``."""
    return f"{SIMULATIONS_PREFIX}/{sim_id}"


class Project:
    """
    The files of one STILT project and the store that holds them.

    Most code uses a project through :class:`stilt.Model`, which creates one
    from its ``project`` argument.

    Parameters
    ----------
    root : str or Path, optional
        Local directory or object-store URI. A temporary directory is created
        when omitted.
    cache_dir : str or Path, optional
        Local cache for files downloaded from an object store.

    Attributes
    ----------
    root : str
        The project root as a string.
    is_cloud : bool
        Whether the root is an object-store URI.
    store : Store
        Reads and writes the project's files.
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
        """Project name, from the directory name or a slug of the URI."""
        return project_slug(self.root) if self.is_cloud else Path(self.root).name

    @property
    def directory(self) -> Path:
        """Local project directory. Raises ``TypeError`` for a cloud project."""
        if self.is_cloud:
            raise TypeError(f"Cloud project {self.root!r} has no local directory.")
        return Path(self.root)

    @property
    def simulations_dir(self) -> Path:
        """Local ``simulations/by-id`` directory. Raises ``TypeError`` for a cloud project."""
        return self.directory / SIMULATIONS_PREFIX

    # -- inputs ----------------------------------------------------------------

    @property
    def has_config(self) -> bool:
        """Whether the project has a ``config.yaml``."""
        return self.store.exists(CONFIG_KEY)

    @property
    def has_receptors(self) -> bool:
        """Whether the project has a ``receptors.csv``."""
        return self.store.exists(RECEPTORS_KEY)

    def load_config(self) -> ModelConfig:
        """
        Load the project's ``config.yaml``.

        Raises
        ------
        FileNotFoundError
            If the project has no ``config.yaml``.
        """
        from stilt.config import ModelConfig

        if not self.has_config:
            raise FileNotFoundError(
                f"No config.yaml found in {self.root}. "
                "Create one with ModelConfig.to_yaml()."
            )
        raw = yaml.safe_load(self.store.read_bytes(CONFIG_KEY).decode()) or {}
        return ModelConfig.model_validate(raw)

    def save_config(self, config: ModelConfig) -> None:
        """Write *config* to ``config.yaml``, with only the settings that were given."""
        self.store.write_bytes(CONFIG_KEY, config.to_yaml().encode())

    def load_receptors(self) -> list[Receptor] | None:
        """Load the project's ``receptors.csv``, or return ``None`` when there is none."""
        from stilt.receptors import read_receptors

        if not self.has_receptors:
            return None
        return read_receptors(StringIO(self.store.read_bytes(RECEPTORS_KEY).decode()))

    def add_receptors(self, receptors: list[Receptor]) -> list[Receptor]:
        """
        Add receptors to ``receptors.csv`` and return the new ones.

        An existing file is only appended to. Receptors it already holds are
        skipped, and new rows use the file's own columns
        (:func:`stilt.receptors.append_receptors_csv`).

        Parameters
        ----------
        receptors : list of Receptor
            Receptors to add.

        Returns
        -------
        list of Receptor
            The receptors that were not in the file before.
        """
        from stilt.receptors import append_receptors_csv, receptors_to_csv

        if not self.has_receptors:
            self.store.write_bytes(RECEPTORS_KEY, receptors_to_csv(receptors).encode())
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
        Return the settings every registered met and variant ran with.

        Returns
        -------
        dict
            ``{"mets": {name: settings}, "variants": {name: settings}}``.
            Each entry holds every field of a :class:`~stilt.config.MetConfig`
            or :class:`~stilt.config.VariantConfig` as it was when
            registered. Both are empty before anything is registered.
        """
        if not self.store.exists(RECORD_KEY):
            return {"mets": {}, "variants": {}}
        raw = yaml.safe_load(self.store.read_bytes(RECORD_KEY).decode()) or {}
        return {"mets": raw.get("mets") or {}, "variants": raw.get("variants") or {}}

    def save_record(self, record: dict[str, dict[str, Any]]) -> None:
        """Write the record of registered settings (see :meth:`load_record`)."""
        text = yaml.safe_dump(record, default_flow_style=False, sort_keys=False)
        self.store.write_bytes(RECORD_KEY, text.encode())


__all__ = [
    "CONFIG_KEY",
    "RECEPTORS_KEY",
    "RECORD_KEY",
    "SIMULATIONS_PREFIX",
    "Project",
    "project_slug",
    "resolve_directory",
    "simulation_prefix",
]
