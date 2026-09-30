"""
The files of a STILT project: its inputs.

A project is a directory holding what the user wrote::

    config.yaml       the settings (never rewritten once loaded from here)
    receptors.csv     the receptors (only appended to)

Results go to the output directory ``config.yaml`` names
(:class:`stilt.output.Output`), ``./output`` by default. The simulations a
project defines are the receptors in ``receptors.csv`` crossed with the
variants in ``config.yaml``.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stilt.config import ModelConfig
    from stilt.receptors import Receptor

CONFIG_KEY = "config.yaml"
RECEPTORS_KEY = "receptors.csv"


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
    return Path(os.path.expandvars(os.path.expanduser(str(directory)))).resolve()


def project_slug(root: str) -> str:
    """Return a lowercase, hyphenated name for a project path, safe for file and job names."""
    parts = [part for part in str(root).rstrip("/").split("/") if part]
    candidate = parts[-1] if parts else "project"
    slug = candidate.lower().replace("_", "-")
    slug = re.sub(r"[^a-z0-9-]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "project"


class Project:
    """
    The input files of one STILT project.

    Most code uses a project through :class:`stilt.Model`, which creates one
    from its ``project`` argument.

    Parameters
    ----------
    root : str or Path, optional
        Project directory. A temporary directory is created when omitted.

    Attributes
    ----------
    directory : Path
        The project directory, absolute.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.directory = resolve_directory(root)

    def __repr__(self) -> str:
        return f"Project({str(self.directory)!r})"

    def __str__(self) -> str:
        return str(self.directory)

    @property
    def root(self) -> str:
        """The project directory as a string."""
        return str(self.directory)

    @property
    def name(self) -> str:
        """Project name: the directory name."""
        return self.directory.name

    @property
    def config_path(self) -> Path:
        return self.directory / CONFIG_KEY

    @property
    def receptors_path(self) -> Path:
        return self.directory / RECEPTORS_KEY

    def output_path(self, config: ModelConfig) -> Path:
        """Return the output directory *config* names, relative to the project unless absolute."""
        raw = Path(os.path.expandvars(os.path.expanduser(config.output)))
        return raw if raw.is_absolute() else (self.directory / raw).resolve()

    # -- inputs ----------------------------------------------------------------

    @property
    def has_config(self) -> bool:
        """Whether the project has a ``config.yaml``."""
        return self.config_path.exists()

    @property
    def has_receptors(self) -> bool:
        """Whether the project has a ``receptors.csv``."""
        return self.receptors_path.exists()

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
                f"No config.yaml found in {self.directory}. "
                "Create one with ModelConfig.to_yaml()."
            )
        return ModelConfig.from_yaml(self.config_path)

    def save_config(self, config: ModelConfig) -> None:
        """Write *config* to ``config.yaml``, with only the settings that were given."""
        self.directory.mkdir(parents=True, exist_ok=True)
        config.to_yaml(self.config_path)

    def load_receptors(self) -> list[Receptor] | None:
        """Load the project's ``receptors.csv``, or return ``None`` when there is none."""
        from stilt.receptors import read_receptors

        if not self.has_receptors:
            return None
        return read_receptors(self.receptors_path)

    def add_receptors(self, receptors: list[Receptor]) -> list[Receptor]:
        """
        Add receptors to ``receptors.csv`` and return the new ones.

        An existing file is only appended to. Receptors it already holds are
        skipped, and new rows use the file's own columns
        (:func:`stilt.receptors.append_receptors_csv`).
        """
        from stilt.receptors import append_receptors_csv, receptors_to_csv

        self.directory.mkdir(parents=True, exist_ok=True)
        if not self.has_receptors:
            self.receptors_path.write_text(receptors_to_csv(receptors))
            return list(receptors)
        known = {r.id for r in self.load_receptors() or []}
        new = [r for r in receptors if r.id not in known]
        if new:
            text = self.receptors_path.read_text()
            self.receptors_path.write_text(append_receptors_csv(text, new))
        return new


__all__ = [
    "CONFIG_KEY",
    "RECEPTORS_KEY",
    "Project",
    "project_slug",
    "resolve_directory",
]
