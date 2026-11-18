"""
Two toy stages, for the tests of the stage interface.

They import nothing from the footprint, as the code that runs stages does
not. ``count`` reads the particles and ``double`` reads ``count``'s file, so
the two are the shortest chain there is. A config names them by import path:
``class: tests.fixtures.stages.CountStage``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict

COUNT = "tests.fixtures.stages.CountStage"
DOUBLE = "tests.fixtures.stages.DoubleStage"

ECHO = "tests.fixtures.stages.EchoStage"

#: Each call of a toy stage's ``run``, as (stage, receptor id, variant), in order.
RUNS: list[tuple[str, str, str]] = []

#: The name of the path each call of a toy stage's ``run`` was handed.
PATHS: list[str] = []


class CountConfig(BaseModel):
    """Settings of :class:`CountStage`."""

    model_config = ConfigDict(extra="forbid")

    scale: int = 1
    fail: bool = False
    silent: bool = False
    stale: bool = False


class CountStage:
    """Writes the number of particle rows of a simulation, times ``scale``, as text."""

    name = "count"
    config_class = CountConfig
    suffix = ".txt"

    def resolve(
        self,
        config: CountConfig,
        model: str,
        model_settings: Mapping[str, Any],
        directory: Path | None,
    ) -> CountConfig:
        """Refuse a transport model that releases 13 particles."""
        if model_settings.get("numpar") == 13:
            raise ValueError(
                f"it cannot count the particles of {model} with numpar 13."
            )
        return config

    def settings(self, config: CountConfig) -> dict[str, Any]:
        """Return the record of a count's settings."""
        return config.model_dump(mode="json")

    def check(
        self, config: CountConfig, receptors: pd.DataFrame, directory: Path | None
    ) -> list[tuple[str, str | None, str]]:
        """List the receptors south of the equator, which this stage pretends it cannot count."""
        south = receptors[receptors["latitude"] < 0].drop_duplicates("receptor")
        return [
            ("count", str(rid), "is south of the equator") for rid in south["receptor"]
        ]

    def run(self, config: CountConfig, sim: Any, path: Any) -> None:
        """
        Write the count of *sim*'s particle rows to *path*.

        With ``fail`` it writes half a file and raises; with ``silent`` it
        returns without writing one; with ``stale`` it raises an error a
        retry may get past.
        """
        RUNS.append((self.name, sim.receptor.id, sim.variant.name))
        PATHS.append(path.name)
        if config.silent:
            return
        if config.stale:
            raise ConnectionError("the count lost its connection")
        if config.fail:
            path.write_text("half a")
            raise RuntimeError("the count was told to fail")
        path.write_text(str(len(sim.particles) * config.scale))


class DoubleConfig(BaseModel):
    """Settings of :class:`DoubleStage`."""

    model_config = ConfigDict(extra="forbid")

    add: int = 0


class DoubleStage:
    """Writes twice the count of the ``count`` stage, plus ``add``, as text."""

    name = "double"
    config_class = DoubleConfig
    suffix = ".txt"
    reads = ("count",)

    def resolve(
        self,
        config: DoubleConfig,
        model: str,
        model_settings: Mapping[str, Any],
        directory: Path | None,
    ) -> DoubleConfig:
        """Return the settings as given."""
        return config

    def settings(self, config: DoubleConfig) -> dict[str, Any]:
        """Return the record of a double's settings."""
        return config.model_dump(mode="json")

    def check(
        self, config: DoubleConfig, receptors: pd.DataFrame, directory: Path | None
    ) -> list[tuple[str, str | None, str]]:
        """Find nothing."""
        return []

    def run(self, config: DoubleConfig, sim: Any, path: Any) -> None:
        """Write twice what the ``count`` stage wrote for *sim*, plus ``add``, to *path*."""
        RUNS.append((self.name, sim.receptor.id, sim.variant.name))
        path.write_text(str(2 * int(sim.path("count").read_text()) + config.add))


class EchoConfig(BaseModel):
    """Settings of :class:`EchoStage`."""

    model_config = ConfigDict(extra="forbid")

    fail: bool = False


class EchoStage:
    """Writes the receptor id as text; it reads no other stage."""

    name = "echo"
    config_class = EchoConfig
    suffix = ".txt"

    def resolve(
        self,
        config: EchoConfig,
        model: str,
        model_settings: Mapping[str, Any],
        directory: Path | None,
    ) -> EchoConfig:
        """Return the settings as given."""
        return config

    def settings(self, config: EchoConfig) -> dict[str, Any]:
        """Return the record of an echo's settings."""
        return config.model_dump(mode="json")

    def check(
        self, config: EchoConfig, receptors: pd.DataFrame, directory: Path | None
    ) -> list[tuple[str, str | None, str]]:
        """Find nothing."""
        return []

    def run(self, config: EchoConfig, sim: Any, path: Any) -> None:
        """Write *sim*'s receptor id to *path*, or raise with ``fail``."""
        RUNS.append((self.name, sim.receptor.id, sim.variant.name))
        if config.fail:
            raise RuntimeError("the echo was told to fail")
        path.write_text(sim.receptor.id)


class Forgetful:
    """A stage whose config class does not read back what ``settings`` wrote."""

    name = "forgetful"
    config_class = DoubleConfig
    suffix = ".txt"

    def settings(self, config: DoubleConfig) -> dict[str, Any]:
        """Return a record that leaves ``add`` out, so reading it back gives the default."""
        return {"add": 0} if config.add else {"add": 1}


def chain(**count: Any) -> dict[str, dict[str, Any]]:
    """Return a ``stages`` block with both toy stages, ``double`` written first."""
    return {"double": {"class": DOUBLE}, "count": {"class": COUNT, **count}}
