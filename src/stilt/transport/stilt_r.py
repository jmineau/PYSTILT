"""
STILT-R as a transport model that runs nothing: the settings of runs imported from STILT-R.

STILT-R (`uataq/stilt <https://github.com/uataq/stilt>`_) runs HYSPLIT from
R. :func:`stilt.migration.import_stilt_r` writes the runs of a STILT-R or
X-STILT project into an output directory, each with the settings STILT-R's
files record. A variant with ``model: stilt-r`` names their folder, so
``project.status``, ``project.particles``, and ``project.jacobian`` read
them. PYSTILT cannot run STILT-R: a run of such a variant fails.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from stilt.exceptions import SimulationError
from stilt.transport import ModelRun
from stilt.transport.hysplit.config import HysplitConfig

if TYPE_CHECKING:
    from stilt.receptors import Receptor


class StiltRConfig(HysplitConfig):
    """
    The settings of imported STILT-R runs: HYSPLIT's, as STILT-R ran it, and the STILT-R version.

    The settings are :class:`~stilt.transport.hysplit.HysplitConfig`'s,
    under the same names, as STILT-R's files record them (``SETUP.CFG``,
    ``CONTROL``, and the ``.rds`` file; see
    :func:`stilt.migration.import_stilt_r`). Every one must be given:
    none has a default, because an imported run's settings name its folder,
    and a default would stand in for a value no file gave. The
    ``config.yaml`` variant the import prints gives them all.
    """

    #: Fields left out of a run's recorded settings. ``version`` is recorded
    #: as the transport model's version instead, as HYSPLIT's build is.
    UNRECORDED: ClassVar[frozenset[str]] = frozenset({"exe_dir", "data_dir", "version"})

    version: str | None = Field(
        None,
        description=(
            "The STILT-R version that made the runs, such as a commit of "
            "uataq/stilt. No STILT-R file records it, so the import is given "
            "it. Recorded as the transport model's version."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _every_setting_given(cls, data: Any) -> Any:
        """Require every recorded setting: an imported run has no defaults."""
        if isinstance(data, dict):
            missing = sorted(set(cls.model_fields) - cls.UNRECORDED - set(data))
            if missing:
                raise ValueError(
                    f"An imported STILT-R run gives every setting, and these are "
                    f"missing: {', '.join(missing)}. Copy the variant "
                    "stilt.migration.import_stilt_r printed."
                )
        return data


class StiltRMetConfig(BaseModel):
    """
    The met of imported STILT-R runs, as written under ``mets:`` in ``config.yaml``.

    It is the directory the runs' ``CONTROL`` files name, as they name it.
    The met files need not exist any more; each run's own files are
    recorded in its particle file (``stilt:met_files``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    met_path: str = Field(
        ...,
        description=(
            "Directory of the met files the runs read, as their CONTROL files "
            "name it: the met_path of STILT-R's run_stilt.r, or a folder in it."
        ),
    )

    def settings(self) -> dict[str, Any]:
        """Return what an imported run records of its met: the directory, as written."""
        return {"met_path": self.met_path}


class StiltRModel:
    """
    The ``stilt-r`` transport model, which ``get_model("stilt-r")`` returns.

    It holds the settings of imported STILT-R runs and runs nothing.
    :meth:`run` fails, so a ``stilt run`` of a receptor STILT-R did not run
    records a failure for it.
    """

    name = "stilt-r"
    config_class = StiltRConfig
    met_config_class = StiltRMetConfig
    #: Nothing runs, so no files are written.
    needs_workdir = False
    batched = False

    def version(self, config: StiltRConfig) -> str:
        """
        Return the STILT-R version of *config*.

        Raises
        ------
        ValueError
            If *config* gives none.
        """
        if config.version is None:
            raise ValueError(
                "A stilt-r variant needs version: the STILT-R version given to "
                "stilt.migration.import_stilt_r, as it printed the variant."
            )
        return config.version

    def data_files(self, config: StiltRConfig) -> dict[str, str] | None:
        """Return ``None``: the data files are STILT-R's own, fixed by its version."""
        return None

    def run(
        self,
        receptor: Receptor,
        config: StiltRConfig,
        met: StiltRMetConfig,
        window: tuple[dt.datetime, dt.datetime],
        workdir: Path | None = None,
        timeout: int | None = None,
    ) -> ModelRun:
        """
        Fail: STILT-R runs are imported, not run.

        Raises
        ------
        SimulationError
            Always, with ``reason`` ``NOT_RUNNABLE``.
        """
        raise SimulationError(
            f"{receptor.id}: PYSTILT cannot run STILT-R. A stilt-r variant reads "
            "the runs stilt.migration.import_stilt_r wrote; run the receptor "
            "with a hysplit variant instead.",
            reason="NOT_RUNNABLE",
        )


__all__ = ["StiltRConfig", "StiltRMetConfig", "StiltRModel"]
