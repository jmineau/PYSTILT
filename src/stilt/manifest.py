"""
Project simulation registry manifest.

One row per registered simulation, holding only what is *not* derivable from the
outputs on disk: identity (``sim_id``, ``met``), the receptor, the scene label,
and the configured footprint targets. Completion is **never** stored here — it is
computed by key from the store (see :mod:`stilt.completion`).

The manifest replaces the per-row registration of the former index. It lives
in the project's ``.stilt/`` directory as ``manifest.parquet`` and is read and
written through a :class:`~stilt.storage.Store`, so it works on local filesystems
and cloud object stores alike.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterable

import pandas as pd

from stilt.receptors import Receptor
from stilt.simulation import SimID
from stilt.storage import ProjectFiles, Store

_COLUMNS = ["sim_id", "met", "receptor", "scene", "footprints"]


class Manifest:
    """Registry of registered simulations, persisted as ``.stilt/manifest.parquet``."""

    def __init__(self, store: Store) -> None:
        self._store = store

    @staticmethod
    def _key() -> str:
        return ProjectFiles.manifest_key()

    def read(self) -> pd.DataFrame:
        """Return the manifest as a DataFrame (empty if none has been written)."""
        if not self._store.exists(self._key()):
            return pd.DataFrame(columns=pd.Index(_COLUMNS))
        return pd.read_parquet(io.BytesIO(self._store.read_bytes(self._key())))

    def _write(self, frame: pd.DataFrame) -> None:
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False)
        self._store.write_bytes(self._key(), buffer.getvalue())

    def register(
        self,
        pairs: Iterable[tuple[str, Receptor]],
        *,
        footprint_names: list[str] | None = None,
        scene_id: str | None = None,
    ) -> None:
        """Upsert one or many simulations into the registry (last write wins)."""
        targets = json.dumps(sorted(set(footprint_names or [])))
        new_rows = [
            {
                "sim_id": sim_id,
                "met": SimID(sim_id).met,
                "receptor": json.dumps(receptor.to_dict()),
                "scene": scene_id,
                "footprints": targets,
            }
            for sim_id, receptor in pairs
        ]
        if not new_rows:
            return
        combined = pd.concat(
            [self.read(), pd.DataFrame(new_rows, columns=pd.Index(_COLUMNS))],
            ignore_index=True,
        )
        combined = combined.drop_duplicates(subset="sim_id", keep="last").reset_index(
            drop=True
        )
        self._write(combined)

    def sim_ids(self) -> list[str]:
        """Return all registered simulation identifiers in stable order."""
        return sorted(self.read()["sim_id"].tolist())

    def has(self, sim_id: str) -> bool:
        """Return whether one simulation is registered."""
        frame = self.read()
        return bool((frame["sim_id"] == str(sim_id)).any())

    def count(self) -> int:
        """Return the number of registered simulations."""
        return int(len(self.read()))

    def receptors_for(self, sim_ids: list[str]) -> dict[str, Receptor]:
        """Return receptors keyed by simulation id for the requested rows."""
        wanted = set(sim_ids)
        frame = self.read()
        return {
            str(row["sim_id"]): Receptor.from_dict(json.loads(str(row["receptor"])))
            for _, row in frame.iterrows()
            if row["sim_id"] in wanted
        }

    def footprint_names(self) -> list[str]:
        """Return the union of configured footprint targets across the registry."""
        names: set[str] = set()
        for raw in self.read()["footprints"]:
            names.update(json.loads(raw) if isinstance(raw, str) else (raw or []))
        return sorted(names)

    def sim_ids_by_scene(self) -> dict[str, list[str]]:
        """Return ``scene -> [sim_id]`` for rows carrying a non-null scene."""
        frame = self.read()
        if frame.empty:
            return {}
        scened = frame[frame["scene"].notna()]
        return {
            str(scene): sorted(group["sim_id"].tolist())
            for scene, group in scened.groupby("scene")
        }


__all__ = ["Manifest"]
