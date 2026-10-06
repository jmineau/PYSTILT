"""
Particle transforms that reweight ``foot`` before the footprint is computed.

A transform is any object with
``apply(particles, receptor=None, directory=None) -> DataFrame``: the
particle table, the receptor they were released from, and the directory a
relative file name in its settings starts from (the project's). The
built-in ones are pydantic models whose fields are their ``config.yaml``
keys::

    transforms:
      - kind: averaging_kernel
        levels: [0, 500, 1000]
        values: [1.0, 0.9, 0.7]
      - kind: pressure_weighting

An averaging kernel that differs per receptor (every satellite sounding has
its own) comes from a table in the project instead::

      - kind: averaging_kernel
        table: kernels.parquet

A ``kind`` containing a dot is the import path of your own transform class
(see the *Custom transforms* guide). Transforms run in list order, starting
from the unweighted particle table. Each returns a new table and leaves its
input unchanged.

The footprint is gridded from ``foot``, so a transform changes it through
``foot``; it may also add columns. The built-ins only scale ``foot``, which
the background and the transport error rely on to read each particle's
weight.

Each built-in transform is a module: :mod:`~stilt.transforms.averaging_kernel`,
:mod:`~stilt.transforms.pressure_weighting`, and
:mod:`~stilt.transforms.lifetime`. :func:`particle_pwf` and
:func:`ak_weights` hold the column weighting ported from X-STILT. They are
public so your own transforms can use them. This module loads transforms
from a config and applies them.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import pandas as pd
from pydantic import Field, TypeAdapter

from stilt.transforms._common import release_coordinate
from stilt.transforms.averaging_kernel import (
    AveragingKernel,
    ak_weights,
    averaging_kernel_table,
)
from stilt.transforms.lifetime import FirstOrderLifetime
from stilt.transforms.pressure_weighting import PressureWeighting, particle_pwf

if TYPE_CHECKING:
    from stilt.receptors import Receptor

BuiltinTransform = Annotated[
    AveragingKernel | PressureWeighting | FirstOrderLifetime,
    Field(discriminator="kind"),
]
_BUILTIN_ADAPTER: TypeAdapter[Any] = TypeAdapter(BuiltinTransform)


# -- loading and dumping ------------------------------------------------------------


def transform_kind(transform: Any) -> str:
    """Return the ``kind`` that declares a transform in a config."""
    kind = getattr(transform, "kind", None)
    if isinstance(kind, str):
        return kind
    cls = type(transform)
    return f"{cls.__module__}.{cls.__qualname__}"


def _import_kind(kind: str) -> type:
    """Import and return the class named by a ``module.Class`` string."""
    module_name, _, attr = kind.rpartition(".")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ImportError(f"{module_name} has no attribute {attr!r}") from exc


def load_transform(spec: Any) -> Any:
    """
    Return the transform for one config entry.

    ``spec`` is either a transform already (any object with ``apply``) or a
    mapping with a ``kind``. The ``kind`` is a built-in name or the import
    path of a pydantic model class, which is validated from the remaining
    keys. Only a pydantic model can be written back to ``config.yaml``.

    Raises
    ------
    ImportError
        If ``kind`` names a class that cannot be imported on this machine.
    """
    if hasattr(spec, "apply"):
        return spec
    if not isinstance(spec, dict):
        raise TypeError(
            f"Transform entries must be mappings with a 'kind' or objects with "
            f"apply(); got {type(spec).__name__}."
        )
    kind = spec.get("kind")
    if not isinstance(kind, str):
        raise ValueError("Transform entries require a string 'kind'.")
    if "." not in kind:
        return _BUILTIN_ADAPTER.validate_python(spec)

    fields = {k: v for k, v in spec.items() if k != "kind"}
    try:
        cls = _import_kind(kind)
    except ImportError as exc:
        raise ImportError(
            f"Transform {kind!r} could not be imported ({exc}). Install the "
            "package that defines it on this machine."
        ) from exc
    if not callable(getattr(cls, "apply", None)):
        raise TypeError(f"{kind} does not define an apply() method.")
    if not hasattr(cls, "model_validate"):
        raise TypeError(
            f"{kind} is not a pydantic model. A transform named in a config "
            "must be one so its settings can be written back."
        )
    return cls.model_validate(fields)


def dump_transform(transform: Any) -> dict[str, Any]:
    """
    Return the config mapping for one transform, the inverse of :func:`load_transform`.

    A mapping is returned as it is. A footprint read from a file keeps a
    transform it cannot import that way.
    """
    if isinstance(transform, dict):
        return dict(transform)
    if hasattr(transform, "model_dump"):
        data = dict(transform.model_dump(mode="json", exclude_none=True))
    else:
        raise TypeError(
            f"{type(transform).__qualname__} cannot be written to config: transforms "
            "declared in config.yaml must be pydantic models."
        )
    data.pop("kind", None)
    return {"kind": transform_kind(transform), **data}


def apply_transforms(
    particles: pd.DataFrame,
    transforms: Sequence[Any],
    receptor: Receptor | None = None,
    directory: str | Path | None = None,
) -> pd.DataFrame:
    """
    Apply *transforms* in order, each with the receptor and the directory.

    Returns ``particles`` itself when there are none.
    """
    for transform in transforms:
        particles = transform.apply(particles, receptor, directory)
    return particles


__all__ = [
    "AveragingKernel",
    "FirstOrderLifetime",
    "PressureWeighting",
    "ak_weights",
    "apply_transforms",
    "averaging_kernel_table",
    "dump_transform",
    "load_transform",
    "particle_pwf",
    "release_coordinate",
    "transform_kind",
]
