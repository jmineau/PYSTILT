"""
Classes named by import path: a transport model or a stage in another package.

``config.yaml`` names such a class as ``class: mypkg.models.MyModel``. No
registration is needed, so every process that opens the project finds the
class, provided its package is installed there.
"""

from __future__ import annotations

import importlib
from typing import Any

from stilt.exceptions import MissingClass

#: Packages whose name on PyPI is not the name they are imported by.
_INSTALL_NAMES = {"stilt": "pystilt"}


def import_class(path: str) -> Any:
    """
    Return the class at the import path *path*, such as ``mypkg.models.MyModel``.

    Raises
    ------
    MissingClass
        If the class's package is not installed here, or the package has no
        such class. The message names what to install.
    """
    module_name, _, attr = path.rpartition(".")
    package = module_name.partition(".")[0]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != package:
            raise MissingClass(
                f"config.yaml names {path}, which could not be imported "
                f"({error}). Check the path, and the version of {package} "
                "installed here."
            ) from None
        install = _INSTALL_NAMES.get(package, package)
        raise MissingClass(
            f"config.yaml names {path}, but {package} is not installed here. "
            f"Install {install} and run this again."
        ) from None
    except ImportError as error:
        raise MissingClass(
            f"config.yaml names {path}, which could not be imported ({error})."
        ) from None
    try:
        return getattr(module, attr)
    except AttributeError:
        raise MissingClass(
            f"config.yaml names {path}, but {module_name} has no {attr}. Check "
            f"the path, and the version of {package} installed here."
        ) from None


__all__ = ["import_class"]
