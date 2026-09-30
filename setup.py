"""
Build each wheel for one platform, carrying only that platform's HYSPLIT.

PYSTILT is pure Python, but it ships a compiled ``hycs_std``. A wheel is built
for one platform (``--plat-name``, see ``just dist``) and holds only the
binary that runs there, so pip will not install it anywhere else. The sdist
holds no binary (see ``MANIFEST.in``). A wheel built from it has none, and
the user points ``exe_dir`` at their own build.

Everything else is configured in ``pyproject.toml``.
"""

import shutil
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_py import build_py


def bundled_build(plat_name: str) -> str | None:
    """Return the ``hysplit/bin`` folder a wheel for *plat_name* carries."""
    if "linux" in plat_name and plat_name.endswith("x86_64"):
        return "linux_x64"
    if plat_name.startswith("macosx"):
        # Apple Silicon runs the x86-64 build through Rosetta, as the driver
        # expects. Published macOS wheels are x86-64 only.
        return "macos_x64"
    return None


class PlatformDistribution(Distribution):
    """A distribution with platform-specific files, installed to platlib."""

    def has_ext_modules(self):
        return True


class PlatformWheel(bdist_wheel):
    """A ``py3-none-<platform>`` wheel: any Python 3, one platform."""

    def get_tag(self):
        _, _, plat_name = super().get_tag()
        return "py3", "none", plat_name


class BuildPyOneBinary(build_py):
    """Copy the package, then drop every HYSPLIT build but the wheel's own."""

    def run(self):
        super().run()
        wheel = self.distribution.command_obj.get("bdist_wheel")
        if wheel is None or self.editable_mode:
            return
        keep = bundled_build(wheel.get_tag()[2])
        bin_dir = Path(self.build_lib, "stilt", "hysplit", "bin")
        # Also clears a build left in build/ by an earlier wheel.
        for build in bin_dir.iterdir():
            if build.is_dir() and build.name != keep:
                shutil.rmtree(build)


setup(
    distclass=PlatformDistribution,
    cmdclass={"bdist_wheel": PlatformWheel, "build_py": BuildPyOneBinary},
)
