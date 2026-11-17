"""
The footprint stage: what a ``stages`` block of ``config.yaml`` names.

A stage is what is made from each simulation's particles. A project lists
its stages by name under ``stages:``, each block giving the stage's import
path as ``class`` and then the stage's settings. :class:`FootprintStage` is
the one stage PYSTILT has.
"""

from __future__ import annotations

from .config import FootprintConfig


class FootprintStage:
    """
    The stage that grids each simulation's particles into its footprint.

    Name it as the ``class`` of a block under ``stages:`` in
    ``config.yaml``. The block's other keys are the footprint settings
    (:class:`FootprintConfig`), the defaults of every variant; a variant
    changes some under its own ``stages`` key, or sets the stage to null to
    make particles only.

    .. code-block:: yaml

       stages:
         footprint:
           class: stilt.footprint.FootprintStage
           grid: {xmin: -113.0, xmax: -110.5, ymin: 40.0, ymax: 42.0, xres: 0.01, yres: 0.01}

       variants:
         hrrr: {}
         hrrr-coarse: {stages: {footprint: {grid: {xres: 0.05, yres: 0.05}}}}
         hrrr-particles: {stages: {footprint: null}}

    Attributes
    ----------
    name : str
        ``footprint``, what its results are called.
    config_class : type
        :class:`FootprintConfig`, which checks the block's settings.
    """

    name = "footprint"
    config_class = FootprintConfig


__all__ = ["FootprintStage"]
