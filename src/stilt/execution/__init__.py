"""
Running simulations: the runner that starts the work and the workers that do it.

:func:`run` runs what is missing here or on Slurm and waits; :func:`submit`
submits it and returns. :func:`task_share` is which receptors one task of a
job array runs, and :func:`job_script` the script such a job array runs, for
a scheduler PYSTILT does not drive itself. The workers are in
:mod:`stilt.execution.worker`.
"""

from .config import ExecutionConfig
from .runner import job_script, run, submit, task_share

__all__ = [
    "ExecutionConfig",
    "job_script",
    "run",
    "submit",
    "task_share",
]
