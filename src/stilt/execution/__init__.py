"""
Running simulations: the runner that starts the work and the workers that do it.

:func:`run` runs what is missing here or on Slurm and waits; :func:`submit`
submits it and returns, and :func:`wait` waits for a job array.
:func:`pending` is the receptors either would run. :func:`task_share` is which receptors one task of a
job array runs, and :func:`job_script` the script such a job array runs, for
a scheduler PYSTILT does not drive itself. The workers are in
:mod:`stilt.execution.worker`.
"""

from .config import ExecutionConfig
from .runner import job_script, pending, run, submit, task_share, wait

__all__ = [
    "ExecutionConfig",
    "job_script",
    "pending",
    "run",
    "submit",
    "task_share",
    "wait",
]
