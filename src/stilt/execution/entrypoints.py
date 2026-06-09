from __future__ import annotations

import time
from typing import TYPE_CHECKING

from stilt.errors import ConfigValidationError

from .execute import execute_batch, execute_task
from .tasks import plan_simulation_task

if TYPE_CHECKING:
    from stilt import Model


def push_simulations(
    model: Model,
    sim_ids: list[str],
    n_cores: int = 1,
    *,
    skip_existing: bool | None = None,
) -> None:
    """Execute an assigned list of simulations without claim polling."""
    if sim_ids:
        execute_batch(
            [
                plan_simulation_task(
                    model,
                    sim_id,
                    skip_existing=skip_existing,
                )
                for sim_id in sim_ids
            ],
            n_cores=n_cores,
        )


def pull_simulations(
    model: Model,
    follow: bool = False,
    poll_interval: float = 10.0,
    *,
    skip_existing: bool | None = None,
) -> None:
    """Drain the work queue through atomic claims."""
    queue = model.queue
    if queue is None:
        raise ConfigValidationError(
            "Pull-mode workers require a Postgres work queue. "
            "Configure it via PYSTILT_DB_URL."
        )

    idle_sleep = max(poll_interval, 0.1)
    max_idle_sleep = min(60.0, max(idle_sleep, poll_interval * 8))
    while True:
        with queue.claim_one() as claim:
            if claim is None:
                if follow:
                    time.sleep(idle_sleep)
                    idle_sleep = min(idle_sleep * 2.0, max_idle_sleep)
                    continue
                return
            idle_sleep = max(poll_interval, 0.1)
            result = execute_task(
                plan_simulation_task(
                    model,
                    claim.sim_id,
                    skip_existing=skip_existing,
                )
            )
            claim.record(result)
