"""Shared SLURM array-sizing helpers.

biotite `standard` QOS limits (verified via scontrol/sacctmgr):
  MaxJobsPerUser       = 10    (running array tasks at once -> the %N throttle)
  MaxSubmitJobsPerUser = 200   (queued + running array elements)
  MaxArraySize         = 1001  (max array index 1000)

So a submitted array must have <= min(MaxSubmitJobs, MaxArraySize) = 200 tasks,
and no more than 10 run simultaneously.
"""
from __future__ import annotations

from dataclasses import dataclass

# Defaults for biotite standard QOS.
MAX_RUNNING = 10
MAX_TASKS = 200          # min(MaxSubmitJobs=200, MaxArraySize=1001)


@dataclass
class ArrayPlan:
    n_items: int
    chunk_size: int
    n_tasks: int
    concurrency: int
    grown: bool          # True if chunk_size was grown to fit max_tasks


def plan_array(n_items: int, chunk_size: int, max_tasks: int = MAX_TASKS,
               concurrency: int = MAX_RUNNING) -> ArrayPlan:
    """Size a chunked array so n_tasks <= max_tasks, growing chunk_size if needed."""
    requested = chunk_size
    n_tasks = (n_items + chunk_size - 1) // chunk_size
    grown = False
    if n_tasks > max_tasks:
        chunk_size = (n_items + max_tasks - 1) // max_tasks
        n_tasks = (n_items + chunk_size - 1) // chunk_size
        grown = chunk_size != requested
    return ArrayPlan(n_items, chunk_size, n_tasks, min(concurrency, n_tasks), grown)
