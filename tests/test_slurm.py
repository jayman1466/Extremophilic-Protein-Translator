"""Unit tests for eptrans.slurm array sizing."""
from eptrans.slurm import plan_array, MAX_TASKS, MAX_RUNNING


def test_small_no_growth():
    p = plan_array(50, chunk_size=1000)
    assert p.n_tasks == 1
    assert p.chunk_size == 1000
    assert not p.grown
    assert p.concurrency == 1  # only one task


def test_fits_within_limits():
    p = plan_array(5000, chunk_size=1000)
    assert p.n_tasks == 5
    assert p.concurrency == 5
    assert not p.grown


def test_grows_to_respect_max_tasks():
    # 199,923 genomes at 1000/chunk = 200 tasks (exactly the cap)
    p = plan_array(199923, chunk_size=1000, max_tasks=MAX_TASKS)
    assert p.n_tasks <= MAX_TASKS
    assert p.concurrency == MAX_RUNNING


def test_grows_when_chunk_too_small():
    # 500/chunk would be 400 tasks > 200 -> must grow
    p = plan_array(199923, chunk_size=500, max_tasks=200)
    assert p.grown
    assert p.n_tasks <= 200
    assert p.chunk_size >= 1000


def test_covers_all_items():
    p = plan_array(199923, chunk_size=1000, max_tasks=200)
    assert p.n_tasks * p.chunk_size >= 199923
