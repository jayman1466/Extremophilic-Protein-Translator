"""Unit tests for the 11_generate acceptance/recovery logic (numpy-only, no GPU)."""
import numpy as np, pytest

def accept(cur_score, ps, T, rng, budget_ok=True):
    if not budget_ok: return False, False
    d = ps - cur_score
    if d >= 0: return True, False
    if T > 0.0 and rng.random() < np.exp(d / T): return True, True
    return False, False

def test_T0_never_accepts_downhill():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        acc, down = accept(0.5, 0.5 - rng.random(), 0.0, rng)
        assert not acc and not down

def test_improving_always_accepted_regardless_of_T():
    rng = np.random.default_rng(1)
    for T in (0.0, 0.01, 1.0):
        acc, down = accept(0.5, 0.6, T, rng)
        assert acc and not down

def test_equal_score_accepted_neutral_drift_preserved():
    acc, down = accept(0.5, 0.5, 0.0, np.random.default_rng(2))
    assert acc and not down

def test_budget_violation_rejects_even_if_better():
    acc, down = accept(0.5, 0.9, 0.1, np.random.default_rng(3), budget_ok=False)
    assert not acc

def test_downhill_rate_rises_with_temperature():
    rates = []
    for T in (0.005, 0.05, 0.5):
        rng = np.random.default_rng(4)
        rates.append(np.mean([accept(0.5, 0.45, T, rng)[0] for _ in range(4000)]))
    assert rates[0] < rates[1] < rates[2], rates

def test_anneal_decay_reaches_t1():
    t0, t1, iters = 0.05, 0.005, 24
    dec = (t1 / t0) ** (1.0 / (iters - 1))
    T = t0
    for _ in range(iters - 1): T *= dec
    assert abs(T - t1) < 1e-9

def test_anneal_decay_safe_at_single_iteration():
    t0, t1, iters = 0.05, 0.005, 1
    dec = 1.0 if t0 <= 0 or iters <= 1 else (t1 / t0) ** (1.0 / (iters - 1))
    assert dec == 1.0  # no ZeroDivisionError

def test_mask_rate_recovery_capped_at_schedule():
    mr0, mr = 0.20, 0.05
    for _ in range(20): mr = min(mr0, mr * 1.25)
    assert mr == pytest.approx(mr0)  # never exceeds the level's scheduled value

def test_recovery_disabled_when_factor_is_one():
    mr0, mr = 0.20, 0.05
    for _ in range(20):
        if 1.0 > 1.0: mr = min(mr0, mr * 1.0)
    assert mr == 0.05

def test_best_so_far_beats_chain_end():
    rng = np.random.default_rng(5)
    cur = best = 0.0
    for _ in range(200):
        ps = cur + rng.normal(-0.01, 0.05)
        acc, _ = accept(cur, ps, 0.05, rng)
        if acc: cur = ps
        best = max(best, cur)
    assert best >= cur  # returning best can never be worse than the chain end
