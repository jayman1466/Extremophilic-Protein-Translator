"""Aggressiveness -> generation-parameter schedule.

One user-facing knob (a level 1..5, or a 0..1 scalar) maps to a *coordinated*
setting of the low-level masked-generation levers, so users never tune
mask_rate / gamma incoherently. The active-site freeze and catalytic-RMSD gate
are invariant across all levels — aggressiveness moves only the mutable-surface
budget, never the protected catalytic core.

Levers (see docs/modeling_design.md 16):
  target_mut_frac : goal fraction of MUTABLE residues changed (the semantic knob)
  mask_rate       : fraction of positions masked per Gibbs pass (magnitude)
  gamma           : conservation exponent in mask weight (1-c_i)^gamma (targeting)
                    low gamma reaches into moderately-conserved positions (bold);
                    high gamma confines masking to the least-conserved surface.

Gibbs iteration count is NOT set here — it is convergence-driven at generation
time (classifier-score plateau OR mutation budget reached OR acceptance collapse
under the MPNN gate OR a hard max-iteration cap). See gibbs_stop_rule().
"""
from __future__ import annotations

N_LEVELS = 5
MAX_GIBBS_ITERS = 40          # hard safety cap
PLATEAU_WINDOW = 4            # passes over which classifier delta is measured
PLATEAU_TOL = 1e-3            # classifier-score delta below this = plateau
ACCEPT_COLLAPSE = 0.02        # accepted-move fraction below this = stuck

# level -> coordinated schedule
_SCHEDULE = {
    1: dict(label="Conservative", target_mut_frac=0.04, mask_rate=0.10, gamma=3.0),
    2: dict(label="Cautious",     target_mut_frac=0.07, mask_rate=0.12, gamma=2.5),
    3: dict(label="Moderate",     target_mut_frac=0.10, mask_rate=0.15, gamma=2.0),
    4: dict(label="Bold",         target_mut_frac=0.15, mask_rate=0.18, gamma=1.4),
    5: dict(label="Aggressive",   target_mut_frac=0.20, mask_rate=0.22, gamma=1.0),
}


def schedule(level: int) -> dict:
    """Map an integer aggressiveness level (1..N_LEVELS) to generation params."""
    level = max(1, min(N_LEVELS, int(level)))
    return dict(level=level, **_SCHEDULE[level])


def span_levels(n_designs: int) -> list[int]:
    """N designs auto-span conservative->aggressive (the '5 designs of varying
    aggressiveness' spec). n<=1 -> the moderate midpoint; otherwise evenly spaced
    across 1..N_LEVELS."""
    if n_designs <= 1:
        return [3]
    if n_designs <= N_LEVELS:
        # evenly spaced integer levels across the range
        step = (N_LEVELS - 1) / (n_designs - 1)
        return [round(1 + i * step) for i in range(n_designs)]
    # more designs than levels: cycle levels, biased to cover the full range first
    base = list(range(1, N_LEVELS + 1))
    out = []
    while len(out) < n_designs:
        out.extend(base)
    return sorted(out[:n_designs])


def resolve(n_designs: int, override: dict | None = None) -> list[dict]:
    """Return the per-design schedules for a job. `override` (from Advanced mode)
    may pin mask_rate/gamma/target_mut_frac, applied on top of each spanned level."""
    levels = span_levels(n_designs)
    out = []
    for lv in levels:
        s = schedule(lv)
        if override:
            for k in ("mask_rate", "gamma", "target_mut_frac"):
                if override.get(k) is not None:
                    s[k] = override[k]
            s["overridden"] = True
        out.append(s)
    return out


def gibbs_stop_rule() -> dict:
    """The convergence criteria the sampler uses (documented, not a user param).
    Sampling stops at whichever fires FIRST."""
    return dict(
        max_iters=MAX_GIBBS_ITERS,
        plateau_window=PLATEAU_WINDOW,
        plateau_tol=PLATEAU_TOL,
        accept_collapse=ACCEPT_COLLAPSE,
        criteria=[
            "classifier-score plateau (delta < plateau_tol over plateau_window passes)",
            "mutation budget reached (target_mut_frac of mutable residues changed)",
            "acceptance collapse (accepted-move fraction < accept_collapse)",
            "hard cap (max_iters passes)",
        ],
    )
