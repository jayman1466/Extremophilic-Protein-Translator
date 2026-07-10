"""Conservation-weighted masking for domain-adaptive MLM (design doc Section 13).

The masking probability at position ``i`` is NOT uniform. It is shaped by
per-position conservation ``c_i`` (in ``[0, 1]``; 1 = fully conserved) and an
aggressiveness exponent ``gamma``:

    P(mask position i)  proportional to  (1 - c_i) ** gamma

- Highly conserved positions (c_i -> 1) are rarely masked (they carry the
  structural / catalytic constraints).
- Variable positions (c_i -> 0) are preferentially masked (evolution already
  tolerates change there, so they are the safe places to push toward
  extremophilic statistics).

Active-site / frozen positions are the ``gamma -> inf`` limit of the same
mechanism: their mask weight is hard-zeroed. This gives a single mask function
that implements both the hard freeze (Stage A) and the soft graded prior
(Stage B).

``gamma`` is the principled 4th aggressiveness knob (Section 9): low gamma masks
broadly (aggressive redesign), high gamma restricts edits to the least-conserved
positions (conservative).

All functions are framework-light (numpy) so they are unit-testable without a
GPU or model weights; ``sample_mask_positions`` accepts an optional numpy
Generator for reproducibility.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "mask_weights",
    "sample_mask_positions",
    "bert_mask_assignment",
    "contact_pairs_from_map",
    "make_span_units",
    "build_mask_units",
    "sample_mask_units",
]


def mask_weights(
    conservation: np.ndarray,
    gamma: float = 1.0,
    frozen: np.ndarray | None = None,
    special: np.ndarray | None = None,
) -> np.ndarray:
    """Per-position *relative* mask weights ``(1 - c_i) ** gamma``.

    Args:
        conservation: array ``c_i`` in ``[0, 1]`` per residue (length L).
        gamma: aggressiveness exponent >= 0. ``gamma=0`` -> uniform over
            non-frozen positions; larger gamma concentrates mass on variable
            positions.
        frozen: optional boolean array (length L). True = active-site / frozen;
            weight hard-zeroed (the gamma->inf limit). Section 13 Stage A.
        special: optional boolean array (length L). True = special token
            (CLS/EOS/pad); weight zeroed, never masked.

    Returns:
        Non-negative weights (length L); NOT normalised (a caller that needs a
        probability distribution divides by the sum). All-zero if every position
        is frozen/special.
    """
    c = np.clip(np.asarray(conservation, dtype=float), 0.0, 1.0)
    if gamma < 0:
        raise ValueError("gamma must be >= 0")
    w = np.power(1.0 - c, gamma)
    # (1-c)^0 == 1 even where c==1; that is the intended gamma=0 uniform case.
    if frozen is not None:
        w = np.where(np.asarray(frozen, dtype=bool), 0.0, w)
    if special is not None:
        w = np.where(np.asarray(special, dtype=bool), 0.0, w)
    w[~np.isfinite(w)] = 0.0
    return w


def sample_mask_positions(
    conservation: np.ndarray,
    mask_rate: float = 0.15,
    gamma: float = 1.0,
    frozen: np.ndarray | None = None,
    special: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Choose which positions to mask, ~``mask_rate`` fraction of maskable ones.

    Positions are drawn WITHOUT replacement with probability proportional to
    ``mask_weights``. The number selected is ``round(mask_rate * n_maskable)``
    where ``n_maskable`` counts positions with positive weight (i.e. excludes
    frozen/special). This keeps the effective mask budget stable regardless of
    how many positions are frozen.

    Returns:
        Boolean array (length L), True where the position should be masked.
    """
    rng = rng or np.random.default_rng()
    w = mask_weights(conservation, gamma=gamma, frozen=frozen, special=special)
    L = len(w)
    out = np.zeros(L, dtype=bool)
    maskable = np.where(w > 0)[0]
    if len(maskable) == 0:
        return out
    k = int(round(mask_rate * len(maskable)))
    k = max(0, min(k, len(maskable)))
    if k == 0:
        return out
    p = w[maskable] / w[maskable].sum()
    chosen = rng.choice(maskable, size=k, replace=False, p=p)
    out[chosen] = True
    return out


def bert_mask_assignment(
    masked: np.ndarray,
    rng: np.random.Generator | None = None,
    p_mask: float = 0.8,
    p_random: float = 0.1,
) -> dict[str, np.ndarray]:
    """Split masked positions into BERT's 80/10/10 replace / random / keep.

    Given the boolean ``masked`` array from ``sample_mask_positions`` (these are
    the positions that contribute to the MLM loss), assign each to one of three
    treatments (design doc Section 12, Loss 1):
      - ``p_mask`` (0.8): replace token with ``<mask>``,
      - ``p_random`` (0.1): replace with a random amino-acid token,
      - remainder (0.1): keep the original token.

    Returns dict of boolean arrays ``{"loss", "replace_mask", "replace_random",
    "keep"}``. ``loss`` == input ``masked`` (all masked positions score the
    loss); the other three partition it.
    """
    rng = rng or np.random.default_rng()
    masked = np.asarray(masked, dtype=bool)
    L = len(masked)
    idx = np.where(masked)[0]
    replace_mask = np.zeros(L, dtype=bool)
    replace_random = np.zeros(L, dtype=bool)
    keep = np.zeros(L, dtype=bool)
    if len(idx):
        u = rng.random(len(idx))
        replace_mask[idx[u < p_mask]] = True
        replace_random[idx[(u >= p_mask) & (u < p_mask + p_random)]] = True
        keep[idx[u >= p_mask + p_random]] = True
    return {"loss": masked, "replace_mask": replace_mask,
            "replace_random": replace_random, "keep": keep}


# ---------------------------------------------------------------------------
# Coupling-aware masking (design doc Section 15 workaround #1).
#
# i.i.d. masking rarely masks BOTH partners of a coupled feature (disulfide,
# salt bridge, local secondary structure) at once, so the model reconstructs one
# partner by copying the visible one and never adapts the JOINT distribution.
# Coupling-aware masking groups coupled positions into *units* and masks each
# unit as a whole, forcing joint reconstruction. Units come from:
#   - contact-pair mode: index pairs from ESM-2's own contact head (residues
#     that co-vary / are spatially close — where disulfides & salt bridges live);
#   - span mode: contiguous blocks (local secondary-structure elements).
# Everything here is numpy / framework-light and unit-testable without a GPU.
# ---------------------------------------------------------------------------


def contact_pairs_from_map(contacts: np.ndarray, threshold: float = 0.5,
                           min_sep: int = 6, top_k: int | None = None) -> list[tuple[int, int]]:
    """Extract coupled residue pairs from a contact-probability matrix.

    ``contacts`` is an ``L x L`` symmetric probability matrix (e.g. ESM-2's
    ``predict_contacts`` output, residue coordinates). A pair ``(i, j)`` with
    ``i < j`` is kept when ``contacts[i, j] >= threshold`` and ``j - i >=
    min_sep`` (skip trivial local i,i+1 contacts — those are covered by span
    mode). If ``top_k`` is given, keep only the ``top_k`` highest-probability
    pairs. Returned indices are in the SAME coordinate system as ``contacts``
    (residue coords); the caller offsets for special tokens.
    """
    c = np.asarray(contacts, dtype=float)
    L = c.shape[0]
    iu, ju = np.triu_indices(L, k=max(1, min_sep))
    probs = c[iu, ju]
    keep = probs >= threshold
    iu, ju, probs = iu[keep], ju[keep], probs[keep]
    if top_k is not None and len(probs) > top_k:
        order = np.argsort(probs)[::-1][:top_k]
        iu, ju = iu[order], ju[order]
    return [(int(i), int(j)) for i, j in zip(iu, ju)]


def make_span_units(L: int, span_len: int = 3, offset: int = 0) -> list[list[int]]:
    """Partition positions ``[offset, L)`` into consecutive blocks of ``span_len``.

    Each block is a masking unit (span mode). ``offset`` skips a leading special
    token (CLS) so blocks align to residues; the trailing special token (EOS) is
    handled by ``build_mask_units`` excluding it.
    """
    if span_len < 1:
        raise ValueError("span_len must be >= 1")
    return [list(range(s, min(s + span_len, L))) for s in range(offset, L, span_len)]


def build_mask_units(L: int, special: np.ndarray | None = None,
                     frozen: np.ndarray | None = None,
                     contact_pairs=None, spans=None) -> list[list[int]]:
    """Build the set of maskable units covering all non-excluded positions.

    Coupled groups (``contact_pairs`` and/or ``spans``, each an iterable of
    index iterables) are formed FIRST; a group contributes a unit only for its
    members that are neither excluded (frozen/special) nor already assigned, and
    only if >= 2 such members remain (a lone survivor is not a "couple"). Every
    remaining maskable position becomes its own singleton unit. A position lands
    in at most one unit.
    """
    excluded = np.zeros(L, dtype=bool)
    if special is not None:
        excluded |= np.asarray(special, dtype=bool)
    if frozen is not None:
        excluded |= np.asarray(frozen, dtype=bool)
    assigned = np.zeros(L, dtype=bool)
    units: list[list[int]] = []

    groups = list(contact_pairs or []) + list(spans or [])
    for grp in groups:
        members = [int(p) for p in grp if 0 <= int(p) < L and not excluded[int(p)]
                   and not assigned[int(p)]]
        if len(members) >= 2:
            for p in members:
                assigned[p] = True
            units.append(members)
    for p in range(L):
        if not excluded[p] and not assigned[p]:
            assigned[p] = True
            units.append([p])
    return units


def sample_mask_units(conservation: np.ndarray, units: list[list[int]],
                      mask_rate: float = 0.15, gamma: float = 1.0,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """Mask whole units until ~``mask_rate`` of maskable positions are covered.

    Units are drawn WITHOUT replacement with probability proportional to their
    mean conservation-derived weight (``mean_i (1 - c_i)^gamma`` over members),
    so a coupled pair/span is masked as a unit and variable regions are still
    preferred (Section 13 prior carries through). The position budget is
    ``round(mask_rate * n_maskable_positions)``; selection stops once the budget
    is reached (whole final unit included, so slight overshoot is possible).

    Returns a boolean array (length L), True where masked.
    """
    rng = rng or np.random.default_rng()
    c = np.clip(np.asarray(conservation, dtype=float), 0.0, 1.0)
    posw = np.power(1.0 - c, gamma)
    L = len(c)
    out = np.zeros(L, dtype=bool)
    if not units:
        return out
    uw = np.array([posw[u].mean() if len(u) else 0.0 for u in units])
    n_maskable = sum(len(u) for u in units)
    budget = int(round(mask_rate * n_maskable))
    if budget <= 0 or uw.sum() <= 0:
        return out
    # weighted sampling WITHOUT replacement via the exponential race
    # (Efraimidis-Spirakis): key_i = -ln(U_i)/w_i, ascending order. Zero-weight
    # units get key=+inf and sort last, so fully-conserved regions are only ever
    # masked if the budget exceeds all positive-weight positions.
    u_rand = rng.random(len(units))
    with np.errstate(divide="ignore"):
        keys = -np.log(u_rand) / uw
    order = np.argsort(keys)
    n = 0
    for idx in order:
        u = units[idx]
        if uw[idx] <= 0 and n > 0:
            break  # don't spend budget on zero-weight units unless nothing else
        out[u] = True
        n += len(u)
        if n >= budget:
            break
    return out
