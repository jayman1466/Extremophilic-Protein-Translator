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
