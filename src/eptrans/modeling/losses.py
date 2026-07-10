"""Training losses (design doc Section 12).

Three scoring objects; only two are training losses. The generation-time
composite (Section 10) is an inference gate, not a differentiable loss, and
lives elsewhere.

Loss 1 - domain-adaptive MLM (shared backbone):
    L_MLM = -(1/|M|) Σ_{i∈M} w_seq · log p_θ(x_i | x_\\M)
  confidence-weighted masked cross-entropy; w_seq = label_confidence.
  Optional forgetting-guard + β·KL(p_θ ‖ p_base) toward base ESM-2.

Loss 2 - per-phenotype classifier (matched-pair aware):
    L_cls = L_BCE + λ · L_pair
  L_BCE   = weighted BCE (pos_weight or focal), w_i = label_confidence;
  L_pair  = Σ margin ranking on each extremophile e vs matched outgroup m,
            max(0, δ - (s_e - s_m)) — forces score(extremo) > score(mesophile).

torch is imported lazily so the module imports on a CPU-only / torch-less box;
the functions themselves need torch. ``confidence_to_weight`` (string ->
float) is pure-python and always available.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = [
    "CONFIDENCE_WEIGHTS",
    "confidence_to_weight",
    "masked_mlm_loss",
    "kl_forgetting_guard",
    "weighted_bce_loss",
    "focal_bce_loss",
    "matched_pair_margin_loss",
    "classifier_loss",
]

# Map genome-confidence tier -> sample weight (w_seq / w_i). Mesophiles carry
# 'none' (label stamped from a confident-mesophile genome, full weight for the
# negative class). Tunable; these are the starting values.
CONFIDENCE_WEIGHTS = {"high": 1.0, "medium": 0.5, "none": 1.0, "low": 0.25}


def confidence_to_weight(conf, default: float = 1.0) -> float:
    """Map a confidence tier string to its sample weight (pure-python)."""
    if conf is None:
        return default
    return CONFIDENCE_WEIGHTS.get(str(conf).lower(), default)


def masked_mlm_loss(logits, targets, mask, seq_weight=None, reduction: str = "mean"):
    """Confidence-weighted masked cross-entropy (Loss 1).

    Args:
        logits: ``(B, L, V)`` MLM logits.
        targets: ``(B, L)`` gold token ids (ignored where ``mask`` is False).
        mask: ``(B, L)`` bool; True where the position contributes to the loss
            (the BERT 'loss' set from masking.bert_mask_assignment).
        seq_weight: optional ``(B,)`` per-sequence weight w_seq (label
            confidence). Broadcast over the sequence's masked positions.
        reduction: 'mean' (over masked positions, weighted) or 'sum'.

    Returns:
        Scalar tensor.
    """
    import torch
    import torch.nn.functional as F

    B, L, V = logits.shape
    flat_logits = logits.reshape(-1, V)
    flat_targets = targets.reshape(-1)
    ce = F.cross_entropy(flat_logits, flat_targets, reduction="none").reshape(B, L)
    m = mask.to(ce.dtype)
    if seq_weight is not None:
        m = m * seq_weight.reshape(B, 1).to(ce.dtype)
    num = (ce * m).sum()
    if reduction == "sum":
        return num
    denom = m.sum().clamp_min(1.0)
    return num / denom


def kl_forgetting_guard(logits, base_logits, mask=None):
    """KL(p_θ ‖ p_base) forgetting-guard toward the base model (Loss 1 option).

    Start β=0; add only if adapted fills look unnatural. Averaged over masked
    positions if ``mask`` given, else over all positions.
    """
    import torch
    import torch.nn.functional as F

    logp = F.log_softmax(logits, dim=-1)
    logq = F.log_softmax(base_logits, dim=-1)
    kl = (logp.exp() * (logp - logq)).sum(-1)  # (B, L)
    if mask is not None:
        m = mask.to(kl.dtype)
        return (kl * m).sum() / m.sum().clamp_min(1.0)
    return kl.mean()


def weighted_bce_loss(scores, labels, sample_weight=None, pos_weight=None):
    """Weighted BCE-with-logits (Loss 2 pointwise term).

    Args:
        scores: ``(N,)`` raw logits s_i.
        labels: ``(N,)`` in {0,1}.
        sample_weight: optional ``(N,)`` w_i (label_confidence).
        pos_weight: optional scalar tensor for class imbalance (weight on the
            positive term), as in ``F.binary_cross_entropy_with_logits``.
    """
    import torch
    import torch.nn.functional as F

    per = F.binary_cross_entropy_with_logits(
        scores, labels.to(scores.dtype),
        pos_weight=pos_weight, reduction="none")
    if sample_weight is not None:
        per = per * sample_weight.to(per.dtype)
        return per.sum() / sample_weight.to(per.dtype).sum().clamp_min(1.0)
    return per.mean()


def focal_bce_loss(scores, labels, gamma: float = 2.0, sample_weight=None):
    """Focal BCE (Loss 2 pointwise alternative for heavy imbalance).

    ``(1 - p_t)^gamma`` down-weights easy examples.
    """
    import torch
    import torch.nn.functional as F

    labels = labels.to(scores.dtype)
    p = torch.sigmoid(scores)
    p_t = p * labels + (1 - p) * (1 - labels)
    ce = F.binary_cross_entropy_with_logits(scores, labels, reduction="none")
    loss = ((1 - p_t) ** gamma) * ce
    if sample_weight is not None:
        loss = loss * sample_weight.to(loss.dtype)
        return loss.sum() / sample_weight.to(loss.dtype).sum().clamp_min(1.0)
    return loss.mean()


def matched_pair_margin_loss(score_ext, score_out, margin: float = 1.0):
    """Margin ranking on matched pairs (Loss 2 pairwise term).

    ``L_pair = mean_j max(0, margin - (s_ext_j - s_out_j))`` — forces the
    extremophile score above its matched outgroup by at least ``margin``. This
    is the loss-function embodiment of the outgroup design: pushes the
    classifier onto the phenotype delta, not clade features.

    Args:
        score_ext: ``(P,)`` logits for the extremophile proteins.
        score_out: ``(P,)`` logits for their matched outgroup proteins
            (positionally aligned).
    """
    import torch

    diff = score_ext - score_out
    return torch.clamp(margin - diff, min=0.0).mean()


def classifier_loss(scores, labels, sample_weight=None, pos_weight=None,
                    pair_ext=None, pair_out=None, lam: float = 1.0,
                    margin: float = 1.0, use_focal: bool = False,
                    focal_gamma: float = 2.0):
    """Combined per-phenotype classifier loss ``L_cls = L_BCE + λ · L_pair``.

    ``pair_ext`` / ``pair_out`` are aligned logit tensors for matched pairs
    (may be None to disable the pairwise term, e.g. a batch with no pairs).
    Returns ``(total, {"bce":..., "pair":...})`` for logging.
    """
    import torch

    if use_focal:
        l_point = focal_bce_loss(scores, labels, gamma=focal_gamma,
                                 sample_weight=sample_weight)
    else:
        l_point = weighted_bce_loss(scores, labels, sample_weight=sample_weight,
                                    pos_weight=pos_weight)
    if pair_ext is not None and pair_out is not None and len(pair_ext) > 0:
        l_pair = matched_pair_margin_loss(pair_ext, pair_out, margin=margin)
    else:
        l_pair = torch.zeros((), dtype=scores.dtype, device=scores.device)
    total = l_point + lam * l_pair
    return total, {"bce": float(l_point.detach()), "pair": float(l_pair.detach())}
