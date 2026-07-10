"""Unit tests for the modeling scaffold — the framework-light, scientifically
load-bearing pieces (masking §13, loss weighting §12) that must be correct
regardless of GPU/model availability."""
import numpy as np
import pytest

from eptrans.modeling import masking
from eptrans.modeling.losses import confidence_to_weight, CONFIDENCE_WEIGHTS


# ---- §13 conservation-weighted masking ----

def test_mask_weights_conserved_never_variable_always():
    cons = np.array([0.0, 0.5, 1.0])
    w = masking.mask_weights(cons, gamma=1.0)
    # (1-c)^1 = [1.0, 0.5, 0.0]
    assert np.allclose(w, [1.0, 0.5, 0.0])


def test_gamma_zero_is_uniform_over_nonfrozen():
    cons = np.array([0.1, 0.6, 0.9])
    w = masking.mask_weights(cons, gamma=0.0)
    assert np.allclose(w, 1.0)  # (1-c)^0 == 1 everywhere


def test_higher_gamma_concentrates_on_variable():
    cons = np.array([0.2, 0.8])
    w1 = masking.mask_weights(cons, gamma=1.0)
    w4 = masking.mask_weights(cons, gamma=4.0)
    # ratio variable/conserved grows with gamma
    assert (w4[0] / w4[1]) > (w1[0] / w1[1])


def test_frozen_is_hard_zero_gamma_inf_limit():
    cons = np.array([0.0, 0.0, 0.0])
    frozen = np.array([False, True, False])
    w = masking.mask_weights(cons, gamma=1.0, frozen=frozen)
    assert w[1] == 0.0 and w[0] > 0 and w[2] > 0


def test_special_tokens_never_masked():
    cons = np.zeros(5)
    special = np.array([True, False, False, False, True])  # CLS ... EOS
    m = masking.sample_mask_positions(cons, mask_rate=1.0, special=special,
                                      rng=np.random.default_rng(0))
    assert not m[0] and not m[-1]


def test_frozen_positions_never_sampled():
    cons = np.zeros(20)
    frozen = np.zeros(20, dtype=bool); frozen[5:10] = True
    m = masking.sample_mask_positions(cons, mask_rate=0.5, frozen=frozen,
                                      rng=np.random.default_rng(1))
    assert not m[5:10].any()


def test_mask_rate_budget_over_maskable():
    cons = np.zeros(100)
    frozen = np.zeros(100, dtype=bool); frozen[:50] = True  # 50 maskable
    m = masking.sample_mask_positions(cons, mask_rate=0.2, frozen=frozen,
                                      rng=np.random.default_rng(2))
    assert m.sum() == 10  # 0.2 * 50


def test_bert_assignment_partitions_masked_set():
    masked = np.zeros(1000, dtype=bool); masked[::2] = True  # 500 masked
    a = masking.bert_mask_assignment(masked, rng=np.random.default_rng(3))
    # loss set == input; three treatments partition it exactly
    assert np.array_equal(a["loss"], masked)
    union = a["replace_mask"] | a["replace_random"] | a["keep"]
    assert np.array_equal(union, masked)
    assert not (a["replace_mask"] & a["replace_random"]).any()
    # ~80/10/10 (loose bounds for randomness)
    assert 0.72 < a["replace_mask"].sum() / 500 < 0.88
    assert 0.04 < a["replace_random"].sum() / 500 < 0.18


# ---- §12 confidence weighting ----

def test_confidence_to_weight_tiers():
    assert confidence_to_weight("high") == 1.0
    assert confidence_to_weight("medium") == 0.5
    assert confidence_to_weight("none") == 1.0  # mesophile negatives full weight
    assert confidence_to_weight("nonsense", default=0.7) == 0.7
    assert confidence_to_weight(None) == 1.0


# ---- torch-dependent loss math (skip if torch missing) ----
torch = pytest.importorskip("torch")
from eptrans.modeling import losses as L


def test_margin_loss_zero_when_ext_beats_out_by_margin():
    se = torch.tensor([2.0, 3.0]); so = torch.tensor([0.5, 1.0])
    # diffs 1.5, 2.0 both >= margin 1.0 -> zero hinge
    assert float(L.matched_pair_margin_loss(se, so, margin=1.0)) == 0.0


def test_margin_loss_positive_when_ext_below_out():
    se = torch.tensor([0.0]); so = torch.tensor([1.0])
    # max(0, 1 - (0-1)) = 2.0
    assert abs(float(L.matched_pair_margin_loss(se, so, margin=1.0)) - 2.0) < 1e-6


def test_weighted_bce_reduces_to_mean_when_unweighted():
    import torch.nn.functional as F
    s = torch.tensor([0.3, -1.2, 2.0]); y = torch.tensor([1.0, 0.0, 1.0])
    ref = F.binary_cross_entropy_with_logits(s, y)
    got = L.weighted_bce_loss(s, y)
    assert abs(float(ref) - float(got)) < 1e-6


def test_sample_weight_upweights_high_confidence():
    import torch.nn.functional as F
    # two examples with DIFFERENT per-example loss: idx0 nearly correct (tiny
    # loss), idx1 badly wrong (large loss).
    s = torch.tensor([5.0, -5.0]); y = torch.tensor([1.0, 1.0])
    per = F.binary_cross_entropy_with_logits(s, y, reduction="none")
    assert float(per[1]) > float(per[0])  # sanity: idx1 is the hard one
    # weighting the hard example -> weighted mean == its per-example loss; the
    # two weightings must DIFFER, and up-weighting the hard one gives more loss.
    w_easy = L.weighted_bce_loss(s, y, sample_weight=torch.tensor([1.0, 0.0]))
    w_hard = L.weighted_bce_loss(s, y, sample_weight=torch.tensor([0.0, 1.0]))
    assert float(w_hard) > float(w_easy)
    assert abs(float(w_easy) - float(per[0])) < 1e-5   # picks out idx0's loss
    assert abs(float(w_hard) - float(per[1])) < 1e-5   # picks out idx1's loss


def test_masked_mlm_loss_ignores_unmasked():
    B, Ln, V = 1, 4, 6
    logits = torch.zeros(B, Ln, V)
    logits[0, 0, 3] = 10.0  # confident correct at pos 0
    targets = torch.tensor([[3, 0, 0, 0]])
    mask_all = torch.tensor([[True, True, True, True]])
    mask_one = torch.tensor([[True, False, False, False]])
    # scoring only the confident position -> lower loss than scoring all
    assert float(L.masked_mlm_loss(logits, targets, mask_one)) < \
           float(L.masked_mlm_loss(logits, targets, mask_all))


def test_classifier_loss_combines_terms():
    s = torch.tensor([1.0, -1.0]); y = torch.tensor([1.0, 0.0])
    tot, parts = L.classifier_loss(s, y, pair_ext=s[:1], pair_out=s[1:],
                                   lam=1.0, margin=1.0)
    # total == bce + 1.0*pair
    assert abs(float(tot) - (parts["bce"] + parts["pair"])) < 1e-5
