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


# ---- matched-pair co-loading (data plumbing, needs a tokenizer via transformers) ----

def test_build_pair_dataset_filters_by_class_and_split():
    pd_mod = pytest.importorskip("pandas")
    pytest.importorskip("transformers")
    import tempfile
    from transformers import EsmTokenizer
    from eptrans.modeling.data import build_pair_dataset, collate_pairs
    vocab = "<cls> <pad> <eos> <unk> L A G V S E R T I D P K Q N F Y M H W C X B U Z O . - <null_1> <mask>".split()
    td = tempfile.mkdtemp(); open(f"{td}/vocab.txt", "w").write("\n".join(vocab))
    tok = EsmTokenizer(f"{td}/vocab.txt")
    labeled = pd_mod.DataFrame({
        "tagged_id": ["E0~p", "M0~p", "E1~p", "M1~p", "E2~p", "M2~p"],
        "sequence": ["LAGV", "SERT", "IDPK", "QNFY", "MHWC", "LAGA"],
    })
    pairs = pd_mod.DataFrame({
        "class": ["thermophile", "thermophile", "halophile"],
        "ext_id": ["E0~p", "E1~p", "E2~p"],
        "outgroup_id": ["M0~p", "M1~p", "M2~p"],
        "ext_split": ["train", "val", "train"],
        "out_split": ["train", "val", "train"],
    })
    # thermophile + train -> only the E0/M0 pair (E1/M1 is val, E2/M2 is halophile)
    ds = build_pair_dataset(labeled, pairs, tok, "thermophile", "train", max_len=32)
    assert len(ds) == 1
    item = ds[0]
    assert set(item) == {"ext_input_ids", "ext_attention_mask",
                         "out_input_ids", "out_attention_mask"}
    batch = collate_pairs([ds[0]], pad_id=tok.pad_token_id)
    # ext and out padded independently; batch dim 1
    assert batch["ext_input_ids"].shape[0] == 1
    assert batch["out_input_ids"].shape[0] == 1


# ---- coupling-aware masking (Section 15 workaround #1) ----

def test_contact_pairs_threshold_and_minsep():
    L = 10
    c = np.zeros((L, L))
    c[0, 8] = c[8, 0] = 0.9   # long-range, above thresh -> kept
    c[0, 2] = c[2, 0] = 0.9   # sep 2 < min_sep 6 -> dropped
    c[1, 9] = c[9, 1] = 0.3   # below thresh -> dropped
    pairs = masking.contact_pairs_from_map(c, threshold=0.5, min_sep=6)
    assert pairs == [(0, 8)]


def test_contact_pairs_top_k():
    L = 20
    c = np.zeros((L, L))
    for k, (i, j, p) in enumerate([(0, 10, 0.9), (1, 11, 0.8), (2, 12, 0.7)]):
        c[i, j] = c[j, i] = p
    pairs = masking.contact_pairs_from_map(c, threshold=0.5, min_sep=6, top_k=2)
    assert set(pairs) == {(0, 10), (1, 11)}  # two highest


def test_make_span_units_partitions():
    units = masking.make_span_units(10, span_len=3, offset=1)
    flat = [p for u in units for p in u]
    assert flat == list(range(1, 10))  # covers 1..9, no gaps/dupes
    assert units[0] == [1, 2, 3]


def test_build_mask_units_couples_first_then_singletons():
    L = 12
    special = np.zeros(L, dtype=bool); special[0] = special[-1] = True
    pairs = [(2, 9)]  # one coupled pair
    units = masking.build_mask_units(L, special=special, contact_pairs=pairs)
    assert [2, 9] in units
    # every non-special position assigned exactly once
    flat = sorted(p for u in units for p in u)
    assert flat == list(range(1, 11))


def test_build_mask_units_respects_frozen_and_lone_survivor():
    L = 10
    frozen = np.zeros(L, dtype=bool); frozen[5] = True
    # pair (5,8): 5 is frozen -> only 8 survives -> not a couple, becomes singleton
    units = masking.build_mask_units(L, frozen=frozen, contact_pairs=[(5, 8)])
    assert [5, 8] not in units and [8] in units
    assert not any(5 in u for u in units)  # frozen never masked


def test_sample_mask_units_masks_whole_units():
    L = 30
    units = [[i, i + 15] for i in range(3)] + [[j] for j in range(6, 15)]
    cons = np.zeros(L)
    m = masking.sample_mask_units(cons, units, mask_rate=0.3,
                                  rng=np.random.default_rng(0))
    # any masked coupled unit is masked as a whole
    for u in units:
        masked_members = [p for p in u if m[p]]
        assert len(masked_members) in (0, len(u))


def test_sample_mask_units_prefers_variable_regions():
    L = 40
    units = [[i] for i in range(L)]
    cons = np.zeros(L); cons[:20] = 1.0  # first half fully conserved
    m = masking.sample_mask_units(cons, units, mask_rate=0.25, gamma=1.0,
                                  rng=np.random.default_rng(1))
    assert m[20:].sum() > m[:20].sum()  # variable half masked more


# ---- truncation guard ----

def test_sliding_windows_short_seq_single_window():
    from eptrans.modeling.data import sliding_windows
    w = sliding_windows("ACDEFG", max_len=10)
    assert w == [(0, "ACDEFG")]


def test_sliding_windows_long_seq_overlaps_and_covers():
    from eptrans.modeling.data import sliding_windows
    seq = "".join("ACDEFGHIKL"[i % 10] for i in range(50))
    w = sliding_windows(seq, max_len=20, overlap=5)
    # windows cover the whole sequence; consecutive windows overlap
    assert w[0][0] == 0
    assert w[-1][0] + len(w[-1][1]) == len(seq)  # last window reaches the end
    starts = [s for s, _ in w]
    assert all(starts[i+1] - starts[i] == 15 for i in range(len(starts)-1))  # step = 20-5


# ---- negative-sampling cap + length bucketing (H200 wall-clock levers) ----

def test_classifier_negative_cap_bounds_negatives():
    pd_mod = pytest.importorskip("pandas")
    pytest.importorskip("transformers")
    import tempfile
    from transformers import EsmTokenizer
    from eptrans.modeling.data import build_classifier_dataset
    vocab = "<cls> <pad> <eos> <unk> L A G V S E R T I D P K Q N F Y M H W C X B U Z O . - <null_1> <mask>".split()
    td = tempfile.mkdtemp(); open(f"{td}/vocab.txt", "w").write("\n".join(vocab))
    tok = EsmTokenizer(f"{td}/vocab.txt")
    # 5 positives, 100 negatives
    rows = [{"tagged_id": f"P{i}~p", "sequence": "LAGV", "label": "thermophile",
             "label_confidence": "high", "split": "train"} for i in range(5)]
    rows += [{"tagged_id": f"N{i}~p", "sequence": "SERT", "label": "mesophile",
              "label_confidence": "none", "split": "train"} for i in range(100)]
    df = pd_mod.DataFrame(rows)
    ds = build_classifier_dataset(df, tok, "thermophile", "train", max_len=32, neg_per_pos=3.0)
    # 5 pos + 15 neg = 20
    assert len(ds) == 20
    # uncapped keeps all 105
    ds_all = build_classifier_dataset(df, tok, "thermophile", "train", max_len=32, neg_per_pos=None)
    assert len(ds_all) == 105


def test_length_bucket_sampler_covers_all_and_batches():
    from eptrans.modeling.train import LengthBucketSampler
    lengths = [10, 500, 12, 480, 11, 490] * 5  # 30 items
    s = LengthBucketSampler(lengths, batch_size=4, pool_mult=2, seed=0)
    batches = list(iter(s))
    flat = sorted(i for b in batches for i in b)
    assert flat == list(range(30))          # every item exactly once
    assert len(s) == 8                       # ceil(30/4)
    # within-pool sorting: at least one batch is length-homogeneous
    assert any(max(lengths[i] for i in b) - min(lengths[i] for i in b) < 5 for b in batches)


def test_mlm_adapter_transfers_into_classifier():
    """The MLM adapter (EsmForMaskedLM, esm.-prefixed keys) must actually load
    into the classifier backbone (EsmModel). Plain load_adapter silently drops
    the mismatched keys; load_mlm_adapter_into_classifier remaps + verifies."""
    import tempfile, torch
    from transformers import EsmConfig, EsmForMaskedLM, EsmModel
    from peft import LoraConfig, get_peft_model, TaskType
    from eptrans.modeling.model import lora_target_modules, load_mlm_adapter_into_classifier
    pytest.importorskip("safetensors")
    tmp = tempfile.mkdtemp()
    vocab = "<cls> <pad> <eos> <unk> L A G V S E R T I D P K Q N F Y M H W C X B U Z O . - <null_1> <mask>".split()
    cfg = dict(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
               intermediate_size=64, max_position_embeddings=64, pad_token_id=1,
               mask_token_id=len(vocab) - 1, position_embedding_type="rotary")
    tgt = lora_target_modules(True)
    lc = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=4, lora_alpha=8,
                    lora_dropout=0.0, target_modules=tgt, bias="none")
    mlm = get_peft_model(EsmForMaskedLM(EsmConfig(**cfg)), lc)
    # tweak a LoRA weight so transfer is detectable
    tw = [n for n, _ in mlm.named_parameters() if "lora_A" in n][0]
    with torch.no_grad():
        dict(mlm.named_parameters())[tw].add_(1.234)
    mlm.save_pretrained(f"{tmp}/mlm_adapter_best")
    clf = get_peft_model(EsmModel(EsmConfig(**cfg), add_pooling_layer=False), lc)
    n = load_mlm_adapter_into_classifier(clf, f"{tmp}/mlm_adapter_best", adapter_name="mlm")
    assert n > 0
    clf.set_adapter("mlm")
    maxv = max(float(p.detach().abs().max()) for nm, p in clf.named_parameters()
               if "lora_A" in nm and "mlm" in nm)
    assert maxv >= 1.234, f"trained weight not transferred (max {maxv:.3f})"


def test_precomputed_contact_pairs_consumed():
    """build_mlm_dataset consumes cached residue-coord pairs via contact_pairs_col
    instead of recomputing, and remaps them to token coords (+1 for CLS)."""
    import numpy as np, pandas as pd
    from transformers import EsmTokenizer
    from eptrans.modeling.data import build_mlm_dataset
    import tempfile
    tmp = tempfile.mkdtemp()
    vocab = "<cls> <pad> <eos> <unk> L A G V S E R T I D P K Q N F Y M H W C X B U Z O . - <null_1> <mask>".split()
    open(f"{tmp}/vocab.txt", "w").write("\n".join(vocab))
    tok = EsmTokenizer(f"{tmp}/vocab.txt")
    seq = "LAGVSERTIDPKQNFLAGVSE"  # 21 residues, single window at max_len=64
    # residue-coord pair (2, 12): distal, > min_sep; should mask jointly
    df = pd.DataFrame([dict(tagged_id="g0~p", sequence=seq, split="train",
                            label_confidence="high", contact_pairs=[[2, 12]])])
    ds = build_mlm_dataset(df, tok, "train", max_len=64, mask_rate=0.15,
                           coupling_mode="contact", contact_pairs_col="contact_pairs",
                           seed=1)
    # sample many times; whenever one partner is in the loss set, the other must be too
    both_seen = False
    for _ in range(200):
        item = ds[0]
        loss_pos = set(np.where(item["labels"].numpy() != -100)[0].tolist())
        # token coords: residue 2 -> 3, residue 12 -> 13 (CLS offset +1)
        a, b = 3, 13
        if a in loss_pos or b in loss_pos:
            assert a in loss_pos and b in loss_pos, "coupled pair must mask jointly"
            both_seen = True
    assert both_seen, "the cached contact pair was never selected across 200 draws"


# ---- spot-safe step checkpointing ----

def _tiny_mlm_setup(tmp):
    import numpy as np, torch, pandas as pd
    from transformers import EsmConfig, EsmForMaskedLM, EsmTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    from eptrans.modeling.model import lora_target_modules
    from eptrans.modeling.data import build_mlm_dataset
    vocab = "<cls> <pad> <eos> <unk> L A G V S E R T I D P K Q N F Y M H W C X B U Z O . - <null_1> <mask>".split()
    open(f"{tmp}/vocab.txt", "w").write("\n".join(vocab))
    tok = EsmTokenizer(f"{tmp}/vocab.txt")
    def mk():
        cfg = EsmConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2,
                        num_attention_heads=2, intermediate_size=64, max_position_embeddings=256,
                        pad_token_id=1, mask_token_id=len(vocab) - 1, position_embedding_type="rotary")
        torch.manual_seed(0)
        return get_peft_model(EsmForMaskedLM(cfg), LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, r=4, lora_alpha=8, lora_dropout=0.0,
            target_modules=lora_target_modules(True), bias="none"))
    rng = np.random.default_rng(0)
    df = pd.DataFrame([dict(tagged_id=f"g{i}~p",
                            sequence="".join(rng.choice(list("LAGVSERTIDPKQNF"), rng.integers(15, 60))),
                            split="train", label_confidence="high") for i in range(40)])
    return tok, mk, build_mlm_dataset(df, tok, "train", max_len=64, mask_rate=0.15)


def test_mlm_step_checkpoint_and_resume():
    import tempfile, os, torch
    from eptrans.modeling.train import train_mlm
    pytest.importorskip("peft")
    tmp = tempfile.mkdtemp()
    tok, mk, tr = _tiny_mlm_setup(tmp)
    out = tempfile.mkdtemp()
    # 40 items / batch 8 = 5 steps/epoch, 3 epochs => 15 total steps.
    # phase 1: run to a MID-epoch checkpoint (step 6 = epoch 1, batch 1), ckpt every 3.
    train_mlm(mk(), tok, tr, epochs=3, batch_size=8, device="cpu",
              out_dir=out, ckpt_every=3, max_steps=6, resume=False)
    ck = torch.load(f"{out}/mlm_ckpt.pt", map_location="cpu")
    assert ck["step"] == 6 and len(ck["trainable"]) > 0
    assert "opt" in ck and "sched" in ck
    # checkpoint records mid-epoch position, not just epoch granularity
    assert ck["epoch"] == 1 and ck["batch_in_epoch"] == 1   # 5 in ep0 + 1 in ep1
    # phase 2: resume to completion. step is DERIVED from (epoch, batch), so the
    # final step must be exactly the schedule total (15), NOT overshoot it.
    train_mlm(mk(), tok, tr, epochs=3, batch_size=8, device="cpu",
              out_dir=out, ckpt_every=3, resume=True)
    ck2 = torch.load(f"{out}/mlm_ckpt.pt", map_location="cpu")
    # last ckpt-divisible step before the 15-step end; must never exceed total (15)
    assert ck2["step"] <= 15, f"resume overshot schedule total: {ck2['step']}"
    assert ck2["step"] == 15                                  # 5*3 epochs, exact
