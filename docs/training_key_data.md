# Key data used in adapter & classifier training

Collated from the modeling code (`src/eptrans/modeling/`), the training scripts
(`scripts/08_train_backbone.py`, `scripts/10_train_cached_probe.py`,
`scripts/10_precompute_contacts.py`), the design doc (`docs/modeling_design.md`
§12–15), and the run records in `labnotebook.md`.

There are **two trained objects** and (for the classifier) **two training
paths**:

| object | script / fn | what trains | actually run? |
|---|---|---|---|
| **Adapter** (domain-adaptive MLM) | `08_train_backbone.py mlm` → `train_mlm` | LoRA deltas on frozen ESM-2 3B | yes — `both`-mode, 420k subsample, 3 epochs |
| **Classifier, end-to-end** | `08_train_backbone.py classifier` → `train_classifier` | LoRA backbone + MLP head | scaffolded; projected ~27 d/epoch, not used at scale |
| **Classifier, cached probe** | `10_train_cached_probe.py` | MLP head on frozen cached embeddings | yes — 5 phenotypes, 30 epochs, ~4m45s |

Every formula below is written to match the code, not just the design doc.

---

## A. The adapter (domain-adaptive MLM)

Continued masked-language-model fine-tuning of ESM-2 3B (`facebook/esm2_t36_3B_UR50D`)
with a LoRA adapter (base frozen, **rank 32 / α 64** in the run — the sbatch
passes `--lora-rank 32 --lora-alpha 64`, matching `model.py`'s default; the
rank was bumped 16→32 for coupling capacity). Trains on TRAIN-split clusters only (leakage rule). One entry point:
`train_mlm` in `src/eptrans/modeling/train.py`.

### A1. How ESM attention maps were used to mask residues

**Plain language.** Ordinary BERT masking is i.i.d.: it rarely masks *both*
partners of a coupled feature (disulfide, salt bridge, secondary-structure
element) at once, so the model reconstructs one partner by copying the visible
one and never adapts the *joint* distribution. To fix this, residues are grouped
into **masking units** that are masked as a whole. The **contact** units come
from ESM-2's own contact head, which is a logistic regression over the model's
**symmetrized row-attention maps** stacked across layers and heads
(`predict_contacts`; `10_precompute_contacts.py` notes the maps only
materialize under `attn_implementation="eager"`). So the attention maps are the
substrate: attention → contact probabilities → coupled pairs → joint masking
units. The conservation prior still steers *which* units are chosen.

**The pipeline (mechanics).**
1. `predict_contacts` yields an `L×L` contact-probability matrix `C` from the
   attention maps.
2. `contact_pairs_from_map` keeps pair `(i,j)`, `i<j`, when
   `C[i,j] ≥ contact_threshold` (0.5) **and** `j − i ≥ contact_min_sep` (6, to
   skip trivial `i,i+1` contacts that span mode already covers), then keeps the
   `top_k` (128) highest-probability pairs. Contacts are **precomputed once** on
   GPU per sequence (`10_precompute_contacts.py`) and cached
   (`contact_pairs.parquet`, full-sequence residue coords) — deriving them
   inside the DataLoader would rerun a 3B forward pass per item per epoch.
3. `build_mask_units` forms units: each contact pair (and each `span_len`=3
   contiguous block, in `both` mode) becomes a unit if ≥2 of its members are
   maskable; every remaining position is a singleton unit. Special tokens
   (CLS/EOS/pad) and `frozen` (active-site) positions are excluded.
4. `sample_mask_units` draws whole units **without replacement** until
   `≈ mask_rate` (0.15) of maskable positions are covered, with per-unit
   probability proportional to the unit's mean conservation weight, using the
   Efraimidis–Spirakis exponential race.

**Formula — unit sampling weight.** For a unit `U` with per-residue
conservation `c_i`,

    w(U) = mean_{i∈U} (1 − c_i)^γ

Selection key (exponential race, ascending): `key_U = −ln(U_rand) / w(U)`; take
units in increasing key order until the position budget
`round(mask_rate · n_maskable)` is met.

**Run note.** The `both`-mode run used `gamma = 1.0` but with **no MSA**, so
per-position conservation defaults to `c_i = 0`; `(1−c)^γ → 1` and the *choice
of which unit* is uniform. Attention-derived coupling was therefore the *active*
structural signal in the adapter run; the conservation prior is inert during
enzyme-agnostic training and only becomes live in the per-enzyme generation
loop.

### A1b. Active-site freeze (the γ→∞ limit)

Per-position mask weight is `(1 − c_i)^γ`, hard-zeroed at `frozen` and `special`
positions. Freezing an active-site residue is the `γ→∞` limit of the same
mechanism — one mask function implements both the hard freeze and the soft
graded prior. (No frozen set is supplied during enzyme-agnostic backbone
training; it is a generation-time input.)

### A2. High vs. medium confidence weighting (adapter)

**Plain language.** Each protein inherits a genome-level phenotype-label
confidence tier. Cleaner genomes drive the adaptation more: the per-position MLM
cross-entropy of a sequence is multiplied by a per-sequence weight equal to that
tier's value. **High-confidence proteins carry full weight (1.0); medium-confidence
carry half (0.5).**

**Formula — tier→weight map** (`losses.CONFIDENCE_WEIGHTS`,
`confidence_to_weight`):

    w_seq = { high: 1.0,  medium: 0.5,  none: 1.0,  low: 0.25 }

`none` = a confident mesophile label (full weight for the negative class);
`low` = weakest tier. Applied as `seq_weight` broadcast over that sequence's
masked positions in `data.build_mlm_dataset`.

### A3. Other weightings (adapter)

- **BERT 80/10/10** over masked positions (`bert_mask_assignment`): of the
  chosen positions, 80% → `<mask>`, 10% → random amino acid, 10% → kept. All
  chosen positions score the loss.
- **Mask budget normalization** (`sample_mask_positions` / `sample_mask_units`):
  the number masked is `round(mask_rate · n_maskable)` over *maskable* positions
  only, so freezing positions does not shrink the effective 15% budget.
- **Optional KL forgetting-guard** weight `β` (`beta_kl`, default **0** — off in
  the run): pulls the adapted distribution toward base ESM-2.
- **LengthBucketSampler** groups similar-length sequences per batch to cut
  padding waste (~2–3× fewer tokens) — a throughput device, not a loss weight.

### A4. Loss function (adapter)

**Plain language.** Confidence-weighted masked cross-entropy: average, over the
masked positions, of the negative log-likelihood of the true residue given the
unmasked context, each sequence's contribution scaled by its confidence weight.
Optionally add `β·KL` toward base ESM-2.

**Formula** (`losses.masked_mlm_loss`, design §12 Loss 1):

    L_MLM = − (1 / Σ_i m_i) · Σ_{i∈M} w_seq · log p_θ(x_i | x_\M)

where `M` is the masked (loss) set, `m_i` the weighted mask, `w_seq` the
sequence confidence weight. With the guard:

    L = L_MLM + β · KL(p_θ ‖ p_base),   β = 0 in the run.

Model selection / early stop: **val pseudo-perplexity** `exp(mean masked CE)`
(unweighted). Optimizer AdamW, lr 1e-4, warmup 5% then linear decay, bf16.

---

## B. The classifier (per-phenotype extremophile head)

A mean-pooled 2-layer MLP head producing one logit per protein, one head per
phenotype (acidophile / alkaliphile / halophile / thermophile /
hyperthermophile). y=1 if the protein carries that phenotype, y=0 for matched
mesophiles; other extremophiles are dropped (per-phenotype-vs-mesophile
contrast). Two paths share the same loss (`losses.classifier_loss`).

### B1. Attention maps in the classifier

**None directly.** The classifier is discriminative (no masking, no
reconstruction), so it uses no attention-derived masking of its own. The
attention-map coupling signal enters **only through the inherited MLM adapter**:

- **End-to-end path** (`08 classifier`) branches from the Stage-1 MLM adapter
  via `load_mlm_adapter_into_classifier` (remaps `esm.`-prefixed keys and
  **raises if 0 tensors transfer** — a real bug that silently trained from a
  random adapter until caught; verified 288 tensors transferred). LoRA still
  adapts the **full attention pathway** (`query`/`key`/`value` +
  `attention.output.dense`; FFN dense excluded) so fine-tuning can re-route
  which residues attend to which.
- **Cached-probe path** (`10_train_cached_probe`, the one run at scale) trains
  only the MLP on embeddings pre-computed once through the frozen MLM-adapted
  backbone — so the coupling adaptation is baked into the fixed features.

### B2. High vs. medium confidence weighting (classifier)

**Plain language.** Same tier→weight map as the adapter, applied as a
per-example weight `w_i` on the BCE term — **high = 1.0, medium = 0.5**, none =
1.0, low = 0.25.

**Formula.** `w_i = confidence_to_weight(label_confidence)`, same
`CONFIDENCE_WEIGHTS` dict as A2, multiplying each protein's BCE term.

**Which path actually used it:**
- **End-to-end** `train_classifier` passes `sample_weight = batch["weight"]`
  (= `w_i`) into `classifier_loss` → confidence weighting **on**.
- **Cached probe** (the run at scale) calls `classifier_loss` **without**
  `sample_weight` → confidence weighting **off**; it relies on `pos_weight`
  alone for imbalance. (So in the models actually trained, high/medium tiers
  entered the *adapter* but not the cached-probe head.)

### B3. Other weightings (classifier)

- **Class-imbalance weight `pos_weight`** on the positive BCE term. End-to-end:
  optional `--pos-weight` (default None). Cached probe: set **per phenotype** to
  `n_neg / n_pos` on the train split (e.g. printed as `pos_weight` per pheno).
- **Focal option** (`focal_bce_loss`, `use_focal`): `(1 − p_t)^γ` down-weights
  easy examples, `γ`=2 default. Available, not used in the cached-probe run.
- **Pair-term weight `λ`** and **margin `δ`** on the matched-pair ranking loss
  (both **1.0** in the runs).
- **Negative subsampling `neg_per_pos`** (end-to-end only, default 3× positives)
  to bound wall-clock. Cached probe drops it and uses **all** negatives
  (an epoch is seconds).
- **Learning rates:** end-to-end `lr_head` 1e-3 / `lr_adapter` 1e-5 (two param
  groups); cached probe single `lr` 1e-3 (Adam).

### B4. Loss function (classifier)

**Plain language.** A pointwise term (weighted BCE — calibration on singletons)
plus a pairwise term (a margin ranking that forces each extremophile protein's
score above its taxonomy-matched mesophile ortholog's, so the head learns the
phenotype *delta* not the clade). Combined as BCE + λ·pair.

**Formula** (`losses.classifier_loss`, design §12 Loss 2):

    L_cls = L_BCE + λ · L_pair

Pointwise, weighted BCE-with-logits (`weighted_bce_loss`):

    L_BCE = − ( Σ_i w_i [ y_i·log σ(s_i) + (1−y_i)·log(1−σ(s_i)) ] ) / Σ_i w_i

with optional `pos_weight` on the positive term; or focal
`L = Σ_i w_i (1−p_t)^γ · BCE_i`.

Pairwise, margin ranking over matched (extremophile e, outgroup m) pairs
(`matched_pair_margin_loss`):

    L_pair = mean_{(e,m)} max(0, δ − (s_e − s_m))

Both members scored through the same backbone+head; the pair loader runs in
lockstep with the singleton loader (cycling, since pairs are fewer). `λ=0`
recovers pure pointwise BCE.

Model selection: **val AUPRC** (average precision), plus a taxonomy-controlled
`pair_acc` / `pair_auc` on held-out matched pairs (`evaluate_pair_metrics`) —
`pair_acc = 0.5` means the head is riding taxonomy, `>0.5` is genuine phenotype
signal.

---

## C. Parameter values used in the actual runs

| quantity | adapter (MLM) run | cached-probe classifier run |
|---|---|---|
| backbone | ESM-2 3B, LoRA r32/α64, full-attention targets | frozen 3B MLM-adapted, MLP head only |
| mask_rate | 0.15 (BERT 80/10/10) | — |
| coupling_mode | `both` (span-3 + contact) | — |
| contact_threshold / min_sep / top_k | 0.5 / 6 / 128 | — |
| gamma | 1.0 (inert; c=0, no MSA) | — |
| beta_kl | 0 (guard off) | — |
| confidence weights | high 1.0 / med 0.5 / none 1.0 / low 0.25 (**on**) | **off** (sample_weight not passed) |
| pos_weight | — | n_neg / n_pos per phenotype |
| λ / margin | — | 1.0 / 1.0 |
| neg_per_pos | (end-to-end 3×) | all negatives |
| lr | 1e-4, 5% warmup→linear, bf16 | 1e-3 Adam |
| epochs | 3 | 30 |
| selection metric | val pseudo-perplexity | val AUPRC (+ pair-AUC) |
