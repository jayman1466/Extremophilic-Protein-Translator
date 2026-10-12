# Ranker QC and recalibration plan

Status: diagnosis complete (2026-10-12), fixes staged. All numbers below come from
persisted run artefacts — `runs/mhk32/heads_meanpool/clf_*/metrics.json`,
`meanpool_production_all.json`, and the `round_stats` / `reject_breakdown` blocks
inside each `candidates_<pheno>.json`. Nothing here was re-trained or re-folded.

---

## 1. Contact-coupling term — fixed

**Bug.** Generation loads the MLM checkpoint through `PeftModel.from_pretrained`.
`PeftModel` does not expose `predict_contacts` (it lives on `EsmPreTrainedModel`,
past `PeftModel.base_model` → `LoraModel.model`), so `_predict_contacts` raised
`AttributeError`. Its inner handler only caught `TypeError`; the outer bare
`except Exception` swallowed it and returned `None`. `contact_pairs_from_map(None)`
then evaluated `np.asarray(None).shape[0]` → `IndexError: tuple index out of
range`, which the caller's own broad handler converted into the documented
"coupling term set to 1.0 for every design" fallback.

**Blast radius.** `p_coupling` was a constant 1.0 in *every* production run of both
pipelines. `score_product` was advertised as a three-term product and was in fact a
two-term product. The forensic marker is already in the output JSON:
`wt_contact_pairs = 0` on every run.

**Fix (3 files).**
- `src/eptrans/modeling/data.py::_predict_contacts` — walk down `base_model`/`module`
  until an object whose *type* defines `predict_contacts` is found; move tensors to
  the model's device; stop swallowing `AttributeError`; validate the map is square.
  A genuinely contact-incapable model still returns `None`, but a wiring bug now
  surfaces.
- `src/eptrans/modeling/masking.py::contact_pairs_from_map` — reject `None` and
  non-2-D input with an explanatory `ValueError` instead of coercing to a 0-d array.
- `scripts/11_generate_pipelineB.py` — if the WT contact map cannot be built, raise
  rather than silently degrade the objective. `GEN_B_REQUIRE_COUPLING=0` opts out and
  makes the two-term score an explicit choice.

**Next step.** Re-rank the existing designs offline. Every sub-score is persisted
per design, so this needs one forward pass of the contact head per design (no MPNN,
no ESMFold, no retraining) to fill in the missing `p_coupling` factor, after which
`score_product` can be recomputed and the rankings compared against the shipped ones.
Only *after* that comparison is weight tuning a meaningful question — right now we
have never observed the three-term objective at all.

---

## 2. Classifier heads — what went wrong and how to fix it

### 2.1 The observation

The same wild-type *E. coli* IS621 protein receives:

| head | `p_classifier` on WT IS621 | head base rate | pos_weight | train pairs |
|---|---|---|---|---|
| hyperthermophile | **1.0000** | 0.0014 | 975.4 | 235 |
| halophile | 0.5022 | 0.1182 | 12.9 | 59,426 |
| thermophile | 0.0001 | 0.0548 | 22.1 | 13,956 |
| alkaliphile | 0.0000 | 0.0105 | 155.7 | 6,377 |

Hyperthermophily is a strict subset of thermophily. A protein cannot be at the
ceiling of one and the floor of the other. The heads are not measuring the same
underlying axis, and at least one of them is not measuring thermostability at all.

### 2.2 Where the 1.0 comes from — three compounding causes

**(a) `pos_weight` is a decision-threshold shift, and the output is read as a
probability.** Each head is `BCEWithLogitsLoss(pos_weight=w)` on mean-pooled
embeddings, and generation consumes `sigmoid(logit)` directly. Training with
`pos_weight=w` moves the optimal logit by `+log w`. For hyperthermophile,
`log(975.4) = 6.88`, so a protein sitting *exactly at the base rate* — i.e. carrying
no evidence at all — receives `sigmoid(log(0.0014/0.9986) + 6.88) = 0.58`. The
operating point of the hyperthermophile head is "0.58 means nothing". Every head
has this offset; the size of the offset ranges from 2.6 (halophile) to 6.9
(hyperthermophile), which is exactly the ordering of how inflated their WT scores look.

**(b) The scores are saturated, so recalibration alone will not rescue them.**
`p = 1.0` at 4 decimal places means `logit ≥ 9.9`. Subtracting the 6.88 shift still
leaves a calibrated probability `≥ 0.95` — a 680× lift over the 0.0014 base rate.
Likewise `p = 0.0000` means `logit ≤ −9.9`. A 2-layer MLP trained 30 epochs on
8.3 M rows with no weight decay has pushed its logits far past the range where the
sigmoid carries ranking information; the head has become a hard classifier with a
~20-logit dead zone, which is why `p_classifier` stopped discriminating among designs.

**(c) The hyperthermophile head learned a taxon, not a phenotype.** 235 training
pairs drawn from a genome set that is ~49% Thermoproteota, against a base rate of
0.0014. Cluster-level leakage is *not* the problem — `stratified_group_split`
assigns whole mmseqs clusters (union-find over the 50% and 40% maps) to one split,
so homologues cannot straddle train/val. What is unguarded is **taxonomic**
leakage: the same handful of archaeal families appear on both sides of the split,
and mean-pooled ESM embeddings encode compositional/phylogenetic identity very
strongly. AUROC 0.9990 on a held-out set that shares phyla with training is
consistent with a near-perfect *phylum* detector and tells us almost nothing about
thermostability transfer to a bacterial protein. That is also why the number is
suspiciously higher than every other head despite having 60× less data —
a genuine phenotype signal gets *harder* with less data, a taxonomic shortcut does not.

### 2.3 Fix strategy, in order

1. **Stop consuming `sigmoid(logit)` as a probability.** Record `logit` and
   `logit − log(pos_weight)` alongside it, and have the generation oracle rank on the
   calibrated logit. This is a reporting change, not a retraining one, and it should
   land before anything else so that every downstream number is interpretable.
2. **Fit a proper calibration map on held-out data.** Platt scaling or isotonic
   regression on the val split, fit per head. Report calibrated probability *and*
   lift over base rate. A head whose calibrated curve is flat is not usable as an
   oracle regardless of its AUROC.
3. **Re-split hyperthermophile by taxonomy, not by cluster.** Hold out whole
   *families* (or whole phyla, given 49% Thermoproteota) and re-measure. The
   expected outcome is that AUROC falls substantially. That number — not 0.9990 —
   is the one to quote and the one that decides whether the head is deployable.
4. **Regularize the head.** Weight decay and early stopping on calibrated
   log-loss rather than AUROC; the current selection rule ("best-AUROC epoch") is
   indifferent to saturation and actively rewards it. Epoch 23 of 30 for
   hyperthermophile against epoch 9 for acidophile is a secondary symptom.
5. **Replace `pos_weight` with balanced sampling or focal loss** for the extreme
   heads (hyperthermophile 975, psychrophile 544, alkaliphile 156). `pos_weight` at
   these magnitudes is not a rebalancing device, it is a 7-logit bias term.
6. **Add a negative control to head QC.** Score a panel of well-characterised
   mesophilic proteins (IS621 itself is a fine first member) under every head. Any
   head that puts a mesophile above the 95th percentile of its own positive class
   fails QC. This single check would have caught the hyperthermophile head before
   it was used for generation.

### 2.4 QC of the other heads — same pathology, different severity

From `head_qc_meanpool.csv`:

| pheno | train pairs | base rate | pos_weight | val AUROC | val AUPRC | val pair-AUC |
|---|---|---|---|---|---|---|
| hyperthermophile | 235 | 0.0014 | 975.4 | 0.9990 | 0.9157 | 0.9590 |
| psychrophile | 3,793 | 0.0035 | 543.8 | 0.8956 | 0.0826 | 0.6071 |
| acidophile | 4,575 | 0.0227 | 64.5 | 0.9832 | 0.7888 | 0.8071 |
| alkaliphile | 6,377 | 0.0105 | 155.7 | 0.9769 | 0.6643 | 0.7518 |
| thermophile | 13,956 | 0.0548 | 22.1 | 0.9688 | 0.7899 | 0.9095 |
| halophile | 59,426 | 0.1182 | 12.9 | 0.9436 | 0.7554 | 0.7744 |

AUROC is *inversely* ordered against training-pair count across all six heads. That
monotone inversion is the signature of the shortcut: the rarer and more
taxonomically concentrated the positive class, the easier it is to identify by
lineage. Read AUPRC instead and the ordering becomes sane — psychrophile collapses
to 0.0826, which is the honest statement that the head has almost no usable signal,
and it is consistent with the 0.60 pair-AUC ceiling already recorded for that
phenotype.

**Severity triage:** hyperthermophile (unusable as shipped, must be re-split and
re-fit), psychrophile (AUPRC 0.083 — already known to be at a biological ceiling,
but the 0.90 AUROC should stop being quoted), alkaliphile (AUPRC 0.66 with
pos_weight 156 — recalibrate and re-check), acidophile / thermophile / halophile
(moderate `pos_weight`, AUPRC in a believable band — recalibration should be enough).

**λ is not implicated.** Across the λ sweep (0 → 4), AUROC moves by ≤0.010 for every
phenotype and hyperthermophile is flat at 0.9990 for all five values. The pathology
is in the loss weighting and the split, not in the pair-loss regularizer.

---

## 3. Calibrating the temperature and conservation caps

### 3.1 Temperature is not the binding constraint

Pooled over all four B-v2 runs, structural-gate acceptance by MPNN sampling
temperature:

| T | proposed | survivors | acceptance |
|---|---|---|---|
| 0.05 | 144 | 97 | 0.674 |
| 0.10 | 144 | 91 | 0.632 |
| 0.20 | 96 | 69 | 0.719 |

Acceptance is flat — and *non-monotone* — in T across the range actually used.
Lowering the temperature further will not buy survivors; it will only reduce
sequence diversity. The lever that does move acceptance is survivor conditioning:
in the three runs that worked, acceptance goes from 4–9/12 during the unbiased
warm-up to 11–12/12 once the bias switches on at round 2.

### 3.2 The RMSD caps are mis-placed, and that is the real knob

Every rejection records `(metric, observed, cap)`. Pooled across runs:

| metric | cap | n rejections | median observed | 10th pct | max |
|---|---|---|---|---|---|
| active_site | 1.0 Å | 125 | 1.163 | 1.09 | 1.478 |
| iface_A_D | 1.5 Å | 118 | 1.637 | 1.55 | 2.113 |
| iface_A_target_G | 1.5 Å | 39 | 1.671 | 1.54 | 2.003 |

18% / 41% / 28% of all rejections fall within 0.1 Å of their cap. The caps are
sitting in the dense part of the RMSD distribution rather than in its tail, so the
gate is behaving as a near-coin-flip on marginal designs, and the difference between
an accepted and a rejected design is frequently smaller than ESMFold's own
coordinate reproducibility. Round numbers (1.0, 1.5) were chosen by convention, not
measured.

**Calibration protocol.** Build the null distribution of the gate statistic under
the null hypothesis "this design is structurally identical to WT":

1. Refold WT *n* ≥ 30 times. For a deterministic folder, perturb the input instead —
   resample the recycling count, or fold a set of synonymous/strictly-conservative
   variants (surface Ala→Ser, Ile↔Leu at buried non-contact positions) that are
   expected to be structurally neutral.
2. Compute `active_site` and per-interface RMSD of each replicate against the WT
   reference, exactly as the gate does.
3. Set each cap at a chosen quantile of that null — the 95th percentile gives a
   gate with a 5% false-rejection rate on structurally-neutral designs, and the
   number is defensible because it is measured on this protein with this folder.
4. Report the measured null median alongside the cap in `candidates_*.json`, so the
   gate's own resolution is visible next to every rejection.

This replaces three hand-chosen constants with one decision (the false-rejection
rate you are willing to accept) and makes the active-site and interface caps
commensurable, which they currently are not.

### 3.3 Conservation freeze — your scepticism is well founded

`conservation_freeze = 0.9` freezes any position whose per-column MSA conservation
exceeds 0.9, treating positions as independent. That is the wrong model for exactly
the reason you give: thermostability adaptations are frequently *compensatory pairs*,
and a per-position filter is blind to the pair. It can also be actively harmful —
freezing one member of a coupled pair while leaving the other free is worse than
freezing both or neither, because it permits the half-mutation that breaks the
interaction.

Three options, in increasing order of effort:

- **Drop the per-position freeze entirely and let the structural gate do the work.**
  Defensible once the caps are calibrated (§3.2), because the gate measures the
  consequence (structural deviation) rather than a proxy for it. The risk is longer
  search: the gate rejects after folding, the freeze rejects before.
- **Replace the per-position threshold with a coupling-aware mask.** Freeze a
  position only if it is both highly conserved *and* a member of a high-confidence
  contact/coevolution pair whose partner is also frozen. The now-working contact map
  from §1 supplies the pairing directly. This keeps the cheap pre-filter while
  removing the half-mutation failure mode.
- **Calibrate the threshold against observed survival.** `round_stats` already
  records which positions were mutated in survivors versus rejects. Fitting
  rejection probability against the conservation of the mutated positions gives an
  empirical curve; place the freeze where that curve turns, rather than at 0.9.
  This is a post-hoc fit on 48 designs per phenotype and should be treated as
  indicative, not decisive.

Recommendation: the coupling-aware mask, because it is the option that §1's fix
newly makes available and it addresses the specific objection rather than working
around it.

---

## 4. Separate finding: the alkaliphile run deadlocked

Not asked about, but it falls out of the same `round_stats` data and it invalidates
the alkaliphile B-v2 result as a survivor-conditioned run.

| round | T | proposed | survivors | top-k used | biased positions |
|---|---|---|---|---|---|
| 0 | 0.05 | 12 | 4 | 0 | 0 |
| 1 | 0.10 | 12 | 6 | 0 | 0 |
| 2–7 | — | 12 each | **0** | 8 | 93 |

The survivor bias was fitted on the 10 warm-up survivors, applied from round 2, and
produced zero survivors in all six subsequent rounds. Because the bias only updates
from survivors, it then froze permanently — a positive-feedback deadlock. Every
shipped alkaliphile design therefore comes from the *unbiased* warm-up rounds, so
that phenotype is survivor-unconditioned in substance despite being labelled B-v2.

Two guards worth adding: require a minimum survivor count before the bias is allowed
to engage, and decay or reset the bias after *k* consecutive zero-survivor rounds.
