# IS621 / EPT — Final decisions (training + generative)

**Status:** consolidated 2026-10-11. Supersedes scattered entries in `labnotebook.md`.
Every number below is quoted from the primary record cited at the end of its section;
nothing here is recalled.

Production run namespace: **`mhk32`** — `/groups/cress/projects/jaymin/IS1111/eptrans/runs/mhk32/`
(`$PERSIST/runs/mhk32`). Full file map: `runs/mhk32/MANIFEST.md`.

---

## 1. Model-selection experiments — what was optimized and what won

Four decisions were screened, each on its own fixed clean validation set, seed 1466.

### 1a. Scope — secreted vs whole proteome
Screened on psychrophile (the hardest phenotype) as a 2×2 against tier, job **1175979**.

| Scope | Tier | AUROC | AUPRC |
|---|---|---|---|
| whole | H+M+L | 0.8038 | 0.0214 |
| whole | H+M | 0.8555 | 0.0306 |
| secreted | H+M+L | 0.7778 | 0.0369 |
| **secreted** | **H+M** | **0.8924** | **0.0872** |

**Won: secreted** (+0.0369 AUROC over whole at H+M). Note `secreted` here is the broad
signal-peptide definition (SP/LIPO/TAT/TATLIPO/PILIN) = 14.67M of 22M proteins, not the
narrower 1.98M soluble-secreted definition used in the earlier scoped_k32 run.

### 1b. Confidence tier — H+M vs H+M+L
Per-phenotype 1×2 at scope=secreted, job **1176274**.

| Phenotype | eval base | all AUROC | H+M AUROC | Δ(H+M − all) |
|---|---|---|---|---|
| thermophile | 0.0548 | 0.9620 | 0.9686 | +0.0066 |
| hyperthermophile | 0.0014 | 0.9889 | 0.9988 | +0.0099 |
| acidophile | 0.0227 | 0.9761 | 0.9830 | +0.0069 |
| alkaliphile | 0.0105 | 0.9651 | 0.9747 | +0.0096 |
| halophile | 0.1182 | 0.9423 | 0.9420 | −0.0003 |
| psychrophile | 0.0035 | 0.7778 | 0.8924 | +0.1146 |

**Won: H+M (low tier excluded) for all six.** Halophile is a within-noise tie (−0.0003)
and was set to H+M for a uniform policy; it also reverses in favour of H+M under pair-AUC
(0.7701 vs 0.7698). Psychrophile's large win is specific to its unusually noisy low tier
(75% of its positives) and does **not** generalize — which is why tier was decided
per-phenotype empirically rather than propagated from one phenotype.

### 1c. Matched-pair loss weight λ
Grid λ ∈ {0, 0.5, 1, 2, 4} × 6 phenotypes at locked scope+tier, job **1176405**
(`phase4_lam_sweep/lam_sweep_all.json`).

**Key finding — the two metrics disagree systematically.** AUROC is maximized (or tied
within 0.0003) at **λ=0 for all six** phenotypes; pair-AUC prefers **λ>0 for all six**.
The margin loss trades a little pointwise separation for better matched-pair ranking.

**Lock policy: pair-AUC**, because the deployment task *is* pair-ranking (score an
extremophile ortholog above its mesophile partner). The AUROC-optimal λ=0 head is a
worse ranker.

| phenotype | locked λ | pair-AUC at λ |
|---|---|---|
| psychrophile | 2.0 | 0.6071 |
| thermophile | 1.0 | 0.9095 |
| hyperthermophile | 4.0 | 0.9590 |
| acidophile | 1.0 | 0.8071 |
| alkaliphile | 2.0 | 0.7518 |
| halophile | 0.5 | 0.7744 |

### 1d. Pooling — K=32 gated attention vs mean
Six heads at their pair-AUC-locked λ (jobs 1176763, 1176816–1176820, sorted-gather patch
commit `87aa205`). Δ = attention − mean.

| Phenotype | λ | Attn AUROC | Attn pair-AUC | Mean pair-AUC | Δ pair-AUC | Δ AUROC |
|---|---|---|---|---|---|---|
| psychrophile | 2.0 | 0.8403 | 0.5970 | 0.6071 | −0.0101 | −0.0553 |
| thermophile | 1.0 | 0.9610 | 0.8880 | 0.9095 | −0.0215 | −0.0078 |
| hyperthermophile | 4.0 | 0.9981 | 0.8916 | 0.9590 | −0.0674 | −0.0009 |
| acidophile | 1.0 | 0.9769 | 0.7807 | 0.8071 | −0.0264 | −0.0063 |
| alkaliphile | 2.0 | 0.9684 | 0.7164 | 0.7518 | −0.0354 | −0.0085 |
| halophile | 0.5 | 0.9331 | 0.7524 | 0.7744 | −0.0220 | −0.0105 |

**Won: mean pooling, all six, no exceptions** — attention is worse on *both* metrics
everywhere. Interpretation: the adaptation signal ESM2-3B encodes for these phenotypes is
diffuse (bulk composition / charge / hydrophobicity spread across the sequence), not
localized to a few salient residues, so K=32 attention slots have nothing to concentrate
on and only add parameters and a harder optimization than a parameter-free mean.

### Selection-metric convention (canonical for this project)
- scope / tier / signal decisions → **AUROC**
- λ lock for deployment heads → **pair-AUC**
- deployment lift reporting → **AUPRC as lift over base rate**
- headline cross-phenotype ranking → **pair-AUC**

---

## 2. Final locked training configuration

### Corpus (frozen, read-only)
- `assemble/labeled_dataset.parquet` — 22,007,249 rows (incl. 330 `CU_` deep-sea MAGs)
- `assemble/labeled_dataset_protein_pairs.tsv`
- Splits leakage-clean: 0 of 11,435,706 clusters straddle a split boundary.
  Split distribution train 17,597,678 / val 2,199,399 / test 2,210,172.

### Stage 08 — MLM LoRA adapter (job **1174115**, gpu_h200)
| Parameter | Value |
|---|---|
| backbone | `facebook/esm2_t36_3B_UR50D` (ESM2-3B) |
| LoRA rank / alpha / dropout | 32 / 64 / 0.05 |
| epochs / lr / mask rate | 3 / 1e-4 / 0.15 |
| masking | coupling-aware (contact mode), 396,750 precomputed contact pairs |
| tier restriction | **medium + high only** (low excluded — with low the adapter is barely distinguishable from vanilla ESM) |
| subsample | 400,000 train / 20,000 val (medium 336,656 + high 83,344; asserted zero low/none/mesophile) |
| result | val_ppl 6.1659 → 6.1657 → **6.1548**, monotonic, best = epoch 2 |

Subsample funnel: 22,007,249 → drop mesophiles → 10,449,465 extremophile → keep M+H →
4,288,083 candidate cluster reps → 400k/20k sampled.
Final val-PPL is **lower** than the superseded scoped baseline (6.1548 < 6.2168),
validating the M+H restriction.

### Stage 09 — embedding cache
`runs/mhk32/embeddings/secretome_mhk32`: 16/16 shards, **17,700,477 rows**,
emb float16 (…, 2560), pos int32 (…, 32), **~2.8 TB**. Adapter remap asserted at build:
288 LoRA tensors, 0 unmatched. Low tail subsampled at build.

### Stage 19 — production classifier heads (job **1178100**, gpu_h200, 3h22m)
| Parameter | Value |
|---|---|
| scope | secreted |
| tier | H+M |
| pooling | mean |
| λ | per-phenotype (table in §1c) |
| neg_per_pos | 3.0 |
| confidence weights | high 1.0 / medium 0.5 / low 0.15 / none 1.0 |
| architecture | Linear(2560, 512) → GELU → Dropout(0.1) → Linear(512, 1) |
| seed | 1466 |

> The `low 0.15` weight is inert in production: tier = H+M excludes low-tier genomes from
> the training set entirely. It differs from `config/config.yaml` (`low: 0.25`), which is
> the pre-lock default — the run value is authoritative.

**Production head metrics:**

| phenotype | λ | AUROC | AUPRC | pair-AUC | best ep | train_n | pos_weight | train_pairs | base_rate |
|---|---|---|---|---|---|---|---|---|---|
| psychrophile | 2.0 | 0.8956 | 0.0826 | 0.6071 | 17 | 8,337,303 | 543.75 | 3,793 | 0.0035 |
| thermophile | 1.0 | 0.9688 | 0.7899 | 0.9095 | 16 | 8,790,675 | 22.09 | 13,956 | 0.0548 |
| hyperthermophile | 4.0 | 0.9990 | 0.9157 | 0.9590 | 23 | 8,320,850 | 975.44 | 235 | 0.0014 |
| acidophile | 1.0 | 0.9832 | 0.7888 | 0.8071 | 9 | 8,506,619 | 64.54 | 4,575 | 0.0227 |
| alkaliphile | 2.0 | 0.9769 | 0.6643 | 0.7518 | 9 | 8,401,465 | 155.71 | 6,377 | 0.0105 |
| halophile | 0.5 | 0.9436 | 0.7554 | 0.7744 | 22 | 9,432,092 | 12.89 | 59,426 | 0.1182 |

**psychrophile is retained but CLOSED** at ~0.61 pair-AUC (AUPRC 0.083 at base rate
0.0035, only 3,793 pairs) — a data/biology limit, not a tuning gap. Two follow-up probes
confirmed this independently: an ANI/T-opt augmentation analysis (do-not-augment) and a
phylogeny-controlled directional-selection probe (no site-specific cold signal).
Deployment leads with **halophile + hyperthermophile**.

### Stage 20 — regrouped calibrated heads (2026-10-26) — USE THESE FOR DESIGN

`scripts/20_train_calibrated_heads.py`, output root
`$PERSIST/runs/mhk32/heads_production/<arm>/clf_<pheno>/head_best.pt`. Same cache, same
architecture, same seed; the change is **taxonomy-grouped train/val splitting** at an
explicit `--group-rank`, plus Platt calibration and a negative-control panel.

| phenotype | rank | split | val_pairs | AUROC | AUPRC | pair-AUC | ECE | cluster overlap | arm |
|---|---|---|---|---|---|---|---|---|---|
| thermophile | phylum | single | 709 | 0.9201 | 0.6482 | **0.8786** | 0.0411 | 0.0308 | `rank_phylum` |
| halophile | phylum | single | 9,913 | 0.8947 | 0.7825 | 0.7243 | 0.0143 | 0.0271 | `rank_phylum` |
| alkaliphile | phylum | single | 656 | 0.9220 | 0.3761 | 0.7187 | 0.0426 | 0.0290 | `rank_phylum_acido_alkali` |
| acidophile | **order** | single | 610 | 0.9616 | 0.6407 | 0.7606 | 0.0124 | 0.1114 | `rank_order` |
| hyperthermophile | family | **20× CV** | 774 pooled | 0.9973 | 0.7975 | **0.8695** [0.854, 0.883] | 0.0008 | 0.1370 | `cv_hyperthermophile_secreted` |
| psychrophile | phylum | single | 45 | 0.6757 | 0.0063 | 0.8336 | 0.0108 | 0.0288 | `rank_phylum_psychrophile_secreted` |

**These are lower than the 1178100 table above and that is the point.** The split basis of
the 1178100 run is not recorded in this document, so the two tables are not a controlled
comparison — but the Stage-20 heads are grouped by taxonomy at a stated rank with the
resulting train/val cluster overlap measured and reported, and they are the heads to use
when scoring designs. A congener-matched pair-AUC near 0.88 on a phylum-grouped split is a
stronger claim than 0.91 on an unrecorded split.

**psychrophile is NOT production.** AUROC 0.6757 is near chance and AUPRC 0.0063 against a
0.00348 base rate is ~1.8× lift. Its pair-AUC 0.8336 rests on 45 pairs and contradicts its
own AUROC. This corroborates the CLOSED call above rather than reopening it.

**Grouping-rank criterion (amended 2026-10-26).** The code gates on group count only:
`coarsest rank with n_groups >= min_groups`. The amended rule adds a pair floor:

```
group_rank = coarsest rank satisfying
    n_groups >= min_groups    AND    val_pairs >= pair_floor   (~200-300)
where val_pairs may be POOLED across CV replicates
```

Evidence: every phenotype clears `--min-groups 20` at phylum, so the finer ranks were never
forced by feasibility. Alkaliphile moved order→phylum and pair-AUC rose 0.658→0.719 on 656
pairs while AUROC/AUPRC fell — the leakage signature. Acidophile at phylum lost 92% of its
pairs (610→49, base rate halved 0.0276→0.0118), so it stays at order. Hyperthermophile
cannot clear the floor on any single split (median 20 pairs/replicate, 6 of 20 replicates
yield **zero**) and clears it only by pooling 20 grouped CV replicates. The second clause
is **not in the code yet**; adding it would have blocked the acidophile phylum arm.

> **Harvest note.** All Stage-20 arms exit 0 with "harvested 0 files" — the trainer writes
> to the persist tree, not the job workdir. Read metrics from
> `heads_production/<arm>/clf_<pheno>/metrics.json`, not from harvested files.

---

## 3. Where the final models and training data live

All paths relative to `$PERSIST = /groups/cress/projects/jaymin/IS1111/eptrans`.

### Models (verified on disk 2026-10-11)
| Artifact | Path | Size |
|---|---|---|
| LoRA adapter (production) | `runs/mhk32/mlm_adapt/mlm_adapter_best/adapter_model.safetensors` | 94,413,200 B |
| adapter config | `runs/mhk32/mlm_adapt/mlm_adapter_best/adapter_config.json` | 1,076 B |
| adapter training history | `runs/mhk32/mlm_adapt/mlm_history.json` | 79,844 B |
| classifier heads ×6 | `runs/mhk32/heads_meanpool/clf_<pheno>/head_best.pt` | 5,249,184 B each |
| per-head metrics ×6 | `runs/mhk32/heads_meanpool/clf_<pheno>/metrics.json` | ~590 B each |
| full λ-grid roll-up | `runs/mhk32/heads_meanpool/meanpool_production_all.json` | — |

`<pheno>` ∈ {acidophile, alkaliphile, halophile, hyperthermophile, psychrophile, thermophile}.
All six heads verified present and byte-identical in size.

### Training data (frozen inputs — symlinked into `runs/mhk32/inputs/`, not duplicated)
| File | Path |
|---|---|
| labelled dataset | `$PERSIST/labeled_dataset_r232_clustered.parquet` |
| ortholog protein pairs | `$PERSIST/labeled_dataset_r232_clustered_protein_pairs.tsv` |
| MLM subsample | `$PERSIST/labeled_mlm_subsample.parquet` |
| mature-seq FASTA | `$PERSIST/mlm_subsample_mature.faa.gz` |
| secreted proteins | `$PERSIST/secreted_proteins_r232.{tsv,faa}` |
| cluster map | `$PERSIST/secreted_clusters_r232_cluster.tsv` |
| cluster reps | `$PERSIST/secreted_clusters_r232_rep_seq.fasta` |
| GenomeSPOT predictions | `$PERSIST/genomespot_predictions_r232.tsv` |
| contact pairs | `runs/mhk32/contact_pairs.parquet` |
| embedding cache | `runs/mhk32/embeddings/secretome_mhk32/` (~2.8 TB, 16 shards) |

### Model-selection provenance (JSON)
| Decision | File |
|---|---|
| scope | `runs/mhk32/phase3_psy_2x2/psy_scope_tier_2x2.json` |
| tier | `runs/mhk32/phase3_pheno_1x2/phenotype_tier_1x2.json` |
| λ | `runs/mhk32/phase4_lam_sweep/lam_sweep_all.json` |
| pooling | `runs/mhk32/phase5_attn/attn_heads_summary.json` |

### Superseded — archived 2026-10-12

All superseded material was moved (same-filesystem rename, reversible — nothing
deleted) after confirming all six production heads and the production adapter
are present on disk.

| Was | Now | Size | Notes |
|---|---|---|---|
| `$PERSIST/embeddings_scoped_k32/secretome_scoped/` | `$PERSIST/old/embeddings_scoped_k32/` | ~2.8 TB | superseded by `runs/mhk32/embeddings/secretome_mhk32/`; regenerable from the adapter via stage 09 (GPU array, ~24 h) |
| `$PERSIST/models_scoped_k32/` | `$PERSIST/old/models_scoped_k32/` | ~399 MB | pre-production heads, old `mlm_adapt`, exploratory eval JSONs, and the only copy of `contact_pairs.parquet` |
| 16 superseded generative runs under `$SCRATCH/gen/` | `$SCRATCH/gen/old/` | small | the three pre-final A batches and the two pre-final B-v1 batches |

Verified after the move: all 8 symlinks in `runs/mhk32/inputs/` still resolve;
`heads_meanpool/` and `mlm_adapt/` untouched.

**Not yet reclaimed.** `old/embeddings_scoped_k32/` is the only item large enough
to matter (`/groups` sits at 94 %). Moving it freed nothing — deleting it would
free ~2.8 TB but costs a ~24 h GPU array to regenerate. Decide explicitly before
deleting. `contact_pairs.parquet` lives only inside `old/models_scoped_k32/`; it
is needed for contact-aware MLM masking if the adapter is ever retrained, so
pull it back out before reclaiming that directory.

Kept generative runs (the benchmark set): `is621_20260818_231612_{alkaliphile,halophile}`,
`is621_20260818_231613_{hyperthermophile,thermophile}` (A);
`is621B_20260819_000816_*` (B-v1); `is621Bv2_20260819_002005_{halophile,hyperthermophile,thermophile}`
+ `is621Bv2_20260819_003434_alkaliphile` (B-v2); plus `tracks_B_v2/`.

---

## 4. Generative pipeline

Pipeline selection was benchmarked on **IS621 (8WT6 chain A, 306 aa)** — §4a–4d below.
The **current design target is laccase (1GYC)** — §4e. The pipeline itself is unchanged;
only the scaffold, the phenotype set and the constraint set differ.

### 4a. IS621 benchmark (completed 2026-08-19)

Three variants were benchmarked on four phenotypes. All share Stages 0–5 (MSA
conservation → active-site transfer → interface detection → frozen-position set →
ESM2-3B + mhk32 adapter + `head_best.pt`) and diverge at Stage 6.

| Pipeline | Script | Stage-6 proposer |
|---|---|---|
| **A** | `scripts/11_generate.py` | MLM-in-the-loop Gibbs + Metropolis-Hastings, contact/span-coupled masking, classifier-score acceptance, periodic refold + RMSD rollback |
| **B-v1** | `scripts/11_generate_pipelineB.py` | LigandMPNN proposer, **iid** 4 rounds × 6 candidates = 24 proposals |
| **B-v2** | same script, survivor flags on | LigandMPNN proposer, **survivor-conditioned** 8 rounds × 12 candidates = 96 proposals, per-residue bias from the top-8 survivors |

### Benchmark (top-1 per phenotype; `product` = reported `score_product`)

| phenotype | pipeline | p_clf | bio | product | n_mut | survivors |
|---|---|---|---|---|---|---|
| thermophile | A | 0.0032 | 0.5376 | 0.0017 | 25 | — |
| | B-v1 | 0.7284 | 0.6160 | 0.5164 | 87 | 15 |
| | **B-v2** | 0.8097 | 0.6275 | **0.5747** | 96 | 82 |
| halophile | A | 0.7723 | 0.1242 | 0.0959 | 9 | — |
| | B-v1 | 0.7234 | 0.1699 | 0.5256 | 86 | 13 |
| | **B-v2** | 0.9050 | 0.1797 | **0.6391** | 93 | 79 |
| hyperthermophile | **A** | 1.0000 | 0.5408 | **0.5408** | 8 | — |
| | B-v1 | 0.0034 | 0.5997 | 0.0024 | 90 | 14 |
| | B-v2 | 0.0553 | 0.6291 | 0.0376 | 100 | 86 |
| alkaliphile | A | 0.0000 | 0.0490 | 0.0000 | 9 | — |
| | B-v1 | 0.6441 | −0.0392 | 0.4486 | 93 | 11 |
| | **B-v2** | 0.8676 | −0.0294 | **0.5939** | 86 | 10 |

B rows' `score_product` folds in seqLL ratio and coupling subscores; A rows use
`p_clf × bio` since A candidates don't emit `score_product`.

> **Caveat added 2026-11-16.** `p_coupling` was identically 1.0 in every run in
> this table — the contact head raised under SDPA attention and the design-side
> handler returned a neutral value. `score_product` here is therefore
> `p_classifier × seqLL_ratio` in effect. The ranking *within* this table is
> unaffected (a constant factor cannot reorder it), but it is not the
> three-factor score the column name implies. Fixed in `49837ac`; a re-run under
> a live coupling term has not happened.

### Verdict
**B-v2 (survivor-conditioned LigandMPNN-in-the-loop) is the default pipeline.**
It beats B-v1 on classifier score in every phenotype (thermo 0.73→0.81, halo 0.72→0.90,
hyper 0.003→0.055, alkali 0.64→0.87) and beats A on product score in 3 of 4.

**Documented exception — hyperthermophile: use Pipeline A.** A reaches p_clf = 1.0 at
only 8 mutations; both B variants collapse the classifier signal at ~85–100 mutations.
This is a recurring signature: highly-specific classifiers reward near-WT trajectories and
MPNN-in-the-loop overshoots the mutation budget.

**Alkaliphile caveat.** The alkaliphile classifier scores WT at exactly 0.0, so Pipeline A
has no gradient to climb (every A design scored 0.0). B-v1/v2 reach 0.64/0.87 only by
introducing 85–95 mutations. Treat alkaliphile designs as exploratory.

### Running B-v2 — the defaults are v1
`scripts/slurm/11_generate_pipelineB_pipeline.sbatch` defaults to **v1** behaviour
(`ROUNDS=4`, `BATCH=6`, `SURVIVOR_TOPK=0`). v2 requires the environment overrides:

```bash
GEN_B_ROUNDS=8 GEN_B_BATCH=12 GEN_B_SURVIVOR_TOPK=8 \
GEN_B_SURVIVOR_MIN_AGREE=2 GEN_B_SURVIVOR_BIAS_STRENGTH=5.0 GEN_B_SURVIVOR_WARMUP=2 \
sbatch scripts/slurm/11_generate_is621_pipelineB.sbatch
```

Shared gates in both pipelines: `--conservation-freeze 0.90`, `--core-rmsd-cap 1.0` Å
(active site), `--interface-rmsd-cap 1.5` Å, `--interface-contact-cutoff 4.5` Å, seed 1466.
All top-1 designs passed the hard gates (active-site RMSD 0.58–0.87 Å; interface
0.19–1.34 Å).

Results: `/groups/cress/projects/jaymin/eptrans_scratch/gen/is621Bv2_*/results.json` with
matching PDBs under `structures/<pheno[:4]>_B{1..3}.pdb`.

### 4e. Current target — laccase, PDB 1GYC (started 2026-10-26)

**Scaffold.** *Trametes versicolor* laccase, 1GYC chain A, 1.9 Å, 499 aa contiguous,
mature protein. UniProt **Q12718** (LAC2_TRAVE); PDB numbering = UNP − 20, so PDB 1–499 =
UNP 21–519. The native 20-aa secretion signal (UNP 1–20) is **not** in the construct.

**Phenotypes: thermophile, alkaliphile, acidophile.** Chosen because all three now have
locked production heads (§2), and because laccase has industrial relevance at each of
those conditions. Hyperthermophile and halophile are not targeted here.

**Motivation for switching off IS621.** Fungal laccase folds with a high-mannose glycan
shell. Running it through LigandMPNN forces the inverse-folding model to stabilise the
fold from amino acids alone, so the designs should be *less* glycosylation-dependent
than WT — i.e. the aglycosylation liability is part of what the redesign is meant to fix,
not a confound.

**Expression host: bacterial, aglycosylated.** Decided 2026-10-26. N-glycosylation
sequons are therefore **not** forbidden in design — no glycosylation occurs in the host,
so a retained sequon is inert. (LigandMPNN has no sequon constraint anyway; `--omit_AA`
is global, not motif-aware, and a post-hoc N→Q/S→A repair would have to be gated on not
touching the freeze set. Both avoided.)

**Freeze set: 16 residues** — `laccase_1gyc_constraints.json`, 483 of 499 designable.

| group | residues | evidence |
|---|---|---|
| copper ligands (11) | 64, 66, 109, 111, 395, 398, 400, 452, 453, 454, 458 | geometric and UniProt agree **exactly** (11 vs 11, zero either-only), and the T1/T2/T3 partition matches |
| disulfide Cys (4) | 85, 117, 205, 488 | 2 bonds, SG–SG 1.98 / 2.04 Å; both annotated |
| redox axial (1) | **463 (Phe)** | 3.65 Å from T1 Cu, **non-coordinating** |

Copper site assignment: **T1** Cu1503 = His395 + Cys453 + His458 (2His+1Cys);
**T2** Cu1502 = His64 + His398 + solvent; **T3α** Cu1500 = His66/His109/His454;
**T3β** Cu1501 = His111/His400/His452. Trinuclear cluster Cu–Cu 3.81–3.91 Å;
T1→cluster 12.3–14.7 Å. **His452-Cys453-His454** is the bridge from T1 to the T3 pair —
do not mutate *or reorder*. Cys453 is the one free cysteine (it is a Cu ligand, not a
disulfide partner).

Phe463 is the only freeze-set member that ligand-distance detection would miss, since it
coordinates nothing. Phe/Leu at the T1 axial position is the high-potential signature of
fungal laccases; Met there (bacterial/plant multicopper oxidases) costs hundreds of mV of
E°. Frozen for that reason, not for geometry.

**Aglycosylation liability (what the design must compensate).** Removing the ordered
proximal NAGs exposes **796.8 Å²**, 4.43% of total surface, of which **27.3%** is
nonpolar. That resolves into three spatial hydrophobic patches plus one isolated residue:
`[232, 297, 298]`, `[57, 155, 158]`, `[329, 338, 386]`, `[325]`. Each patch sits under a
different glycan. **This 797 Å² is a floor on two counts**: only the ordered proximal NAG
is modelled (the native glycan is high-mannose), and two annotated sequons carry no
modelled sugar at all.

Sequons: 8 annotated by UniProt (54, 141, 208, 217, 251, 333, 341, 436); 6 occupied in
the crystal; 141 and 208 annotated with no sugar modelled. Of those two, 141 is
solvent-exposed (SASA 62.9 Å², plausible real site) and 208 is buried at 8.4 Å² and
unlikely to be glycosylated in practice despite the annotation. All designable.

**Status.** Structure prep complete; MSA build running (job `6a52a244`, single-split
UniRef30, 674 GB resident). Design not yet started.

### Operational rule learned
**Never edit an on-disk sbatch/script while jobs that read it are still running.** All
four B-v1 jobs (1178497–1178500) crashed with a stale file handle at the `11d_assemble`
stage when the shared inner sbatch was edited mid-run. Generation had already completed, so
recovery only required re-running `11d_assemble.py` (with a stubbed `mpnn.json`, which
argparse requires but the B path doesn't use).

---

## 5. Open items

1. ~~**Biophysical composite re-weighting.**~~ **WITHDRAWN 2026-11-16 — the premise was
   false.** This item said `p_coupling` dominates and penalises non-WT contact patterns,
   pushing a ~90-mutation redesign toward zero or negative. `p_coupling` was a constant
   1.0 in every run cited (two stacked faults: `PeftModel` not exposing
   `predict_contacts`, and SDPA attention returning no attention weights, so the contact
   head raised and the design-side handler returned neutral). A constant cannot dominate,
   and re-weighting against one is meaningless. The negative alkaliphile bio score
   (−0.0294) has some other cause, not yet identified.
   **The underlying question is untested, not answered:** whether a 90-mut redesign
   *should* pay a contact-map penalty was never measured, because the term that would
   measure it never ran. Re-opened as item 6.
   Superseding fix: `49837ac`. Evidence: jobs `ce8c83fa`, `4bb43cca`; notebook
   2026-11-16.
2. **Hyperthermophile B variant.** Either cap the mutation budget or shorten to ≤2 rounds
   so B stops overshooting the near-WT optimum A finds.
3. **`MANIFEST.md` §3 val-PPL is stale** — it carries the superseded scoped baseline's
   trajectory (6.237 → 6.217 → 6.2168) rather than the mhk32 adapter's
   (6.1659 → 6.1657 → 6.1548). Corrected here; fix in the manifest.
4. ~~**Corpus table all-split totals.**~~ **RESOLVED 2026-10-12.** The `corpus_tab.py`
   aggregation had in fact completed on the h200 node (output written 2026-08-19 09:31);
   only the file transfer back failed, which is why it looked like the job never ran.
   `assemble/corpus_tables_export.json` was read directly and Table 1 of
   `results/mhk32_training_data_tables.md` is now complete. The prefixed-vs-bare accession
   join is confirmed clean (`ext_unmapped = 0` for all six phenotypes). Note that `all` and
   `H+M` are identical in every pair field — the pair table is built after the tier filter,
   so low-tier genomes never enter a pair. Psychrophile is present at `whole_proteome`
   scope only in this corpus. **Provenance error also corrected:** the genome-tier counts
   previously in that table (thermophile 12,219 etc.) were labelled as coming from
   `labeled_dataset.parquet` but were in fact the pre-selection label sweep over all
   199,923 r232 representatives — roughly 2× the post-selection corpus (thermophile 5,425).
   The table now carries the measured post-selection counts and names both populations.
5. **Pair floor not implemented.** `20_train_calibrated_heads.py` gates `--group-rank` on
   `--min-groups` only. Add a `--min-val-pairs` clause (§2) so an arm that would validate on
   <200 pairs is rejected before it consumes a node, with pooling allowed for CV arms.
6. **`docs/pipeline_architecture.md` predates B-v2** (doc commit `b033fff` 2026-08-18;
   v2 feature commit `8cf4f97` 2026-08-19). Its "Pipeline B" diagram is the v1 4×6 iid loop.
   The B-v2 loop is documented in the `eptrans-generative-design` skill instead; fold it
   back into this doc when convenient.
7. **Re-run the generative comparison under a live coupling term.** Opened 2026-11-16
   when item 1 was withdrawn. Every B result in §4 was scored with `p_coupling`
   constant at 1.0, so the three-factor composite has never actually been exercised, and
   the question item 1 thought it had answered — does a ~90-mutation redesign pay a
   contact-map penalty, and should it — is still open. Two sub-questions: whether the
   A-vs-B-v2 verdict survives a live third factor, and whether the hyperthermophile
   exception (item 2) is partly an artifact of scoring without it.
   **Blocked on hardware, not code.** `gpu_h200` drained by root since 2026-11-13 for a
   faulty GPU; the `gpu` partition QOS caps a job at 12 G against an ~11 GB backbone plus
   activations; the `igi` association grants no alternate QOS. CPU-only verification is
   available meanwhile via `scripts/diagnostics/` (~20 min/run on `standard`).

### Companion skill

`eptrans-generative-design` (created 2026-10-12) carries the operational form of
§4: pipeline-selection rule, the v2 env overrides, structural gates, frozen scorer
paths, output layout, and the stale-file-handle / `11d_assemble` recovery rules.
Update it alongside this section whenever the generative pipeline changes.

## 6. Infrastructure note

> **Update 2026-10-26: the Slurm outage below is RESOLVED.** `sbatch`, `squeue`, `sinfo`
> and `sacct` all work from the login node. Two issues seen since: (a) the synchronous
> command channel intermittently returns exit 0 with empty stdout and stops executing —
> recreate the handle, or fall back to batch submission, which stayed reliable throughout;
> (b) filesystem-wide `find` probes wedge the channel, so scope them to a directory.
>
> **Standing partition rule (user, 2026-10-12):** never submit to `standard` when it is
> full — check `sinfo` idle counts first and prefer `memory` when it makes more sense. The
> `gpu_h200` node (`node-224-2t-8gpu-1`, 2 TB) accepts CPU-only jobs when both are at zero
> idle. Never set a small wall-time cap on biotite; deploy before running; do not
> repeatedly cancel jobs under 25 min.

### Original outage note (2026-10-11, historical)

The Slurm client stack on the biotite login node was **down**: `sbatch`, `squeue` and
`sinfo` all fail with `resolve_ctls_from_dns_srv: res_nsearch error: Unknown host` →
`Could not establish a configuration source`. This is a cluster-side DNS/SRV problem, not
a credential or path issue. Batch submission is unavailable until it's fixed; login-node
foreground and background shell work normally.

Separately, if SSH to biotite fails with a control-socket bind error, the multiplexer
directory under `/tmp` has been wiped — recreate it (`mkdir -p /tmp/op-mux-*`) and the
connection recovers immediately.
