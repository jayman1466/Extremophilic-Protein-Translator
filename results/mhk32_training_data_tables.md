# IS621 Input Training-Data Tables (mhk32 production run)

_Tables 2–4 are from the mhk32 consolidated production summary (labnotebook
§3–§8). Table 1 is measured directly from the current post-MAG corpus by the
`corpus_tab.py` aggregation (biotite job, 2026-08-19), with the mhk32 per-head
counts reported separately alongside for comparison._

---

## Table 1 — Corpus of training data

**Source:** `assemble/corpus_tables_export.json` (written 2026-08-19 09:31 on biotite),
computed directly from the 22,007,249-row `labeled_dataset.parquet` and
`labeled_dataset_protein_pairs.tsv`. The prefixed-vs-bare accession join
(`GB_`/`RS_`/`CU_` stripped before matching `ext_acc`) is clean: `ext_unmapped = 0`
for all six phenotypes.

### 1a. Extremophile genomes in the assembled corpus, by confidence tier

| Phenotype | Total | High | Medium | Low | H+M |
|---|---|---|---|---|---|
| thermophile | 5,425 | 1,970 | 2,018 | 1,437 | 3,988 |
| hyperthermophile | 881 | 100 | 385 | 396 | 485 |
| psychrophile | 1,597 | 10 | 365 | 1,222 | 375 |
| acidophile | 2,887 | 730 | 1,115 | 1,042 | 1,845 |
| alkaliphile | 1,245 | 130 | 614 | 501 | 744 |
| halophile | 4,793 | 686 | 4,003 | 104 | 4,689 |
| **all (non-unique)** | **16,828** | **3,626** | **8,500** | **4,702** | **12,126** |

A genome carrying two labels (e.g. `hyperthermophile;thermophile`) is counted once
per label, so the total row is not a genome count.

These are **post-selection** counts — genomes that survived the lineage caps and
entered the assembled dataset. The pre-selection label sweep over all 199,923 r232
representatives is a different and roughly 2× larger population (e.g. thermophile
12,219 there vs 5,425 here); an earlier version of this table reported those numbers
under a `labeled_dataset.parquet` provenance label, which was wrong.

**Mesophilic outgroup pool: 25,452** genomes present in the
corpus as outgroups. Distinct from the **104,486** confident-mesophile genomes among
the 199,923 GTDB r232 representatives — that is the pool *selection drew from*;
25,452 is what survived lineage caps and matching into the dataset. The outgroup pool
carries no extremophile tier, so it has no tier breakdown.

### 1b. Genome pairs and protein pairs (H+M), at each phenotype's corpus scope

| Phenotype | Scope | Ext genomes paired | Outgroup genomes | Genome pairs | Protein pairs |
|---|---|---|---|---|---|
| thermophile | secreted | 695 | 695 | 695 | 17,596 |
| hyperthermophile | whole_proteome | 232 | 232 | 232 | 37,355 |
| psychrophile | whole_proteome | 329 | 329 | 329 | 309,989 |
| acidophile | secreted | 416 | 416 | 416 | 5,682 |
| alkaliphile | secreted | 359 | 359 | 359 | 7,819 |
| halophile | secreted | 1,881 | 1,881 | 1,881 | 74,046 |

**`all` and `H+M` are identical in every pair field.** Pairing runs *after* the tier
filter, so a low-tier genome never enters a pair; the two columns the request asked to
separate cannot differ by construction. Verified field-by-field in the export.

### 1c. Protein pairs by split

| Phenotype | Train | Val | Test | Total |
|---|---|---|---|---|
| thermophile | 13,956 | 1,760 | 1,880 | 17,596 |
| hyperthermophile | 30,387 | 3,555 | 3,413 | 37,355 |
| psychrophile | 248,647 | 31,498 | 29,844 | 309,989 |
| acidophile | 4,575 | 580 | 527 | 5,682 |
| alkaliphile | 6,377 | 639 | 803 | 7,819 |
| halophile | 59,426 | 7,065 | 7,555 | 74,046 |

### 1d. H+M protein-pairs into each mhk32 production head (train split)

The counts that actually fed each trained head, from mhk32 §3:

| Phenotype | Scope | Train H+M protein-pairs |
|---|---|---|
| thermophile | secreted | 13,956 |
| hyperthermophile | secreted | 235 |
| acidophile | secreted | 4,575 |
| alkaliphile | secreted | 6,377 |
| halophile | secreted | 59,426 |
| psychrophile | secreted | 3,793 |
| psychrophile | whole | 248,647 |

**Reconciliation with 1c.** Five of these seven rows are identical to the corpus
train counts in 1c (thermophile, acidophile, alkaliphile, halophile, psychrophile
whole). The two that differ are the two whose scope differs from the corpus scope:
hyperthermophile ran its head on **secreted** (235) while the corpus carries it at
**whole_proteome** (30,387), and psychrophile-secreted (3,793) has no counterpart
because the corpus holds psychrophile only at whole_proteome. The psychrophile
"secreted AND whole" comparison the request asked for is therefore available at
head level (1d) but not at corpus level (1c).

---

## Table 2 — Tier sweep (H+M vs H+M+L): AUROC and pair-AUC

**Psychrophile 2×2 (scope × tier)**, job 1175979. All four heads scored on ONE fixed
clean secreted H+M val set (pos=3,640, neg=1,040,184, base=0.0035); λ=1.0:

| Scope | Tier | AUROC | AUPRC |
|---|---|---|---|
| whole | H+M+L | 0.8038 | 0.0214 |
| whole | H+M | 0.8555 | 0.0306 |
| secreted | H+M+L | 0.7778 | 0.0369 |
| **secreted** | **H+M** | **0.8924** | **0.0872** |

**Per-phenotype tier 1×2 (scope=secreted, H+M+L vs H+M)**, job 1176274; each scored on
its own clean eval set; λ=1.0:

| Phenotype | eval base | all(H+M+L) AUROC | H+M AUROC | Δ(H+M−all) | all AUPRC | H+M AUPRC |
|---|---|---|---|---|---|---|
| thermophile | 0.0548 | 0.9620 | 0.9686 | +0.0066 | 0.7870 | 0.7917 |
| hyperthermophile | 0.0014 | 0.9889 | 0.9988 | +0.0099 | 0.8736 | 0.9152 |
| acidophile | 0.0227 | 0.9761 | 0.9830 | +0.0069 | 0.7810 | 0.7881 |
| alkaliphile | 0.0105 | 0.9651 | 0.9747 | +0.0096 | 0.6440 | 0.6662 |
| halophile | 0.1182 | 0.9423 | 0.9420 | −0.0003 | 0.7522 | 0.7499 |

**Locked:** scope=secreted, tier=H+M for all six phenotypes.

---

## Table 3 — λ sweep (job 1176405): AUROC and pair-AUC

Grid λ ∈ {0, 0.5, 1, 2, 4} × 6 phenotypes at locked scope=secreted, tier=H+M, seed 1466,
mean-pooled cached-probe heads.

**AUROC grid** (bold = row max):

| pheno | λ0 | λ0.5 | λ1 | λ2 | λ4 |
|---|---|---|---|---|---|
| psychrophile | **0.9057** | 0.8988 | 0.8967 | 0.8956 | 0.8965 |
| thermophile | **0.9693** | 0.9691 | 0.9688 | 0.9687 | 0.9688 |
| hyperthermophile | **0.9990** | 0.9990 | 0.9989 | 0.9990 | 0.9990 |
| acidophile | **0.9838** | 0.9830 | 0.9832 | 0.9832 | 0.9827 |
| alkaliphile | **0.9785** | 0.9771 | 0.9770 | 0.9769 | 0.9766 |
| halophile | **0.9470** | 0.9436 | 0.9421 | 0.9407 | 0.9367 |

**pair-AUC grid** (bold = row max):

| pheno | λ0 | λ0.5 | λ1 | λ2 | λ4 |
|---|---|---|---|---|---|
| psychrophile | 0.5606 | 0.5884 | 0.5969 | **0.6071** | 0.6007 |
| thermophile | 0.9067 | 0.9072 | **0.9095** | 0.9089 | 0.9058 |
| hyperthermophile | 0.9150 | 0.9531 | 0.9541 | 0.9551 | **0.9590** |
| acidophile | 0.7996 | 0.7966 | **0.8071** | 0.8014 | 0.8003 |
| alkaliphile | 0.7371 | 0.7416 | 0.7405 | **0.7518** | 0.7482 |
| halophile | 0.7718 | **0.7744** | 0.7702 | 0.7691 | 0.7675 |

**Key finding:** AUROC is maximized (or tied ≤0.0003) at λ=0 for all six; pair-AUC prefers
λ>0 for all six. λ locked by pair-AUC (deployment = pair-ranking): psychrophile 2.0,
thermophile 1.0, hyperthermophile 4.0, acidophile 1.0, alkaliphile 2.0, halophile 0.5.

---

## Table 4 — Mean vs attention pooling: AUROC and pair-AUC

Each head at its pair-AUC-locked λ; scope=secreted, tier=H+M, seed 1466. Mean-pool
reference = `lam_sweep_all.json` at the same locked λ. Δ = attention − mean.

| Phenotype | λ | Attn AUROC | Attn pair-AUC | Mean AUROC | Mean pair-AUC | Δ AUROC | Δ pair-AUC |
|---|---|---|---|---|---|---|---|
| psychrophile | 2.0 | 0.8403 | 0.597 | 0.8956 | 0.6071 | −0.0553 | −0.0101 |
| thermophile | 1.0 | 0.9610 | 0.888 | 0.9688 | 0.9095 | −0.0078 | −0.0215 |
| hyperthermophile | 4.0 | 0.9981 | 0.8916 | 0.9990 | 0.9590 | −0.0009 | −0.0674 |
| acidophile | 1.0 | 0.9769 | 0.7807 | 0.9832 | 0.8071 | −0.0063 | −0.0264 |
| alkaliphile | 2.0 | 0.9684 | 0.7164 | 0.9769 | 0.7518 | −0.0085 | −0.0354 |
| halophile | 0.5 | 0.9331 | 0.7524 | 0.9436 | 0.7744 | −0.0105 | −0.0220 |

**Verdict:** mean pooling wins on BOTH metrics for all six phenotypes — attention is never
better. Mean pooling is the production choice.
