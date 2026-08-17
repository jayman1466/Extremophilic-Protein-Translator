# Lab Notebook — Extremophilic Protein Translator

A running log of pipeline development: steps taken, key metrics, and figures.
Newest entries are appended at the bottom. For *what the pipeline is*, see
[README.md](README.md); this file records *what was actually done and found*.

- **Project:** database of secreted proteins from GTDB extremophiles for
  protein-language-model fine-tuning.
- **Data release:** GTDB **r232**.
- **Compute:** biotite SLURM cluster (`/groups/cress/projects/jaymin/IS1111/`);
  data pre-downloaded and pre-parsed. SignalP 6.0 installed on biotite.

---

## Summary metrics (running)

| Metric | Value |
|--------|-------|
| GTDB release | r232 |
| Species representatives (total) | **199,923** |
| — Bacteria | 189,801 |
| — Archaea | 10,122 |
| Reps with `ncbi_isolation_source` | 94,977 (48%) |
| Reps with isolation-source extremophile flag | 9,492 |
| Reps with organism-name extremophile signal | 12,193 |

---

## Stage 01 — GTDB indexing

**Goal:** parse GTDB r232 metadata, filter to species representatives, and
build accessors mapping each genome to its on-disk proteome / genome files.

**What was done**
- Verified the on-biotite data layout matches the documented conventions
  (`gtdb/` metadata + per-genome `.faa.gz`/`.fna.gz`, `work/` combined FASTA +
  coords + genome index). All file-format conventions confirmed:
  - genome id keeps GTDB prefix (`GB_`/`RS_`); combined FASTA headers are
    `>{GENOME}~{PROTID}`; `genome_index.tsv` is headerless `<bare_acc>\t<abs_path>`.
- Wrote `src/eptrans/gtdb.py`: metadata parsing, representative filter,
  taxonomy expansion (`d__…;p__…` → per-rank columns), and a `GenomeIndex`
  accessor that resolves an accession to its `.faa.gz` and `.fna.gz` paths.
- `scripts/01_index_gtdb.py` produces `results/gtdb_reps_metadata.parquet` and a
  per-phylum bar chart.

**Key metrics**
- **199,923** species representatives (189,801 bacteria + 10,122 archaea) —
  internally consistent with the 199,923-row `genome_index.tsv` (1:1 coverage).
- Metadata: 113 columns; environment proxy is `ncbi_isolation_source` (col 62).
- QC fields available: `checkm2_completeness`, `checkm2_contamination`
  (defaults: completeness ≥ 90, contamination ≤ 5).

**Figure** — representatives per phylum (validation sample; the full run
produces the same over all reps):

![GTDB representatives per phylum (sample)](results/gtdb_reps_sample_per_phylum.png)

**Validation:** parser tested on a staged r232 metadata sample (51 reps),
robust to embedded newlines in free-text NCBI fields. 6 unit tests pass
(`tests/test_gtdb.py`).

---

## Stage 01b — Metadata-based extremophile flagging

**Goal:** map isolation-source (and, weakly, organism-name) text to candidate
extremophile classes, with retained evidence strings for auditability.

**What was done**
- Built a curated keyword dictionary in `src/eptrans/binning.py`, grounded in
  the *actual* r232 isolation-source vocabulary (tabulated the most common
  values first). Classes: thermophile, hyperthermophile, psychrophile,
  acidophile, alkaliphile, halophile.
- Handled deliberate ambiguities:
  - `soda lake` → **halophile + alkaliphile** (hypersaline *and* alkaline)
  - `salt marsh` (tidal) → **not** halophile; `basalt` → not halophile
  - `cold seep` → **not** psychrophile (ambient deep-sea, not cold-adapted)
  - `hydrothermal sulfide chimney` → thermophile + hyperthermophile
- Organism-name signals collected **separately and down-weighted**, because
  genus names correlate with clade and would leak phylogeny into the label.
- Every match retains the substring that triggered it (`*_evidence` columns).
- `scripts/01b_flag_metadata.py` → `results/metadata_flags.parquet` + figure.

**Key metrics** (of 199,923 reps; isolation-source evidence)

| Class | isolation-source | organism-name |
|-------|-----------------:|--------------:|
| thermophile | 4,415 | 3,690 |
| hyperthermophile | 949 | 52 |
| psychrophile | 1,454 | 681 |
| acidophile | 1,999 | 6,649 |
| alkaliphile | 1,394 | 422 |
| halophile | 1,617 | 1,642 |

**Figure** — metadata flags by class (left: isolation source split by domain;
right: isolation-source vs organism-name evidence):

![Metadata extremophile flags](results/metadata_flags_counts.png)

**Observation / design validation:** for **acidophile**, organism-name evidence
(6,649) far exceeds isolation-source evidence (1,999) — largely the genus
`Acidobacterium` / class `Acidobacteriae`, i.e. taxonomic names rather than
habitat. This is exactly the clade-vs-trait confound the pipeline guards
against, and why organism-name signals are never used alone for a confident
label. Archaea are enriched in thermophile / hyperthermophile / halophile
classes, as expected.

---

## Stage 02 — GenomeSPOT wrapper + prediction parsing

**Goal:** wrap GenomeSPOT to predict temperature / pH / salinity / oxygen from a
genome's DNA + protein FASTA, and parse its output into a tidy per-genome table.

**What was done**
- Installed GenomeSPOT (Barnum et al. 2024) into a dedicated `genomespot` env
  with the **critical** pins: `scikit-learn==1.2.2` (README: essential — models
  mispredict on other versions) and `hmmlearn==0.3.0`.
- Wrote `src/eptrans/genomespot.py`: subprocess wrapper around
  `python -m genome_spot.genome_spot` + parsers for `.predictions.tsv` /
  `.predictions.json`. Flattens 10 targets × {value, error, is_novel, warning}
  into a per-genome row, plus a `__suspect` flag per trait.
- Handled interpretation subtleties: `is_novel` (features unusual vs 98% of
  training data) and `warning` clamping — a `min_exceeded` on salinity min /
  optimum at 0 is benign and explicitly **not** flagged suspect.
- `scripts/02_run_genomespot.py`: serial/local driver over an accession list
  using the `GenomeIndex` accessors.

**Validation** — ran on the shipped test genome `GCA_000172155.1`; output
reproduces the README reference exactly:

| target | value | error | flag |
|--------|------:|------:|------|
| temperature_optimum | 22.95 °C | 6.48 | — |
| ph_optimum | 7.07 | 0.91 | — |
| salinity_optimum | 0.20 % w/v | 1.94 | — |
| salinity_min | 0 | 1.18 | min_exceeded (benign) |
| oxygen | tolerant | 0.974 | — |

6 parser tests added; **32 tests total passing**. GenomeSPOT repo + models
archived as an artifact (`genomespot_repo.tar.gz`) for reproducible reuse.

---

## Stage 02b — Reconcile precomputed GenomeSPOT predictions

**Goal:** reuse the GenomeSPOT paper's precomputed predictions (GTDB **r214**)
for r232 reps, compute the recompute **delta**, and emit a reconciled TSV with
genome absolute paths.

**What was done**
- Established that the paper applied GenomeSPOT to GTDB **r214** (abstract:
  "all 85,205 species"). The per-genome predictions are **not** in the repo; the
  `analyze_all_species` notebook reads them from a local `data/predictions_gtdb/`
  and merges to a table it calls `supplementary_data_4.tsv` (presumed the paper's
  Supplementary Data 4).
- Wrote `src/eptrans/reconcile.py` with **three-tier accession matching**
  (strongest first), recording which level matched for auditability:
  1. `exact` — full bare accession incl. source prefix + version
  2. `noversion` — GCA/GCF + numeric, ignoring `.version` (version bumps)
  3. `assembly` — numeric id only (bridges GenBank↔RefSeq `GCA`↔`GCF`)
- r232 reps with no match → **delta** (need fresh GenomeSPOT); precomputed rows
  matching no r232 rep → **dropped** (organism no longer a rep).
- `attach_genome_paths()` adds absolute `.fna.gz` paths from `genome_index.tsv`.
- `scripts/02b_reconcile_predictions.py` emits the reconciled TSV (+ delta
  accession list, stats JSON, summary figure).

**Validation** — the true reuse fraction needs the real Supp Data 4 (bioRxiv
supplementary blocked from this environment; pending user download). Verified
end-to-end on a **synthetic precomputed set** built from real r232 accessions
with injected version-bumps and GCA↔GCF swaps (75% coverage, 3,000 dropped rows):

| bucket | count | note |
|--------|------:|------|
| reuse: exact | 89,965 | identical accession |
| reuse: noversion | 29,989 | version bump |
| reuse: assembly | 29,988 | GCA↔GCF swap |
| delta (recompute) | 49,981 | not in precomputed |
| dropped precomputed | 3,000 | not a rep in r232 |

All 199,923 genome paths attached. 8 reconcile tests added; **40 tests total**.

![Reconciliation summary (synthetic validation)](results/genomespot_reconcile_summary_SYNTH.png)

> **Note:** figure/counts above are the synthetic validation. Real r214→r232
> reuse fractions will be produced once Supplementary Data 4 is loaded.

---

## Stage 03 — Combined environmental binning logic

**Goal:** reconcile the two independent evidence sources (metadata flags +
GenomeSPOT predictions) into a final per-class extremophile label with a
confidence tier, plus a `confident_mesophile` flag for outgroup selection.

**Combination rule** (`eptrans.binning.combine_label`):

| metadata | prediction | → label | confidence |
|----------|-----------|---------|-----------|
| class X | class X | X | **high** (agree) |
| — | class X | X | **medium** (prediction only) |
| class X | — / conflict | X | **low** |
| — | — | (none) | none |

- `confident_mesophile` = all predicted optima inside the mesophile envelope
  (temp 20–40 °C, pH 6–8, salinity ≤3 % w/v) **and** no metadata extremophile
  flag — the pool from which phylogenetically-matched outgroups are drawn.
- `scripts/03_combine_bins.py` writes `combined_labels.{parquet,tsv}` with
  per-class booleans, final label, confidence, and mesophile flag; plus two
  figures (class counts by tier; metadata-vs-prediction agreement per class).

**Validation** (metadata flags + *synthetic* predictions, n=199,923): pipeline
runs end-to-end; with random synthetic predictions "both agree" is minimal and
"prediction only" dominates (expected for random data). Real GenomeSPOT
predictions will concentrate agreement in the metadata-flagged classes. Figures:

![Combined labels by confidence (synthetic)](results/combined_label_counts_SYNTH.png)
![Evidence agreement per class (synthetic)](results/combined_agreement_SYNTH.png)

> Counts are synthetic-prediction placeholders; final numbers await real
> GenomeSPOT predictions (Supp Data 4 + delta recompute).

---

## Stage 04 — Phylogenetically-controlled genome selection

**Goal:** satisfy the two competing methodological constraints so the downstream
model learns the *trait*, not the *clade*:
1. **Diversity** — extremophiles for each class span the tree (cap picks per
   lineage so no clade dominates).
2. **Matched outgroups** — each extremophile paired with a phylogenetically
   *close* confident mesophile (same genus → family → … ), so extremophile and
   mesophile can't be separated by clade alone.

**What was done** (`src/eptrans/selection.py`)
- `select_extremophiles()`: diversity cap of N per lineage-rank (default 5 per
  family), preferring high-confidence labels.
- `find_outgroup()`: walks genus → family → order → class → phylum, returning
  the closest unused confident mesophile and recording the matched rank.
- `select_with_outgroups()`: full pipeline → extremophiles, outgroups, pairs.
- `scripts/04_select_genomes.py` writes the four tables + two figures.

**Validation** (synthetic combined labels; 100 per class, cap 5/family):

| metric | value |
|--------|------:|
| extremophiles selected | 600 |
| outgroups matched | 598 / 600 |
| pairs sharing **genus** | 302 |
| pairs sharing **family** | 221 |
| pairs sharing order/class/phylum | 41 / 25 / 9 |

Selected extremophiles per class span **21–36 distinct phyla** and ~88–96
distinct genera — high diversity with tight per-pair clade matching.

![Pair phylogenetic closeness (synthetic)](results/selection_match_ranks_SYNTH.png)
![Extremophile diversity per class (synthetic)](results/selection_phylum_spread_SYNTH.png)

8 selection tests added; **48 tests total**.

---

## Stage 05 — SignalP 6.0 wrapper + secreted-protein extraction

**Goal:** identify and extract secreted / cell-surface-exposed proteins (those
carrying a signal peptide) from the selected genomes' proteomes.

**Secreted definition:** SignalP 6.0 class ∈ {SP, LIPO, TAT, TATLIPO, PILIN}
(i.e. any predicted signal peptide; class `OTHER` = not secreted).

**Host reconnaissance (biotite)**
- SignalP 6.0 is a **pyenv shim** (`/home/jayminp/.pyenv/shims/signalp6`,
  pyenv 3.11.3), **not** on the default non-login PATH. Jobs must
  `export PYENV_ROOT="$HOME/.pyenv"; export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"` first.
- Confirmed CLI:
  `signalp6 --fastafile <faa> --output_dir <dir> --format {txt,none} --organism other --mode {fast,slow}`.
- ⚠️ **Model weights are not installed** — `signalp/model_weights/` contains only
  a README (no `.pt` files), so no mode can currently run (fast mode errors on the
  missing distilled model). The weights are license-gated (DTU academic download)
  and must be installed on the host before real runs.

**What was done**
- `src/eptrans/signalp.py`: parser for `prediction_results.txt` (exact format
  taken from the installed SignalP source), `SignalPrediction` dataclass with
  `is_secreted`, `extract_secreted()` (precursor or mature chain via cleavage
  site), `summarize()`, and command builder.
- `scripts/05_run_signalp.py`: `--emit-slurm` writes a SLURM array job (pyenv
  export baked in; decompresses `.gz` proteomes; one genome per array task); the
  default mode parses per-genome outputs → secreted-protein FASTA
  (`>{GENOME}~{PROTID}` headers) + per-protein table + summary JSON.

**Validation** (parser + extraction on a synthetic SignalP output set that
reproduces the exact 6.0 format): 2 genomes, 5 proteins → 3 secreted (60%);
class filtering, cleavage-site math (mature = residues after cleavage), and
`{GENOME}~{PROTID}` headers all correct. `--emit-slurm` generated a 600-task
array over the selected extremophiles. 7 signalp tests; **55 tests total**.

---

## Stage 06 — labeled dataset assembly with leakage-aware splits

**Goal:** join secreted proteins (stage 05) to genome environmental classes
(stage 03), producing a protein-level supervised dataset for PLM/classifier
fine-tuning, with splits that don't leak homologs across train/val/test.

**What was done**
- `src/eptrans/dataset.py`: `assign_labels()` (each secreted protein inherits its
  source genome's class, or `mesophile` for a confident-mesophile outgroup),
  `stratified_group_split()` (whole groups → one split, stratified by group
  majority label), `assemble_dataset()` (groups = mmseqs sequence clusters when a
  cluster map is supplied, else genome-level fallback).
- **Two leakage controls**: (1) sequence-similarity — near-duplicate secreted
  proteins (mmseqs cluster) never span splits; (2) genome memorization — all
  proteins of one genome share a split. Enforced by assertion
  (`max_splits_per_group == 1`).
- `scripts/06_assemble_dataset.py`: dataset parquet/TSV + stats.json + per-split
  count figure.

**Validation:** 5 dataset tests including the no-leakage guarantee and the
cluster-map homolog test (a shared cluster lands entirely in one split).
(Full suite after all stages below: **65 tests passing**.)

---

## Stage 05b — SLURM scaling under biotite QOS limits

**biotite `standard` QOS (verified via scontrol/sacctmgr):** MaxJobsPerUser=10
(running array tasks), MaxSubmitJobsPerUser=200 (queued+running), MaxArraySize=1001.

- `src/eptrans/slurm.py`: `plan_array()` sizes chunked arrays so n_tasks ≤ 200,
  throttled `%10`; auto-grows chunk-size to fit.
- `scripts/05_run_signalp.py` and `scripts/02_emit_genomespot_slurm.py` both emit
  chunked arrays: GenomeSPOT uses `xargs -P <cpus>` intra-node (1 genome/invocation,
  ~5s each); SignalP concatenates a chunk's proteomes into one FASTA (headers
  namespaced `GENOME~PROTID`) and uses internal batching (`--torch_num_threads`,
  `--write_procs`, `--bsize`).
- Full scale (199,923 genomes): 200 tasks × 1000/task × 16 cores, `--array=0-199%10`.
- **Compute-time note:** GenomeSPOT ~5s/genome. At 16 cores/task × 10 running
  tasks (160-way) → ~105 min wall for all genomes; at 48 cores/task → ~35 min.
  Reusing the paper's precomputed r214 predictions (Supp Data 4) would cut the
  recompute to the delta (~4× saving), but recompute is cheap CPU work — the user
  chose the recompute path.

---

## Stage 07 — end-to-end local pilot

**Goal:** run the whole chain on a small, phylogenetically-diverse real genome
set and confirm each stage produces sensible output.

**Pilot set** (`data/pilot_genomes.tsv`, 14 genomes): 2 each of hyperthermophile,
psychrophile, acidophile, alkaliphile, halophile + 1 thermophile + 3 mesophiles,
spanning 4 GTDB phyla (Acidobacteriota, Actinomycetota, and two candidate phyla),
with real isolation sources (deep-sea hydrothermal vent, ice core, acid mine
drainage, hypersaline soda lake, alkaline hot spring). All 14 confirmed in the
genome index (fna + proteome paths).

**GenomeSPOT (real predictions):** ran locally on all 14 (repo + models + pinned
env), 14/14 in ~45 s. Predictions are biologically sensible and demonstrate the
value of the combination approach:
- Acidophiles (acid mine drainage) → pH optima **3.4 / 5.0** ✓
- Halophile (hypersaline soda lake) → salinity **6.0 %**, pH **9.2** ✓
- Psychrophile (ice core, *Cryobacterium*) → temp optimum **17.4 °C** ✓
- **"Hyperthermophiles"** (metadata: deep-sea vent) → GenomeSPOT predicts only
  **40–48 °C**, *not* hyperthermophilic — the "isolated-from ≠ thrives-at" case
  the design guards against. These correctly fall to low confidence.

**Combined binning (real):** 14 genomes → confidence tiers **high 3 / low 8 /
none 3**; 1 confident mesophile. High-confidence = the acidophiles + halophile
where metadata and prediction agree.

**Selection:** at 14-genome scale only 1 confident mesophile exists, so only 1
extremophile could be phylo-matched to an outgroup (logic exercised; matched
pool is a scale artifact of the pilot, not a defect).

**SignalP (real):** the full pilot proteome is **47,972 proteins**; SignalP 6.0
fast mode runs at ~1.7 seq/s on CPU (~8 h for the full set), so the pilot
subsamples to 150 proteins/genome (**2,100 total**) and runs as a SLURM batch job
(job 1149752, `COMPLETED` in 3:57 on a `standard` node, 16 threads). Result:
**323 / 2,100 secreted (15.4 %)** — a biologically reasonable signal-peptide
fraction. By class: **208 SP (Sec/SPI), 103 LIPO (Sec/SPII), 6 TAT, 6 TATLIPO,
0 PILIN**. All 14 genomes contribute secreted proteins (4–29 % per genome).
(SignalP model weights confirmed installed by the user.)

**Dataset assembly (real):** 268 secreted proteins across 12 genomes labeled by
environmental class (hyperthermophile 66, thermophile 65, psychrophile 36,
halophile 34, acidophile 33, alkaliphile 19, mesophile 15). Leakage check passed
(no genome spans multiple splits). At pilot scale (~2 genomes/class) the
stratified group split puts everything in train — `round(2 × 0.1) = 0` for
val/test — which is expected small-n rounding, not a logic error: a 30-genome
demo yields the correct 120/15/15 train/val/test with zero leakage.

**Pilot figures**
- `results/pilot_env_predictions.png` — predicted OGT / pH / salinity per genome,
  colored by metadata-implied class, with class-threshold guide lines.
- `results/pilot_phylo_spread.png` — phylum × class grid colored by combined
  confidence tier.
- `results/pilot_combined_label_counts.png`, `results/pilot_combined_agreement.png`
  — per-class label counts and metadata/prediction agreement.
- `results/pilot_secreted_counts.png` — SignalP secreted fraction per genome,
  colored by class.
- `results/pilot_dataset_splits.png` — labeled-dataset class counts per split
  (all-train at pilot scale; see note above).

## SignalP throughput benchmark (measured)

Benchmarked SignalP 6.0 fast mode on 10,000 proteins across CPU / A5000 / H200:

![SignalP throughput: CPU vs A5000 vs H200 across batch sizes](results/signalp_gpu_benchmark.png)

| Config | best seq/s | per-GPU 24 h |
|--------|-----------:|-------------:|
| H200 (bsize 32) | **12.75** | 1.10 M |
| A5000 / `gpu` (bsize 64) | 9.60 | 0.83 M |
| CPU `standard` (16 threads) | 8.86 | 0.77 M |

**Key finding: GPU gives only ~1.4× over CPU (H200), ~1.1× (A5000).** SignalP 6.0
fast mode is **CPU-decode-bound**, not GPU-bound — the transformer runs on short
N-terminal windows, so the GPU idles while CPU-side CRF decoding and result
assembly set the pace. Larger batch sizes were *slower* (padding waste). **Verdict:
run SignalP on `standard` (CPU), not the GPU partitions** — no speedup, worse queue.

**Scale consequence:** all 199,923 reps ≈ 685 M proteins → 30–80 days at any of
these rates (infeasible). The phylo-controlled **selection** stage exists to avoid
this: SignalP runs only on the selected subset (~100 genomes/class × 6 + outgroups
≈ 1–1.5 k genomes ≈ 4–5 M proteins). At the `standard` CPU node's measured
480-way rate (~266 seq/s), that is **~4–5 h**; the GPU partitions are no faster
per unit and have worse queues, so `standard` is the right target. Production
order: GenomeSPOT-all → bin → **select** → SignalP-on-selected.

**End-to-end status:** the full chain runs — GTDB metadata → flag → GenomeSPOT →
combine → select → SignalP → labeled dataset — on real data, producing a
268-protein labeled secreted-protein dataset from 12 genomes (of the 14 pilot
genomes; 2 fell outside the labeled classes). At production
scale the same scripts run via the chunked SLURM emitters.
- **Stage 06** — labeled dataset assembly (leakage-aware splits).
- **Pilot** — end-to-end run on a small genome set + report.

## Full-scale production run (GenomeSPOT complete)

- **GenomeSPOT on all 199,923 r232 reps — DONE.** SLURM array (20 tasks ×
  10,000 genomes × 48-way `xargs`, `%10` throttle; job 1149841). All 20 chunks
  reported "done (10000 outputs)"; queue empty. Per-task ~46 min (~13 s/genome
  effective on real genomes, higher than the 5 s test genome). Outputs:
  `eptrans_scratch/genomespot/chunk_*/<acc>.predictions.tsv` (per-genome
  long-format, 10 targets).
- **Aggregation** (`aggregate_genomespot.py`, SLURM job 1149958): pivots the
  ~200 k per-genome long-format files to one wide row each (10 targets ×
  value/error/warning), joins genome absolute `.fna.gz` paths from
  `genome_index.tsv` (this folds in the 02b path-attachment), writes
  `genomespot_predictions_r232.tsv` with headers. Reading 200 k tiny files off
  VAST is I/O-bound; run as a batch job, not interactive SSH.

## Phase 1 localization enrichment

Per user (retain signal type + mature chain; defer TM topology): added an
`anchoring` field derived from SignalP class — **soluble** (SP, TAT: Sec/Tat
cleaved, released) vs **membrane_anchored** (LIPO, TATLIPO, PILIN: lipobox /
pilin, extracellular-facing but membrane-tethered) vs **none** (OTHER:
cytoplasmic or transmembrane — indistinguishable without a topology tool). The
secreted per-protein table now carries `signalp_class, anchoring, cs_after,
cs_prob` + per-class probs; `dataset.assign_labels` passes these through. TM
extracellular-loop extraction (DeepTMHMM/Phobius) is deferred to a later phase.

_Infrastructure notes: biotite SSH + scratch dir + GitHub credential all
configured. Job submission via SLURM (`standard`/`memory`/`gpu` partitions).
biotite SSH latency is intermittently high — trivial `squeue`/`find` calls can
hang past the 560 s ceiling; prefer batch jobs over interactive SSH for anything
touching the ~200 k-file GenomeSPOT output tree._

## Full-scale binning + final 5-class selection (r232)

**Combined binning** (`03_combine_bins.py --bare-join`) on the real
GenomeSPOT predictions for all **199,923** genomes. The metadata flags keep the
GTDB `GB_`/`RS_` prefix while the aggregated GenomeSPOT TSV uses bare
accessions, so the merge normalizes both sides to the bare form (`--bare-join`).
Confidence tiers: **high 3,638** (metadata + prediction agree), **medium
18,863** (prediction-only), **low 5,499** (metadata-only / conflict), none
171,923; **104,486 confident mesophiles** form the outgroup pool.

![Combined labels by confidence tier (r232, all 199,923 genomes)](results/combined_label_counts_r232.png)
![Metadata vs prediction agreement per class (r232)](results/combined_agreement_r232.png)

**Final selection** (`04_select_genomes.py`) — 5 overlapping classes, chosen
with the user for per-phenotype independent models:

| class | selected | high | medium | phyla | genera |
|---|--:|--:|--:|--:|--:|
| acidophile | 583 | 223 | 360 | 37 | 436 |
| alkaliphile | 476 | 101 | 375 | 32 | 385 |
| halophile | 2,206 | 421 | 1,785 | 81 | 1,680 |
| thermophile (≥50 °C, high-only) | 1,171 | 1,171 | 0 | 88 | 901 |
| hyperthermophile (≥80 °C) | 216 | 76 | 140 | 25 | 161 |

Rules: high+medium confidence (thermophile restricted to high-only to fit the
wall-clock budget), cap 3 genomes/family for diversity, mesophile outgroups
reused across classes (deduplicated). **4,498 unique extremophiles + 2,773
outgroups = 7,271 genomes ≈ 24.9 M proteins.** Diversity is strong in every
class (25–88 phyla, top-phylum share 16–25 %) except hyperthermophile
(49 % Thermoproteota — real biology; hyperthermophily is concentrated in a few
archaeal lineages, which is exactly why its matched mesophile outgroups matter).

Decisions on the hyperthermophile "low" tier: **rejected** (median predicted
optimum 35 °C — isolated-from-hot but not predicted thermophilic; these are the
false positives the combination approach is designed to catch).

**Confidence retained end-to-end** (user: high/medium tiers will weight training
data after SignalP): extremophiles/outgroups TSVs carry `final_confidence`, the
pairs table records both `extremophile_confidence` and `outgroup_confidence`,
and `dataset.assign_labels` stamps `label_confidence` on every secreted protein.

![Final selection summary — genomes/proteins per class (r232)](results/selection_r232_summary.png)
![Matched-pair phylogenetic closeness by rank (r232)](results/selection_r232_match_ranks.png)
![Extremophile phylogenetic diversity per class (r232)](results/selection_r232_phylum_spread.png)

## SignalP production run (job 1149978)

SignalP 6.0 fast mode on all 7,271 selected genomes: **10 array tasks × 750
genomes × 48 threads** (`--array=0-9%10`, standard partition, `--time
72:00:00`, `--mem 48G`, `--bsize 32`), out-root
`eptrans_scratch/signalp_r232`. Estimated ~26 h at ~266 seq/s (480-way);
72 h ceiling is a safety margin (user request). GPU gives no speedup — SignalP
fast mode is CPU-decode-bound (benchmark: H200 only ~1.4× over 16-thread CPU).

_Transfer note: `host.compute.upload()` fails on the /groups VAST mount, and
base64-as-command-argument silently drops large payloads (>~8 KB). Reliable
method: base64 the file, append in ≤6 KB chunks (`printf %s <chunk> >>
file.b64`), then `base64 -d`._

## Reference databases (function-retention oracle)

**Already staged on biotite** (system-wide `/shared/db`, recorded for provenance):

| database | release / pull | path | size |
|---|---|---|---|
| UniRef50 (FASTA) | 2025-11-13 | `/shared/db/uniref/uniref50/latest/uniref50.fasta` | 23 G |
| UniProtKB mmseqs DB (Swiss-Prot + TrEMBL) | 2025_01 | `/shared/db/uniprot/latest/mmseqs/` | 137 G |
| Pfam-A HMMs | r37 | `/shared/db/pfam/latest/Pfam-A.hmm` | 3.4 G |
| Foldseek PDB DB | 2026-02-04 | `/shared/db/foldseek/latest/db/pdb` | 71 M |
| Foldseek AlphaFold DB | 2026-02-04 | `/shared/db/foldseek/latest/db/alphafold_uniprot` | 75 G |

Note: the UniProt mmseqs DB stores **sequences only** — it does not carry the
`ACT_SITE`/`BINDING`/`METAL` feature tables, so the Swiss-Prot flat file is still
required for active-site annotations.

**Staged for this project** (`scripts/download_dbs.sh`, `scripts/download_mcsa.py`):

- **M-CSA** (snapshotted 2026-07-09 via API — flat files are frozen at EBI):
  1,003 entries → **5,201 catalytic-residue rows**, 991 distinct UniProt
  accessions with positions + residue codes + PDB id/chain + catalytic roles, all
  7 EC classes. Saved as `mcsa_catalytic_residues.tsv` (+ full `mcsa_entries.json.gz`).
  This is the tier-1/2 reference for the active-site ladder (direct match +
  homolog transfer).
- **Swiss-Prot flat file** (`uniprot_sprot.dat.gz`, ~950 MB gz): the annotation
  copy with ACT_SITE/BINDING/METAL features. Download on biotite.
- **InterPro** (`entry.list` + `interpro.xml.gz`; optional huge `protein2ipr.dat.gz`
  ~50 GB for full local position mapping — otherwise use the protein-annotation
  MCP connector per-enzyme). Download on biotite.

### Download completion (2026-07-09)

Staged into `eptrans_scratch/db/` (to be relocated to a persistent reference dir):

| database | version | files | size |
|---|---|---|---|
| Swiss-Prot flat file | Release **2026_02** (10-Jun-2026), 575,503 reviewed entries | `uniprot_sprot.dat.gz` (699 MB) + `reldate.txt` | 667 M |
| InterPro | current (54,191 entries) | `entry.list` (2.9 MB) + `interpro.xml.gz` (42 MB) | 43 M |

InterPro `protein2ipr.dat.gz` (~50 GB) intentionally NOT downloaded — per-enzyme
annotation goes through the `mcp-protein-annotation` connector instead.

**Pending relocation** (deferred until project wraps): move `db/` out of scratch
to a persistent reference location; sweep durable scratch products
(`genomespot_predictions_r232.tsv`, parsed SignalP secretome) to persistent
storage; leave `chunk_*` intermediate trees in scratch.

## SignalP merge → master secretome (job 1151569)

The 10 SignalP chunks (job 1149978) were merged into one secretome table +
mature-chain FASTA by `scripts/merge_signalp_chunks.py` (stdlib-only, SLURM job
1151569, `--mature`).

| metric | value |
|---|---|
| proteins scanned | 17,603,649 |
| secreted classified (class ≠ OTHER) | 1,985,565 (11.3%) |
| secreted written to FASTA | **1,985,508** (57 fewer: dropped empty/degenerate mature chains) |
| genomes | 7,268 |
| by class (of 1,985,565 classified) | SP 1,270,289 · LIPO 570,012 · TAT 78,884 · PILIN 49,691 · TATLIPO 16,689 |

Outputs in `IS1111/eptrans/`: `secreted_proteins_r232.tsv` (12 cols, 221 MB),
`secreted_proteins_r232.faa` (mature chains, headers `GENOME~PROTID class=… anchoring=…`, 865 MB).

## Stage 07 — mmseqs clustering for leakage-controlled splits (job 1151585)

`mmseqs easy-cluster` at **50% identity / 80% coverage** (cov-mode 0) on the
secreted mature chains → cluster map used as the split grouping (no homolog
leaks across train/val/test).

| metric | value |
|---|---|
| input proteins | 1,985,508 |
| clusters | 1,253,362 |
| redundancy | 1.58 members/cluster (1.00M singletons, max 987) |

Output: `secreted_clusters_r232_cluster.tsv` (`cluster_rep<TAB>member`, 156 MB).

## Stage 06 (production) — cluster-level labeled dataset

`06_assemble_dataset.py` in cluster regime: split on **clusters** (drop the
genome union-find; clusters already co-locate matched orthologs). Per-protein
`label_confidence` (high/medium/none) retained for training-time weighting.

| metric | value |
|---|---|
| proteins | 1,985,508 |
| split | train 1,590,716 · val 198,322 · test 196,470 |
| derived ortholog pairs (`L_pair`) | **90,984**, 100% same-split by construction |
| pairs by class | halophile 63,846 · thermophile 15,604 · alkaliphile 6,314 · acidophile 5,051 · hyperthermophile 169 |
| extremophile / mesophile proteins | 1,044,442 / 941,066 |

Outputs: `results/labeled_dataset_r232_clustered.parquet` (18 cols),
`_protein_pairs.tsv` (side-car index for the pairwise margin loss).

![Production labeled dataset — class × split, confidence, anchoring, pairs (r232)](results/dataset_production_r232.png)

## Modeling — design + training scaffold

Full design in `docs/modeling_design.md`. Goal: input an enzyme, output the same
activity on a more extremophilic scaffold. Two-stage, per-phenotype design
(§11): **one shared LoRA-adapted ESM-2 3B backbone** (label-agnostic MLM domain
adaptation) → **5 per-phenotype classifier heads** (thermophile,
hyperthermophile, acidophile, alkaliphile, halophile), each vs matched
mesophiles.

`src/eptrans/modeling/` (env `eptrans-ml`: torch / transformers 5.13 / peft 0.19):
- **`masking.py`** — §13 conservation-weighted mask `(1−c)^γ`; active-site freeze
  = γ→∞ limit; **coupling-aware masking** (span + contact-pair via ESM-2's contact
  head) so disulfides/salt-bridges/local structure are masked jointly.
- **`losses.py`** — §12 confidence-weighted masked CE (+ KL guard); weighted/focal
  BCE; matched-pair margin; `L_cls = L_BCE + λ·L_pair`.
- **`model.py`** — LoRA ESM-2 3B, full-attention targets (q/k/v + attention-output
  dense), rank 32 / α 64; mean-pool classifier head.
- **`data.py`** — FASTA join by `tagged_id`; MLM / classifier / pair datasets;
  negative-sampling cap (3× positives); sliding-window truncation guard
  (65,199 chains > 1022, max 30,084).
- **`train.py`** — MLM (bf16 autocast, length-bucket batching, early-stop on val
  pseudo-perplexity) + classifier (model-select on val AUPRC, pair co-loading).

CLI `scripts/08_train_backbone.py` (mlm | classifier), SLURM
`scripts/slurm/08_train_backbone.sbatch`. 27 modeling + 72 pipeline tests pass.

## Stage-1 MLM launch (job 1151972)

Cluster-stratified subsample (`scripts/09_subsample_mlm.py`): dedupe to one
representative per cluster, then **400k train + 20k fixed val**. Domain
adaptation only needs the dominant distributional shift, which a rank-32 LoRA
extracts from a de-redundant fraction of the 1.59M corpus (≈4–5 h/epoch on one
H200 vs ~18 h for the full set). All 420k join cleanly to the mature FASTA;
median mature length 269 aa.

Config: ESM-2 3B, LoRA rank 32 / α 64, bf16, 3 epochs, keep best-by-val-PPL,
gpu:1 on `gpu_h200`. ESM-2 3B weights (11 GB) pre-staged to `$HF_HOME` on the
login node. Submitted as **job 1151972**.

## Background master-secretome fill (job 1151578)

190-task array (`0-189%5`, ~1014 genomes/task) applying SignalP to the remaining
192,652 GTDB representatives, building a whole-GTDB master secretome over ~1–2
months without crowding foreground work. Resumable (per-chunk `.done` sentinel).

## Stage-1 MLM — Jarvis H200 spot migration

The biotite `gpu_h200` queue was not moving, so Stage-1 training moved to a
**Jarvislabs H200 spot instance** ($1.99/hr, region `india-noida-01`) with a
**persistent 150 GB filesystem** (`fs_id=2829`) so spot preemption never loses
staged data or checkpoints. Training runs (`scripts/jarvis/run_mlm.sh`) off the
filesystem: repo clone + 400k/20k subsample parquet + gzipped mature FASTA +
ESM-2 3B weights (HF cache) all live on `/home/jl_fs/`.

Operational notes for future Jarvis work:
- **Spot vs on-demand:** the SDK books spot with `is_spot=True`. A paused
  instance **resumes as on-demand** — to keep spot pricing after any stop, destroy
  and re-`create` rather than resume. Verify via `jl list` (`Type` column) and the
  cumulative `cost` slope (~$0.03/min at spot vs ~$0.07/min on-demand).
- **SSH:** register a public key account-wide (`jl ssh-key add`) *before* creating
  the instance; a key added afterward only injects on a fresh create.
- **Environment:** the `pytorch` template ships torch 2.11+cu130 (H200 driver);
  keeping it and pinning only `transformers==5.13.0 peft==0.19.1 pyarrow` (the libs
  the code depends on) is more robust than forcing the older cu124 torch. sklearn
  1.9 needs Python ≥3.11 (template is 3.10) — a Stage-2 dep only, deferred.
- **Spot safety:** `train_mlm(ckpt_every=200, resume=True)` writes a LoRA-sized
  step-checkpoint to the filesystem; a preemption costs ≤200 steps.

Measured throughput: **~1.94 step/s (31 seq/s) at batch 16**, ~25k steps/epoch →
**~3.6 h/epoch, ~10.7 h for 3 epochs (~$21 spot)**.

### Stage-1 result (COMPLETE)

Coupling-aware `both`-mode MLM ran 3 epochs on the 420k subsample with the
precomputed contact cache. **Val pseudo-perplexity trace: 6.237 → 6.217 → 6.2168**
(epochs 0/1/2, at steps 26,347 / 52,695 / 79,043). The adaptation is essentially
captured in the first pass (ESM-2 unadapted ≈10–15 PPL → 6.24); epochs 2–3 refine
at the margin and the curve plateaus with no val uptick (no overfitting on 400k).
The ~6.2 floor sits higher than a plain i.i.d.-masked run would, consistent with
coupling-aware masking making reconstruction genuinely harder by design — the true
test of *learned* coupling is the post-training probe (mask one partner of a known
salt-bridge/disulfide pair, check the other co-varies), not this number.

`mlm_adapter_best` (best-by-val-PPL, 94.4 MB rank-32 LoRA) transferred to biotite
`$PERSIST/models/mlm_adapt/`. Jarvis instance destroyed — **Stage-1 total $42.32**
spot (precompute + 3 epochs + the ~$8 restart-idle incident), well under budget.

## Stage-2 — per-phenotype classifiers (coupling-aware, biotite)

Five per-phenotype classifier heads (`scripts/slurm/08b_train_classifiers.sbatch`,
job 1152290, array 0-4%2 on `gpu_h200 gpu:1`), each branching from the SAME
coupling-aware Stage-1 adapter — the coupling-adapted representations propagate;
the classifier is discriminative (no masking), so adapter inheritance IS the
coupling-aware path.

**Critical bug caught + fixed before launch:** the MLM adapter trains on
`EsmForMaskedLM` (keys `base_model.model.esm.encoder…`) while the classifier is a
bare `EsmModel` (keys `…encoder…`, no `esm.` prefix). `peft.load_adapter` returns
OK but **silently drops** the mismatched keys — every classifier would have trained
from a *random* adapter, inheriting zero coupling-awareness. Proven by tweaking a
LoRA weight to +1.234 and reading back 0.177. Fixed with
`load_mlm_adapter_into_classifier` (remaps keys, **raises if 0 transfer**). Verified
end-to-end on the real 3B: **288 LoRA tensors transferred, 0 unmatched**,
max|lora_A|=0.183 (real trained values, not random).

Classifier config: 5 epochs, lr_head 1e-3 / lr_adapter 1e-5, margin-loss on matched
pairs (λ=1.0, margin=1.0), best-by-val-AUPRC, natural class sizes (per-phenotype
independent models). Data staged to `$PERSIST`: `labeled_dataset_r232_clustered.parquet`
(1,985,508 rows), `..._protein_pairs.tsv` (90,984 pairs), FASTA already present.

### Matched protein-pair funnel (per phenotype)

The margin loss (and the taxonomy-controlled pair eval below) consumes
**protein-level** ortholog pairs, derived as **cluster ∩ matched-genome-pair**
(`_derive_protein_pairs`, `src/eptrans/dataset.py`): for each matched
(extremophile, mesophile-outgroup) genome pair, a sequence cluster containing a
protein from *both* genomes yields one ortholog pair (one protein per genome,
highest `cs_prob` as a stable tie-break). Measured from
`labeled_dataset_r232_clustered_protein_pairs.tsv` (90,984 pairs total):

| phenotype | protein pairs | genome pairs | ext genomes | protein pairs / genome pair |
|---|---:|---:|---:|---:|
| halophile | 63,846 | 1,472 | 1,472 | 43.4 |
| thermophile | 15,604 | 560 | 560 | 27.9 |
| alkaliphile | 6,314 | 292 | 292 | 21.6 |
| acidophile | 5,051 | 341 | 341 | 14.8 |
| **hyperthermophile** | **169** | **22** | **22** | **7.7** |

**Why hyperthermophile is the outlier (169 pairs, ~125 train / ~44 val), not a
low count across the board.** The collapse is the product of two independent
funnels, and hyperthermophile is small on *both*:

1. **Few matched genome pairs.** Only **22** hyperthermophile genomes survive
   into matched pairs — vs 1,472 for halophile. Hyperthermophiles are a small,
   taxonomically narrow slice of GTDB (mostly *Pyrococcus, Thermococcus,
   Methanocaldococcus, Thermotoga, Aquifex*), so there are simply few
   extremophile genomes to match, regardless of outgroup availability. Having
   "roughly as many taxonomy-matched outgroups as extremophiles" (true here:
   ext genomes == out genomes each row, so pairing is 1:1) does **not** raise
   this — the binding constraint is the *extremophile* side, not the outgroup
   side.
2. **Lowest per-pair ortholog yield.** Each hyperthermophile genome pair yields
   only **7.7** protein pairs vs halophile's 43.4. This is the cluster-intersection
   step: two genomes contribute a protein pair only for clusters they *both* land
   in. Hyperthermophile proteomes are small (compact thermophile genomes) *and*
   their proteins cluster tightly within-clade but poorly with the mesophile
   outgroup at the mmseqs identity threshold, so fewer shared clusters ⇒ fewer
   orthologs per genome pair.

Net: 22 genome pairs × 7.7 orthologs ≈ 169. The pair signal is therefore
well-powered for halophile/thermophile/acidophile/alkaliphile and **weak only for
the hyperthermophile POC** — which is exactly the phenotype where we're leaning on
it least (the POC's purpose is the pipeline, not a defensible pair-AUC).

### Pair-aware eval + per-epoch snapshots (added mid-run)

Added `evaluate_pair_metrics` (`train.py`): a **taxonomy-controlled** eval on
held-out matched pairs — `pair_acc` = fraction with `s_ext > s_out` (0.5 = the
head is riding taxonomy; >0.5 = genuine phenotype signal that pointwise val AUPRC
*cannot* isolate, since val singles still carry organism-level taxonomic signal),
plus paired `pair_auc` and mean `margin_gap`. Wired into `train_classifier` via
`val_pair_ds` → logs `val_pair_acc`/`val_pair_auc` per epoch. Standalone
`scripts/08d_eval_pairs.py` computes the same metrics post-hoc from a
`clf_epoch<E>/clf_matched.pt` snapshot, so an already-running job (the
hyperthermophile POC 1152524, which predates the wiring) still gets a pair-AUC.
Also added per-epoch **matched (adapter, head) snapshots** (`clf_epoch<E>/`,
never clobbered) so an early epoch stays independently loadable for the
generation step, plus step-checkpointing (`clf_ckpt.pt`) + flushed progress
logging + mid-epoch resume.

### Frozen-backbone cached-embedding probe (design #1) — RESULTS

The dominant Stage-2 cost is re-encoding millions of proteins through the 3B
backbone every step/epoch (end-to-end A5000 bs6 projected **~27 d/epoch**
thermophile, **~26 d/epoch** halophile — cell-102 projection table). Since the
coupling-aware MLM adapter already separates phenotype ~linearly (train loss
collapses in ~60 steps), we embed the whole secretome **once** through the frozen
MLM-adapted backbone and train heads on the cache.

**Pipeline** (`scripts/09_embed_secretome.py` + `scripts/10_train_cached_probe.py`,
commit chain 501ab54 → 918d3dc → f4a0794):
- **Stage A — embed once** (job 1152597, `09_embed_secretome.sbatch`, gpu-partition
  array 0–7): bare `EsmModel` + Stage-1 MLM LoRA adapter (`set_adapter('mlm')`),
  eval/no_grad, **masked mean-pool matching `MeanPoolClassifierHead`**, fp16 vectors
  + `tagged_id` per shard. 8 shards × 248,189 rows × 2560 dim (~1.27 GB/shard);
  `.done_shard{i}` idempotency guard. ~3 h wall under A5000 contention (~29 seq/s
  per shard, 3-way node sharing). Spot-check: 100% finite, all rows non-zero,
  ids==rows, mean ~0.002 / std ~0.29.
- **Stage B — train heads** (job 1152605, `10_train_cached_probe.sbatch`,
  `afterok:1152597`): all 5 heads on cached vectors, **ALL negatives** (no
  neg_per_pos), 30 epochs, **~4m45s total**. Loss = weighted BCE + **ACTIVE
  matched-pair margin** (pair-aware / "alternative": train-split pairs scored each
  step, `--pair-batch-size 256`, cycled) — the anti-taxonomy mechanism on frozen
  features. Best-AUPRC epoch → `clf_<pheno>_cached/head_best.pt`.

**Results (best epoch, pair-aware):**

| phenotype | val AUPRC | pair-AUC | pair-acc | base rate (val pos frac) | best epoch | val pairs |
|---|---:|---:|---:|---:|---:|---:|
| **hyperthermophile** | **0.898** | **0.924** | 0.957 | 0.0133 | 25 | 23 |
| **thermophile** | **0.862** | **0.905** | 0.937 | 0.2146 | 27 | 1,495 |
| **halophile** | 0.818 | 0.748 | 0.798 | 0.4246 | 27 | 6,616 |
| acidophile | 0.745 | 0.749 | 0.837 | 0.0873 | 24 | 578 |
| alkaliphile | 0.674 | 0.768 | 0.800 | 0.0738 | 28 | 666 |

**Reads:**
- **Signal is genuine, not taxonomy.** Every pair-AUC ≫ 0.5 (held-out
  taxonomy-matched orthologs). Thermal phenotypes strong (hyper 0.924, thermo
  0.905); salt/pH weaker (0.75–0.77) but clearly above chance — consistent with
  thermoadaptation having the most distinctive proteome signature. The
  matched-outgroup design works as intended.
- **Frozen-feature heads plateau by ~epoch 10** (hyperthermophile: AUPRC 0.83→0.89
  by ep10, then ±0.01 noise for 20 more epochs). Best epochs at 24–28 are
  noise-band peaks, not late gains → **more epochs won't help; the ceiling is the
  embedding, not training time.** To push weaker phenotypes: richer pooled feature
  or end-to-end reshape, not more epochs.
- **Epoch moving forward for generation: `head_best.pt` (hyperthermophile epoch 25,
  AUPRC 0.898 / pair-AUC 0.924)** + the frozen MLM-adapted backbone = coherent
  `(adapter, head)` pair (adapter fixed by construction).

Artifacts: `cached_probe_results.png` (epoch trajectory + per-phenotype summary),
`cached_probe_results.csv`. Heads on biotite at
`models/cached_probes/clf_<pheno>_cached/head_best.pt`.

## Coupling-aware masking — thresholds and the contact-pair cache

The masked-LM objective masks positions to reconstruct; **coupling-aware masking**
(design §15 #1) additionally masks *coupled* positions jointly so multi-residue
features (salt bridges, disulfides, local secondary structure) are learned as
units rather than reconstructed by copying a visible partner. Thresholds
(`config/config.yaml → modeling.mlm`, all now wired through `08_train_backbone.py`):

| Parameter | Value | Meaning |
|---|---|---|
| `mask_rate` | 0.15 | fraction of positions masked (BERT 80/10/10 corruption) |
| `gamma` | 1.0 | conservation exponent in `P(mask) ∝ (1−c)^γ`; **uniform until MSA conservation is wired** (single-sequence training → `c=0`) |
| `coupling_mode` | `both` | mask span **and** contact-pair units jointly |
| `span_len` | 3 | contiguous block length (local secondary structure) |
| `contact_threshold` | 0.5 | ESM-2 contact-head probability to call a coupled pair |
| `contact_min_sep` | 6 | min residue separation (skip trivial i,i+1; those are span mode) |
| `top_k` | 128 | max contact pairs cached per sequence (bounds storage) |
| **immutable/frozen** | boolean | active-site/ligand residues, `γ→∞` limit — never masked; applied at *generation* time, not Stage-1 pretraining |

**Cost fix (`scripts/10_precompute_contacts.py`):** contact-pair derivation was
coded to run a 3B contact-head forward pass *per item per epoch* inside the
DataLoader — infeasible at 420k×3. Contacts are now **precomputed once on GPU**
(keyed by `tagged_id`, residue coords, top-128) into `contact_pairs.parquet`;
`build_mlm_dataset(contact_pairs_col=…)` consumes them for free, remapping
full-sequence coords into each sliding window. Two required fixes surfaced on the
real 3B model: `predict_contacts` needs `attention_mask`, and the contact head
needs `attn_implementation="eager"` (the default SDPA backend returns no
attentions → empty stack). First launch used baseline i.i.d. masking (flag
omitted); relaunched with `--coupling-mode both --contact-pairs …`.

## §16 — Masked generation engine (design)

Settled pipeline for PLM-driven masked generation (the fine-tuned MLM as a
*proposer*, complementing MPNN):

1. **Freeze the immutable set** — catalytic (M-CSA) + ligand-contacting +
   (once available) high-conservation columns. Never masked.
2. **Select mutable positions** — the aggressiveness axis: surface-only →
   +second-shell → all-non-immutable, conservation-gated.
3. **Gibbs + contact-pair sampling** (committed, replaces single-pass): iterative
   masked-predict-remask, decoding contact-paired positions *jointly* (same ESM-2
   contact head as training), so coupled substitutions co-emerge instead of being
   filled independently.
4. **Score** — per-phenotype classifier gives the *directional* signal (is it
   moving toward the target phenotype).
5. **Structural gate (MPNN)** — fold-free MPNN score against the wild-type
   backbone screens structurally implausible proposals cheaply; survivors are
   refolded (ESMFold/AF2) and gated on **catalytic-atom RMSD** vs wild-type.

Division of labor: **PLM proposes → classifier scores → MPNN gates**, with the
active site protected preventively (frozen, never mutated) *and* verificationally
(catalytic-RMSD gate).

**Deferred / to test after the adapter lands:**
- **Contrastive steering** (delta-logit `logit_adapted − logit_base`): flagged as
  an *optional, validate-first* prior. Stage-1 pooled all 5 phenotypes into one
  bucket, so the bulk delta-logit is muddy for **pH** specifically — acidophile and
  alkaliphile signatures point opposite ways and cancel. Temp/salinity are more
  directionally coherent (no cold bucket, monotonic halophile surface). Plan:
  measure per-phenotype delta-logit distributions on the trained model — confirm
  acido/alkali anti-correlate before trusting any contrastive term. Default steering
  comes from the per-phenotype classifier, not the bulk MLM.
- **Standalone vs MPNN-coupled generation** (open): whether masked-gen is a
  co-equal proposer or the fold-free fallback for structure-poor inputs. User
  deciding; tightly-coupled PLM→classifier→MPNN is the high-quality path,
  MLM-standalone the degraded-input path.
- **Contact-precompute batching** — current precompute is single-sequence
  (~25 seq/s, GPU ~47%); batched inference would cut the ~4.5 h one-time pass.

## Generation-stage tools provisioned on biotite (2026-07-13)

The generation pipeline's structural gate (Stage 6) and folding oracle are now
installed and validated on biotite. Both were built on the **login node**
(igi.biotite, which has internet) and validated on **GPU nodes** (which do NOT —
conda/pip/HF-download all fail there with NameResolutionError; the split is the
standard no-egress-compute-node pattern).

**LigandMPNN** (`conda env: ligandmpnn`, repo `eptrans_scratch/software/LigandMPNN`)
— one install ships **all MPNN variants**: ProteinMPNN, LigandMPNN, SolubleMPNN,
membrane. So the auto-select design (LigandMPNN for holo / ProteinMPNN for apo)
needs no second install. Validated on bundled `inputs/1BC8.pdb` (a 2-Zn protein):
`protein_mpnn` and `ligand_mpnn` both exit 0; LigandMPNN correctly detected the
2 Zn ions + 41 ligand-context residues and produced a higher-confidence design
(ligand_confidence 0.555 vs ProteinMPNN 0.397; seq_rec 0.52 vs 0.49) — the
metalloenzyme case the gate is built for. Gotchas: numpy pinned 1.26.4 (2.x
breaks); vendored openfold used deprecated `np.int`/`np.float` — sed-patched to
builtins in `openfold/{np/residue_constants.py,np/relax/utils.py,data/templates.py}`
(re-apply if repo re-cloned); `prody` is a required dep missing from the repo's
install docs.

**ESMFold** (`conda env: esmfold`) — HF `transformers` `EsmForProteinFolding`
path (not the fair-esm/openfold build), weights `facebook/esmfold_v1` in the
shared HF cache (`HF_HOME=.../IS1111/eptrans/hf_cache`, 27 GB, co-located with the
3B ESM-2). Validated: folded a 78-res test sequence in 1.9 s on an RTX A5000,
mean pLDDT 0.80, PDB written (`esmfold_validation_1BC8test.pdb` artifact). Run
with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` on GPU nodes. `model.esm.half()`
puts the ESM trunk in fp16 (standard ESMFold inference); the "contact_head newly
initialized" warning is benign (folding path unaffected).

**Databases** (verified present, from `config/config.yaml`): Foldseek PDB
`/shared/db/foldseek/latest/db/pdb`, Foldseek AlphaFold
`/shared/db/foldseek/latest/db/alphafold_uniprot`, UniRef30
`/shared/db/uniclust/30_2020_06`, ColabFold envDB
`/shared/db/colabfold/latest/colabfold_envdb_202108_db`, UniProt mmseqs
`/shared/db/uniprot/latest/mmseqs`. `mmseqs` + `foldseek` on PATH
(`/shared/software/bin`). Foldseek is the Stage-3 recall booster — transfers
M-CSA/Swiss-Prot catalytic-residue annotations from PDB/AlphaFold structural
homologs when no direct sequence match exists.

Pipeline tool status: Stages 1–2 (mmseqs/conservation) ✅, Stage 3 (foldseek +
annotation) ✅, Stages 4–5 (ESM-2 3B + cached heads) ✅ already, Stage 6a
(LigandMPNN/ProteinMPNN) ✅ now, Stage 6b (ESMFold) ✅ now. All generation-stage
tools are installed; what remains is the runtime driver in `SlurmBackend.submit`.

## Stage 1–3 fixes and full end-to-end laccase run (07-13)

Three fixes landed after the first end-to-end laccase run (which used a
uniform-conservation fallback because the MSA DB was misconfigured):

**1. MSA database + memory (commit a1a1861).** The first run's conservation was
uniform because `/shared/db/uniclust/30_2020_06` is an HH-suite DB (hhblits), not
mmseqs, and the fallback (`uniprot_kb`, 90 GB) OOM'd a single-query `easy-search`.
Fixed: target the clustered **UniRef50** FASTA (`/shared/db/uniref/uniref50/latest`,
~24 GB, non-redundant), add `--split-memory-limit` (env `MMSEQS_MEM_LIMIT`, default
80G) as a hard footprint cap. UniRef50's 50%-identity clustering also gives
first-order de-redundancy against taxonomic over-sampling.

**2. Henikoff conservation weighting (commit 32553f8).** Replaced a crude
percent-identity discount with true Henikoff & Henikoff (1994) position-based
sequence weights: per column c, a sequence with residue r gets weight
`1/(k_c · n_{c,r})` (k_c = distinct residues in the column, n_{c,r} = sequences
sharing r), averaged over covered columns and renormalised to mean 1. This corrects
phylogenetic over-sampling that raw counts miss. Conservation is the weighted
modal-residue fraction per column. Unit-tested: 10 near-identical + 2 divergent
sequences → over-sampled group weight 0.4 each, divergent 4.0 each (10× up-weight).

**3. Swiss-Prot annotation channel + adaptive Otsu freeze (commits dd8a7c7, 66f36d9,
0fd29f8).** The foldseek-vs-AlphaFold structural channel returned 0 transferable
annotations (verified: 0/300 AF hits present in the reviewed Swiss-Prot or M-CSA
tables — AF DB is dominated by unreviewed TrEMBL entries). Root cause: annotated
laccase homologs are reachable by *sequence* homology, not among the top structural
AF neighbours. Fix: a **reviewed Swiss-Prot mmseqs DB** (575,503 seqs, built once on
the login node) is now the PRIMARY channel — `mmseqs` WT-seq search → transfer
ACT_SITE/BINDING/SITE (+ M-CSA catalytic) in UniProt coordinate space via the
alignment. foldseek-vs-AF is off by default (`GEN_USE_FOLDSEEK=1` to enable as a
recall booster). The raw union over-transfers (213/519 = 41%, extended BINDING
ranges), so an **adaptive Otsu (1979) threshold** on the per-position homolog vote
distribution finds the natural break between the incidental-range tail and the
recurrent-catalytic cluster — parameter-free, adapts per enzyme to hit count and
annotation depth; hard floor of 2 votes.

### Full pipeline run (H200, job 82374b61, all fixes)

| stage | result |
|---|---|
| Swiss-Prot transfer | 145 homologs → 213 raw positions → **Otsu kept 60** (threshold 6 votes, max support 22) |
| Cu-ligand retention | **10 of 11** kept by transfer (His84/86/129/131, His415/418/420, His472/Cys473/His474); His422 fell 1 vote below the Otsu cut but is re-caught by the Cu-motif backstop |
| active site (Stage 4) | 80 residues = 60 transferred + 19 conservation + 1 motif |
| conservation | Henikoff-weighted, mean 0.432, 2000 MSA hits |

Design results (best per phenotype, WT classifier scores thermophile 0.017 /
halophile 0.009):

| phenotype | best design | clf score | # muts | active-site CA-RMSD |
|---|---|---|---|---|
| thermophile | ther_2 | 0.911 | 146 | 1.36 Å |
| thermophile | ther_1 | 0.366 | 87 | 1.09 Å |
| halophile | halo_3 | 0.744 | 188 | **14.9 Å** ⚠ |
| halophile | halo_2 | 0.467 | 162 | **17.2 Å** ⚠ |
| halophile | halo_1 | 0.326 | 116 | 1.90 Å |

**Key observations.**
- Real Henikoff conservation lifted thermophile from a ~0.3 ceiling (uniform-fallback
  run) to 0.91 — masking now targets variable positions and spares the constrained
  active-site/conserved core. This confirms the earlier ruling that thermophiles
  needed *real conservation*, not more Gibbs cycles.
- **The active-site RMSD gate is flagging classifier gaming on the halophile head:**
  the two top-scoring halophile designs (0.744, 0.467) have catastrophic active-site
  RMSDs (14.9, 17.2 Å) from ~160–190 mutations — the fold collapses around the frozen
  residues, so the high scores are not physically trustworthy. Only the low-scoring
  halo_1 (0.326) holds a coherent site (1.9 Å). Argues for a tighter mutation budget
  or promoting RMSD from a reported metric to a hard filter.

Artifacts: `laccase_gen_full_results.json`, `laccase_active_site_transfer.json`,
`laccase_ther2_full.pdb` (clf 0.911, RMSD 1.36 Å), `laccase_halo3_full.pdb`
(clf 0.744 but RMSD 14.9 Å — gaming example). Annotation tables archived as
`annotation_tables.tar.gz`.

## Web portal: real demo output + Cloud Run readiness (07-13)

Wired the public portal to showcase the real generation run instead of synthetic
placeholders (commits db2110e, 42d95db):

- **Example enzyme** = the 519-aa secreted laccase (multicopper oxidase) behind the
  demo run, so "Load example → Generate" reproduces the bundled results.
- **`DemoBackend` now serves the actual results from Biotite job 82374b61** (Swiss-Prot
  transfer + Otsu active-site freeze + Henikoff-weighted conservation) from a checked-in
  fixture (`webapp/fixtures/laccase_demo/` = real `results.json` + 7 ESMFold structures),
  filtered to the requested phenotypes. `EPT_DEMO_SYNTHETIC=1` restores the old
  randomized fixture. Verified end-to-end in a fresh Flask env: served scores match the
  artifact byte-for-byte (ther_2 0.9105, halo_3 0.7437), all structures render.

Deploy path (`webapp/deploy_cloudrun.sh`) is unchanged and ready: source deploy of the
thin gunicorn frontend, GCS bucket FUSE-mounted for the SQLite DB + result files, IAM
grants for Cloud Build. The public deployment runs the **demo backend** (no SSH keys,
no cluster access) — the real Slurm generation stays campus-side, so the portal shows
the laccase results with no credential exposure. Deploy is a user-run step (needs
authenticated `gcloud` pointed at the target GCP project); everything it ships is on
`origin/main`.

## 2026-08-04 — Retrain staging: scope decision, pair audit, class balance

Retrain of the adapter and classifiers from scratch on the merged (GTDB + deep-sea)
dataset. Phase 0 of the approved plan. Cluster stages are blocked on login-node
instability; everything below is local and measured.

### Merge (committed f1ca5fc)
`results/combined_labels_r232_plus_deepsea.parquet` = 204,007 rows
= 199,923 GTDB r232 species reps + 4,084 deep-sea MAGs. Idempotent re-run.

MAG confidence tiers earned under the standard rubric:

| class | high | medium | low |
|---|--:|--:|--:|
| thermophile | 590 | 272 | 1,191 |
| hyperthermophile | 0 | 40 | 242 |
| halophile | 0 | 81 | 0 |
| acidophile | 0 | 34 | 0 |
| alkaliphile | 0 | 5 | 0 |
| psychrophile | 2 | 3 | 1,376 |

Corrects an earlier claim in-session that the MAGs had no route above `low`:
thermophile reaches `high` for 590, because vent metadata and hot GenomeSPOT
predictions agree. The psychrophile row is the honest non-contribution — 1,417
cold-flagged MAGs, 1,376 stuck at `low`.

In-situ temperature was TESTED as a promotion route for cold and REJECTED, using
the hot end as control on the same tag: ambient >=50 C (n=12) agrees with
prediction 11/12 (92%); ambient <=15 C (n=201) agrees 2/201 (1.0%), mean
over-prediction +23.5 C. Ambient corroborates where hot, is uninformative where
cold.

### Per-sample cap (committed f1ca5fc)
MAG labels are SAMPLE-level: 4,084 MAGs from 858 samples, worst sample 38
thermophile MAGs across 30 distinct families — passes a per-family cap while
resting on ONE environmental observation. Added `max_per_sample` to
`select_extremophiles` / `select_with_outgroups`, config value 5.

Measured on the merged table, thermophile:

| cap | MAG extremophiles | samples | worst sample | pairs |
|---|--:|--:|--:|--:|
| None | 423 | 112 | 21 | 4,994 |
| 5 | 302 | 117 | 5 | 4,927 |
| 2 | 192 | 120 | 2 | 4,868 |

cap=5 cuts worst-case replication 4x for 1.3% of total pairs and RAISES the
sample count (freed slots go to other samples). Blank `source_sample_id` means
"isolate genome, its own sample" and is EXEMPT — pooling blanks would cap all of
GTDB. Verified an exact no-op on the GTDB-only baseline.

### Protein scope — DECIDED (committed 9028ef3)
    secreted:       thermophile, halophile, acidophile, alkaliphile
    whole_proteome: hyperthermophile, psychrophile

Thermophile stays secreted. The large-cluster concern that held it open WAS
resolved (measured prevalence spectrum: f_max = 108/158 = 0.684, ZERO clusters
above f=0.70 over 158 genomes / 224,022 clusters; plus the class-stratified
`max_pairs_per_cluster_class`), but the user chose to keep it secreted anyway:
whole-proteome scope would add ~1M pairs and make thermophile ~90% of the
dataset. Recorded in config as a decision with its evidence, not an open question.

### CORRECTION: emitted vs usable genome pairs (committed 9028ef3)
Every genome-pair count quoted earlier this session was rows EMITTED by
`select_with_outgroups`, which include extremophiles where `find_outgroup`
returned None at every rank. Those have no mesophile contrast and are not pairs.

| class | emitted | usable | no outgroup | % lost |
|---|--:|--:|--:|--:|
| thermophile | 4,927 | 3,413 | 1,514 | 30.7 |
| halophile | 2,935 | 2,459 | 476 | 16.2 |
| acidophile | 799 | 766 | 33 | 4.1 |
| alkaliphile | 581 | 565 | 16 | 2.8 |
| hyperthermophile | 291 | 234 | 57 | 19.6 |
| psychrophile | 214 | 214 | 0 | 0.0 |
| **TOTAL** | **9,747** | **7,651** | **2,096** | **21.5** |

Thermophile loses most: most extremophiles competing for a finite mesophile pool
under used-once matching. Psychrophile loses none, consistent with being 87.4%
genus-matched. The psychrophile:thermophile gap is 16x on usable pairs, not 23x.

### SignalP requirement
13,946 distinct genomes = 8,799 extremophiles + 5,147 outgroups
(13,548 GTDB + 398 MAG-derived). `reuse_outgroups=True` saves **1,567 genomes**
(6,714 -> 5,147 outgroups) at IDENTICAL pair counts, so reuse is free here rather
than a tradeoff. Thermophile is 59.8% of the bill — the direct cost of keeping it
secreted. Accession list: results/ (signalp_required_accessions.tsv artifact).

### Class balance — the imbalance runs OPPOSITE to expectation
Question asked was whether to downsample thermophile and halophile. Under the
committed scope they are NOT the over-represented classes.

| phenotype | scope | usable gp | protein pairs | share |
|---|---|--:|--:|--:|
| psychrophile | whole | 214 | 175k-350k | 53-69% |
| halophile | secreted | 2,459 | 83,286 | 16-25% |
| thermophile | secreted | 3,413 | 58,840 | 12-18% |
| alkaliphile | secreted | 565 | 7,689 | 1.5-2.3% |
| acidophile | secreted | 766 | 6,886 | 1.4-2.1% |
| hyperthermophile | whole | 234 | 1,275 | 0.3-0.4% |

Total 333k-508k protein pairs. Two decisions compound for psychrophile: whole
proteomes (~9x more protein per genome) AND the tightest rank matching (1.16), so
nearly every genome pair yields orthologs. Downsampling thermophile/halophile
would make imbalance WORSE.

Psychrophile yield is UNMEASURED. A rank-distance log-linear fit over the three
measured classes (r = -0.97) extrapolated to rank distance 1.16 gave 6,152
pairs/genome-pair — **6x more than a 2,045-protein proteome can contain**. The fit
extrapolated outside its range [2.68, 4.46] on n=3 points; DISCARDED. Replaced
with an ortholog-sharing bound (40-80% x 2,045), anchored on halophile's measured
26% at a looser rank distance. The ~90 s cluster probe on the 214 pairs would
replace this with a real number and should be run before training.

### Decision: loss weighting, not downsampling
Inverse-frequency loss weighting for psychrophile lands at 0.45-0.89 across the
whole 53-69% uncertainty band — under 2x swing — so the classifier decision is
ROBUST to the unmeasured yield and no downsampling is needed. Precedent:
OGTFinder (2025) chose weighted RMSE over under/oversampling at 0.9% (n=58)
psychrophiles of 6,401.

The real risk is not thermophile over-representation but psychrophile genome
REDUNDANCY in the MLM: 214 genome pairs supplying most tokens means the adapter
sees ~400 genomes repeatedly. Admitting low-tier psychrophiles mitigates this 6x.
Post-training check: per-class AUPRC on a balanced validation slice.

### Rubric-rank sample weights
| tier | weight | MLM | classifier |
|---|--:|:--:|:--:|
| high | 1.00 | yes | yes |
| medium | 0.50 | yes | yes |
| low | 0.10 | yes | **no** |

`low` = 0.10 is anchored on the measured 0.14-0.63% precision of habitat-only
cold evidence across four populations (n=1452/639/443/1416). A calibrated weight
would be ~0.005, indistinguishable from exclusion; 0.10 keeps the sequence
visible to the MLM for embedding structure without letting it drive the loss. It
is a deliberate floor, not a likelihood ratio.

Psychrophile genome counts by rule: MLM (h+m+l) 2,306; classifiers (h+m) 387.
Admitting `low` buys the MLM 1,919 extra genomes, a 6x increase — the specific
thing that decision purchases.

### Blocked
SignalP coverage diff, staging, protein collection, clustering, splits, and the
combined gpu_h200 job all require biotite. The login node wedged on every SSH
call for several hours (5+ force-cleared executions, two of them delegation
dispatches that spawned zero children after 20+ min). Resume from the coverage
diff; the accession list is already computed.

### CORRECTION (same day): two errors in the class-balance analysis above

**1. Hyperthermophile pair count was 7.9x too low.** I reported 1,275 pairs from
234 genome pairs (5.4 pairs/gp). The user flagged it as implausible and was right.
I computed it by CHAINING the end-to-end secreted rate (0.97) with the probe's
whole/secreted ratio (5.62) = 5.5 pairs/gp. That chaining is invalid: 0.97 is an
end-to-end PRODUCTION rate that already absorbed cluster/cap/split losses, and
5.62 is a ratio measured INSIDE the probe, so multiplying double-counts the
losses. The probe measured whole-proteome pairs DIRECTLY:

| class | probe pairs | probe gp | pairs/gp DIRECT | chained (wrong) |
|---|--:|--:|--:|--:|
| halophile | 9,184 | 17 | 540.2 | 422.0 |
| alkaliphile | 5,725 | 13 | 440.4 | 277.2 |
| acidophile | 5,009 | 13 | 385.3 | 233.8 |
| thermophile | 5,014 | 15 | 334.3 | 206.9 |
| hyperthermophile | 648 | 15 | **43.2** | **5.5** |

Hyperthermophile has the smallest secreted rate, so the distortion was largest
there — which is why it surfaced in that class and not the others. Corrected
hyperthermophile: 234 gp x 43.2 = **10,108 pairs**, not 1,275. At ~2,045 proteins
per proteome that is 2.1% of proteins forming usable ortholog pairs — low but
plausible given 52.7% of its pairs are phylum-level matches.

I had ALREADY identified this exact inconsistency earlier in the session ("the
probe measured 43.2 directly but chaining gives 5.5") and then used the chained
value anyway.

**2. I measured the wrong quantity for imbalance.** Read from
scripts/08_train_backbone.py and src/eptrans/modeling/data.py:

- MLM: consumes `--labeled` + `--fasta` and has NO `--pairs` argument.
  `build_mlm_dataset` iterates PROTEINS; each item is one masked sequence.
- classifier: `--labeled` drives weighted BCE over individual PROTEINS;
  `--pairs` adds an AUXILIARY margin term max(0, d - (s_ext - s_out)).
  `neg_per_pos=3.0` caps negatives at 3x positives.

So PROTEINS are the training units. Protein pairs serve two narrower roles: the
auxiliary margin loss, and the leakage-control grouping for splits. Sizing the
training set or judging class imbalance on pairs was the wrong denominator.

Corrected composition (proteins = usable_gp x 2 genomes x 2,045 proteins,
x 11.28% for secreted scope):

| phenotype | scope | usable gp | proteins | protein share | pairs |
|---|---|--:|--:|--:|--:|
| thermophile | secreted | 3,413 | 1,574,594 | 30.5% | 58,840 |
| halophile | secreted | 2,459 | 1,134,464 | 22.0% | 83,286 |
| hyperthermophile | whole | 234 | 957,060 | 18.6% | 10,108 |
| psychrophile | whole | 214 | 875,260 | 17.0% | 131k-350k |
| acidophile | secreted | 766 | 353,395 | 6.9% | 6,886 |
| alkaliphile | secreted | 565 | 260,663 | 5.1% | 7,689 |

**TOTAL 5,155,436 proteins.** This REVERSES the "imbalance runs opposite to
expectation" claim above: on proteins, thermophile IS the largest class at 30.5%,
as the user originally assumed. The max/min protein ratio is 6.0x, versus the
19.1x I computed on pairs.

The downsampling recommendation is UNCHANGED but its reasoning is replaced: not
"thermophile isn't the biggest" (it is), but "the spread is only 6x, which
inverse-frequency weighting handles comfortably". User has confirmed
inverse-frequency loss weighting as the approach.

Psychrophile remains the only unmeasured class (131k-350k pairs; its protein count
875,260 is firm since it derives from genome count, not pair yield). The ~90 s
cluster probe on its 214 genome pairs would settle the pair range.

## Final dataset composition — locked tiers (2026-08-04)

Selection parameters, all explicit (the `max_total_per_class` default of 100 in
`04_select_genomes.py` is a **pilot** value and is deliberately NOT used):

```
max_per_lineage      = 5      (rank: family)
max_total_per_class  = none   (uncapped — see note below)
max_per_sample       = 5      (sample_col: source_sample_id, caps deep-sea MAGs per metagenome)
outgroup_match_rank  = genus, falling back to family/order/class
reuse_outgroups      = True   (a mesophile may serve several phenotypes)
seed                 = 1466
```

### Per-phenotype inclusion criteria

| phenotype | protein scope | confidence tier | evidence route |
|---|---|---|---|
| halophile | secreted | high + medium | keyword + GenomeSPOT prediction |
| hyperthermophile | whole proteome | high + medium | keyword + GenomeSPOT prediction |
| psychrophile | whole proteome | high + medium | **measured OGT; GenomeSPOT excluded** |
| thermophile | secreted | **high only** | keyword + GenomeSPOT prediction |
| acidophile | secreted | high + medium | keyword + GenomeSPOT prediction |
| alkaliphile | secreted | high + medium | keyword + GenomeSPOT prediction |

**Two distinct rubrics are in play.** Quoted from source, not paraphrased:

*GenomeSPOT rubric* (`eptrans.binning.combine_label`, all classes except psychrophile):

| tier | rule |
|---|---|
| high | metadata keyword AND prediction agree on ≥1 class |
| medium | prediction only |
| low | metadata only, or metadata/prediction conflict |
| none | no evidence |

*Measured-OGT rubric* (`scripts/03b_merge_measured_ogt.py:classify`, psychrophile
only — docstring opens "GenomeSPOT is never consulted"). Thresholds
`STRICT_C=15.0`, `LENIENT_C=20.0`, `TOLERANT_C=25.0`, `TMIN_FREEZING_C=0.0`;
`CONVENTIONAL_TEMPS={25,28,30,37}` are dropped from all four OGT sources as
laboratory-convention artifacts:

| tier | rule |
|---|---|
| high | OGT ≤15 °C with cold habitat OR ≥2 independent sources; or OGT ≤20 °C with cold habitat or Tmin ≤0 °C |
| medium | OGT ≤15 °C single-source; OGT ≤20 °C uncorroborated; OGT <25 °C with cold habitat |
| low | cold habitat only, no measurement |
| none | OGT ≥25 °C — **overrides** cold-sounding metadata (61 r232 genomes) |

Tmin never admits a genome alone; it only promotes medium→high. OGT pooled from
four sources (TEMPURA, Madin, Toki, OGTFinder-optima).

### Genome pairs and protein counts

| phenotype | scope | tier | emitted | **usable** (drop) | proteins | share | protein pairs |
|---|---|---|--:|--:|--:|--:|--:|
| halophile | secreted | high+med | 2,935 | **2,459** (−476, 16.2%) | 1,134,464 | 27.7% | 83,286 |
| hyperthermophile | whole | high+med | 291 | **234** (−57, 19.6%) | 957,060 | 23.4% | 6,309 |
| psychrophile | whole | high+med | 214 | **214** (−0, 0.0%) | 875,260 | 21.4% | 72,156–125,366 |
| thermophile | secreted | high only | 1,555 | **1,117** (−438, 28.2%) | 515,330 | 12.6% | 19,257 |
| acidophile | secreted | high+med | 799 | **766** (−33, 4.1%) | 353,395 | 8.6% | 6,886 |
| alkaliphile | secreted | high+med | 581 | **565** (−16, 2.8%) | 260,663 | 6.4% | 7,689 |
| **TOTAL** | | | 6,375 | **5,355** (−1,020, 16.0%) | **4,096,172** | | **195,583–248,793** |

**emitted vs usable (the parenthesised drop).** `select_with_outgroups` emits a
row per selected extremophile even when no mesophile outgroup can be found. Only
rows with a non-null `outgroup_acc` form a pair; the parenthetical gives the
count and percentage lost. Verified rather than assumed: all 1,020 dropped rows
have `matched_rank` NULL, i.e. no confident mesophile existed in the same genus,
family, order, class **or phylum** — these are not near-misses, and no loosening
of the match rank recovers them. Raising `max_per_lineage` would not help either;
the constraint is the absence of a mesophile relative, not a per-family quota.

The rate varies 10× by class and the ordering is informative:

| phenotype | drop | why |
|---|--:|---|
| thermophile | 28.2% | worst; whole clades (e.g. Thermotogota, Aquificota) are thermophilic throughout, so no mesophile sister exists at any rank |
| hyperthermophile | 19.6% | same mechanism, more extreme per genome but fewer genomes |
| halophile | 16.2% | largest absolute loss (476) simply from being the largest class |
| acidophile | 4.1% | acidophily is scattered across otherwise-mesophilic lineages |
| alkaliphile | 2.8% | same |
| **psychrophile** | **0.0%** | every one of the 214 matched — the measured-OGT route admits only cultivated species, which sit in well-sampled lineages with mesophilic relatives |

Psychrophile's clean 0% is a side effect of its rubric, not evidence it is
better-behaved: requiring a measured optimum restricts it to cultivated organisms,
and those are exactly the taxa with sequenced mesophilic neighbours.

**Provenance of each column.** Measured: emitted, usable, and every genome count —
from the selection run above. Derived, not measured: proteins = usable × 2 genomes
× 2,045 proteins/genome, × 11.28% for secreted scope (SignalP secreted fraction,
1,985,508/17,603,649 over 7,268 genomes). Protein-pair rates by source —
secreted classes use production end-to-end rates (halophile 33.87, thermophile
17.24, alkaliphile 13.61, acidophile 8.99 pairs/gp); hyperthermophile uses the
prevalence probe's **direct** whole-proteome measurement of 43.2 pairs/gp over 15
genome pairs. Do NOT chain the secreted rate with a whole/secreted ratio — that
double-counts cluster/cap/split losses and understated hyperthermophile 7.9×
(5.5 vs 43.2 pairs/gp).

**Psychrophile pairs are the one unmeasured cell**, bounded 131k–350k by assuming
30–80% ortholog sharing across its 214 genome pairs. Its protein count (875,260)
is firm, since it derives from genome count. A ~90 s mmseqs probe on those pairs
would settle it; the range affects only the classifier's auxiliary margin term,
not the MLM training set.

### What the tier lock changed

Thermophile high-only (vs high+medium) drops it 3,413 → 1,117 usable pairs (−67%)
and from 30.5% → 12.6% of proteins, moving it from largest class to fourth.
Halophile keeps high+medium and becomes largest at 27.7%. Max/min protein spread
is 4.4×, down from 6.0×.

Justification, from the production run's own record: thermophile there was
**already high-only** (1,171 selected, 0 medium) and scored AUPRC 0.862 /
pair-AUC 0.905 — and high-only now yields 1,117 usable pairs vs that run's 560,
i.e. **1.99× more data behind a known-good result**. Halophile in that run was
high+medium (421 high + 1,785 medium) at AUPRC 0.818 / pair-AUC 0.748; there is
**no high-only halophile measurement anywhere**, and high-only now would give 498
pairs vs the 1,472 that produced 0.818 (0.34×) with family breadth collapsing
1,114 → 233. Caveat: the rubric has changed since that run (deep-sea keywords,
measured-OGT psychrophiles, MAGs), so "high" then and now are not identical
populations — the comparison is directionally sound, not exact.

`max_total_per_class` uncapped is a deliberate change from the config default of
100, recorded here because its silent use earlier obscured the source of
thermophile's apparent growth: removing that cap accounts for 94.9% of the
560 → 3,413 change, and the deep-sea MAGs only 3.0%.

Table: `results/final_dataset_composition.csv`

### Outgroup match rank by phenotype (locked tiers)

![Genome pairs matched at each taxonomic rank, by phenotype](/Users/jaymin/.claude-science/orgs/a6bad67d-13d2-4ca5-bb85-8556ca2e897d/artifacts/proj_b583785f018b/54650e03-52ca-4b0c-9bb6-ffc1e34eb672/v4dc04111_matched_rank_by_phenotype.png)

Measured on the 5,355 usable pairs from the selection above (`matched_rank`
column; unusable rows are excluded, since a NULL rank is the absence of a match,
not a rank level).

| phenotype | n | genus | family | order | class | phylum | genus+family |
|---|--:|--:|--:|--:|--:|--:|--:|
| psychrophile | 214 | **87.4%** | 9.8% | 2.3% | 0.5% | 0.0% | **97.2%** |
| halophile | 2,459 | 14.4% | 38.7% | 22.9% | 13.7% | 10.3% | 53.1% |
| alkaliphile | 565 | 11.9% | 33.8% | 31.3% | 19.1% | 3.9% | 45.7% |
| acidophile | 766 | 15.3% | 29.5% | 21.8% | 22.1% | 11.4% | 44.8% |
| thermophile | 1,117 | 2.1% | 25.4% | 24.1% | 24.1% | 24.3% | 27.6% |
| hyperthermophile | 234 | 1.7% | 3.8% | 6.0% | 23.5% | **65.0%** | **5.6%** |

**Phylogenetic control varies 17.5× across phenotypes** (97.2% vs 5.6%
genus-or-family). This is the confound-severity ranking for the whole dataset,
and it runs in the same direction as the emitted→usable drop: classes whose
extremophily is clade-wide both lose more candidates AND match their survivors
more distantly.

- **Hyperthermophile is the worst case: 65% of its pairs are phylum-level
  matches.** A phylum-level outgroup shares only the deepest split, so any
  classifier trained on these pairs can reach high AUPRC by learning clade
  identity rather than thermostability. Its 0.898 AUPRC in the production run
  should be read with this in mind — the pair-AUC 0.924 is the more informative
  number precisely because it is within-pair.
- **Psychrophile is the best-controlled class in the dataset** at 87.4% genus.
  Same cause as its 0% unusable rate: the measured-OGT rubric admits only
  cultivated species, which sit in densely-sequenced lineages alongside
  mesophilic congeners. An unintended benefit of a rubric chosen for a different
  reason (GenomeSPOT's cold miscalibration).
- **Thermophile's 2.1% genus** is the cost of high-only: the tier restricts it to
  genomes where keyword and prediction agree, and those concentrate in
  thermophilic clades with no mesophilic congener. High+medium would spread it
  across more mesophile-adjacent lineages — a real argument against the lock that
  the AUPRC evidence outweighs, but which should be stated.
- pH classes sit in the middle (45%) because acidophily and alkaliphily are
  scattered through otherwise-mesophilic lineages.

**Consequence for evaluation:** a clade-held-out split is not optional for
hyperthermophile and thermophile. Reporting per-phenotype AUPRC without it will
overstate both, and the overstatement is largest exactly where n is smallest.

Table: `results/matched_rank_by_phenotype.csv`

### MAG ingest into custom_genomes/ — complete (job 1164146)

`03d_ingest_mags_custom_tree.sbatch`, 48:47 wall on `gpu`, exit 0:0.

```
INGESTED 330      FAILED 0      SKIPPED_ALREADY_PRESENT 0
INDEX_BEFORE 199923  ->  INDEX_AFTER 200253  (INDEX_ADDED 330, INDEX_CU_ROWS 330)
TOTAL_PROTEINS 728895     FNA_ON_DISK 330     FAA_ON_DISK 330
```

Mean 2,209 proteins/genome (vs the 2,045 GTDB average used in the projections,
so the whole-proteome protein counts are conservative by ~8%).

Post-ingest verification, not assumed: 205 bacteria / 125 archaea on disk
matching the GTDB-Tk domain split exactly; **all 320 SignalP-needed accessions
resolve** through the same `$ROOT/$DOM/${ACC}_protein.faa.gz` lookup the GTDB
path uses (UNRESOLVED_COUNT 0); headers are bare `>{PROTID}` as the convention
requires (e.g. `>10E_k119_382340_1`).

Runtime note: **8.87 s/genome** end-to-end (2,927 s / 330), not the ~1 s I
projected from the GTDB-Tk timing run's Prodigal figure (1.74 s/genome). That
figure was Prodigal alone; this loop also gzips contigs and the proteome to the
VAST mount per genome. **Use ~9 s/genome for future custom-genome ingests.**

I first recorded 14.9 s/genome, taken from a mid-run partial (45 genomes in 11.2
min) while the job was still executing. The completed job ran 1.7x faster than
that partial implied -- early genomes pay VAST metadata and page-cache warmup that
amortises away. Do not size a job from its first few percent.

### SignalP on the 2,582 uncovered genomes — submitted

Requirement recomputed under the locked tiers: **9,286 genomes** (5,726
extremophile + 3,560 outgroup, zero overlap) — down from 13,946 before the
thermophile high-only lock. 6,641 already covered by the r232 production run plus
completed chunks of the still-running `05b` fill array (1152569, 41/190 chunks
done). **6,704 of the 9,286 were already covered, leaving 2,582 new** (2,262 GTDB
+ 320 custom); 9,286 - 6,704 = 2,582 reconciles.

An earlier note in this session put the covered count at 6,641, which no longer
reconciles (9,286 - 6,641 = 2,645). That figure was measured before the `05b`
array advanced further; total coverage rose to 10,444 genomes, of which 6,704
intersect this requirement. 6,704 is the number consistent with the 2,582 actually
staged, confirmed by the 8 chunk logs summing to 2,582 genomes.

Split 8 ways at CHUNK_SIZE=323, per the partition capacity measured immediately
before submit (`sinfo -p gpu,gpu_h200,memory -o '%P %D %C %t'`): gpu
10/342/0/352, gpu_h200 4/220/0/224, memory 3008/0/0/3008 — memory has ZERO idle
CPUs despite its short queue, so the earlier plan to use it was redirected.

| job | partition | tasks |
|---|---|---|
| 1164156 | gpu | 0–3 |
| 1164157 | gpu_h200 | 4–7 |

No `--gres` requested: SignalP 6.0 fast mode is CPU-decode-bound (see the
throughput benchmark section), so reserving a GPU would idle an accelerator for
~1.4×. `05b` left running alongside on `standard` as agreed.

### Why protein pairs per genome pair don't track the ~300-core-gene expectation

Question raised: a genome pair sharing ~300 core essential housekeeping genes
should yield ≳300 protein pairs under whole-proteome scope. Measured against the
73-pair prevalence probe, the premise mostly **holds** — the exception is the one
class that matters:

| phenotype | pairs/genome pair | genus+family match | vs ~300 expectation |
|---|--:|--:|---|
| halophile | 540.2 | 53.1% | 1.8× above |
| alkaliphile | 440.4 | 45.7% | 1.5× above |
| acidophile | 385.3 | 44.8% | 1.3× above |
| thermophile | 334.3 | 27.6% | at expectation |
| **hyperthermophile** | **43.2** | **5.6%** | **7× BELOW** |

Four of five classes reach or exceed 300. So the funnel is not systematically
lossy — **hyperthermophile specifically is**, and the cause is measurable:
pairs/gp tracks phylogenetic closeness monotonically across all five classes,
Spearman **rho = 1.000** (Pearson r = 0.973, p = 0.005, n = 5).

The mechanism is the clustering step, not gene content. `_derive_protein_pairs`
emits one pair per (cluster ∩ matched-genome-pair), so a core gene becomes a pair
only if mmseqs places **both** orthologs in the **same** cluster at 50% identity /
80% coverage. For a genus- or family-matched pair that nearly always happens; for
a phylum-matched pair it usually does not, because 50% identity sits near the
twilight zone for cross-phylum orthologs. Hyperthermophile is 65.0% phylum-matched
versus ~30% genus+family for the others, so most of its shared core is present in
both genomes yet split across separate sub-family clusters — the ortholog exists,
the pair never forms.

This is corroborated independently by the prevalence probe: at the production
threshold, maximum cluster prevalence was 108/158 genomes (0.684) and **zero
clusters spanned >70% of genomes**. There is no single pan-genome "EF-Tu cluster";
universal families fragment along phylogeny at 50% identity. That measurement was
originally made to size the redundancy cap, and it explains this too.

**Consequence.** Hyperthermophile's low yield is not a bug to be fixed by raising
a cap — it is the arithmetic of having no close mesophilic relatives, the same
root cause as its 19.6% unusable rate and its 65% phylum matching. Three levers
exist, in order of expected effect:

1. **Lower the clustering identity threshold for whole-proteome classes**
   (50% → 30–40%, or profile-based search) so cross-phylum orthologs co-cluster.
   Directly targets the measured mechanism; costs specificity, and would need a
   re-run of stage 07 for those classes only.
2. **Accept it** and rely on the class's 234 genome pairs × 43.2 = ~10.1k pairs,
   treating hyperthermophile as the small-n class it is.
3. Raising `max_per_lineage` or loosening `outgroup_match_rank` does **not** help
   — all 57 of its unusable rows have NULL matched_rank, i.e. no mesophile
   relative exists at any rank.

Not yet measured: whether a 30% threshold actually recovers cross-phylum core
pairs at acceptable precision. A cheap test is re-clustering the probe's 15
hyperthermophile genome pairs at 30/40/50% and reading pairs/gp at each.

### Correction: the pair column mixed two incompatible rate definitions

Prompted by the question "does the table need updating, unless end-to-end means
something I'm not seeing." It does mean something specific — and that is exactly
why the column was wrong.

**Two rate kinds were sitting in one column.**

- `SEC_RATE` (secreted classes) is a production **end-to-end** rate: pairs that
  survived the *full* pipeline — clustering, redundancy cap, leakage-aware split —
  per selected genome pair.
- `DIRECT` (hyperthermophile) is a prevalence-probe **raw** rate: pairs straight
  out of mmseqs, before cap or split.

Raw and end-to-end are not interchangeable. Measuring the gap on the four classes
where both exist (probe raw ÷ whole/secreted multiplier vs production end-to-end):

| phenotype | probe raw, secreted-equiv | production end-to-end | loss |
|---|--:|--:|--:|
| halophile | 43.4 | 33.9 | 1.28× |
| thermophile | 27.9 | 17.2 | 1.62× |
| alkaliphile | 21.6 | 13.6 | 1.59× |
| acidophile | 14.8 | 9.0 | 1.65× |

**Median raw→end-to-end loss = 1.60×** (range 1.28–1.65). Hyperthermophile's 43.2
was therefore ~1.6× too generous relative to its column-mates: discounted, 27.0
pairs/gp → **6,309 pairs** (was 10,108).

**Psychrophile was worse — the assumption exceeded every measurement.** Its
614–1,636 pairs/gp came from 2,045 proteins × 30–80% ortholog sharing. But the
highest whole-proteome raw rate ever measured is halophile's 540.2, so my *lower*
bound was 1.14× above the observed maximum. The error is the one the core-gene
analysis identified: it assumed each shared ortholog becomes a pair, when
clustering fragmentation means only a fraction do. Ortholog sharing is not pair
conversion.

Re-derived from the rank–yield relationship (`pairs/gp = 9.54 × genus+family% +
11.4`, R² = 0.947 on the 5 measured classes):

- **floor 337 pairs/gp** — the best-matched *measured* class (halophile, 540 raw)
  discounted 1.60×; defensible because psychrophile is better-matched still
- **ceiling 586 pairs/gp** — fit extrapolated to its 97.2% genus+family, then
  discounted
- → **72,156–125,366 pairs** (was 131,396–350,104)

⚠️ The ceiling is an **extrapolation**: the fit spans 5.6–53.1% genus+family and
psychrophile sits at 97.2%, 1.8× beyond the fitted range. Treat the floor as the
planning number and measure directly — a ~90 s mmseqs run on its 214 genome pairs
settles it.

**Revised total: 195,583–248,793 protein pairs** (was 258,622–477,330). Protein
counts are **unchanged at 4,096,172** — they derive from genome counts, not pair
rates, so the MLM training set is unaffected. Only the classifier's auxiliary
pair-margin term sees this.

### Identity-threshold sweep — the low hyperthermophile yield is a clustering artifact (job 1164165)

`12_identity_sweep_probe.sbatch`, 8:09 wall on `gpu`. Re-clustered the same
158-genome probe proteome (323,178 proteins) at 30/40/50% identity, changing only
`--min-seq-id`; all other flags byte-identical to the original probe.

**Positive control passed exactly.** id50 reproduced 43.2 / 334.3 / 385.3 / 440.4 /
540.2 pairs/gp — the published probe values to the decimal. The harness is sound,
so the other arms are interpretable.

| phenotype | genus+family | 50% | 40% | 30% | fold (30 vs 50) |
|---|--:|--:|--:|--:|--:|
| alkaliphile | 45.7% | 440.4 | 521.3 | 571.7 | 1.30× |
| halophile | 53.1% | 540.2 | 698.6 | 784.0 | 1.45× |
| acidophile | 44.8% | 385.3 | 502.8 | 574.9 | 1.49× |
| thermophile | 27.6% | 334.3 | 473.3 | 568.9 | 1.70× |
| **hyperthermophile** | **5.6%** | **43.2** | **122.8** | **210.5** | **4.87×** |

**Prediction recorded before the run: hyperthermophile 2–5×, genus/family classes
1.2–1.5×. Measured: 4.87× vs 1.30–1.70×.** The differential is **3.28× beyond the
global lift**, so this is a targeted fix, not a global sensitivity knob — the
distinction the all-five-classes control was built to decide.

Gain is inversely correlated with phylogenetic closeness (Pearson r = **−0.904**,
p = 0.035; Spearman rho = −0.900, n = 5): the more distantly a class is matched,
the more it recovers. That is the mechanism confirmed directly — lowering identity
recovers ortholog pairs that cross-phylum divergence had split into separate
clusters.

**Sensitivity control settles the confound.** id30 at default sensitivity 210.5
vs id30 `-s 7.5` 220.7 pairs/gp, only +4.8%. So the **identity threshold was the
binding constraint, not the prefilter** — had the gap been large, the result would
have been an artifact of mmseqs missing remote homologs rather than evidence about
identity.

**Specificity cost, the reason not to simply adopt 30% everywhere:**

| run | clusters | singleton % | max prevalence | total pairs |
|---|--:|--:|--:|--:|
| id50 | 224,022 | 83.7% | 0.684 | 25,580 |
| id40 | 176,087 | 78.4% | 0.848 | 34,130 |
| id30 | 130,712 | 74.4% | 0.867 | 39,925 |
| id30 -s 7.5 | 126,564 | 73.6% | 0.930 | 40,559 |

Clusters fall 41.7% as families merge, and max prevalence rises 0.684 → 0.867
(0.930 at high sensitivity), approaching one cluster per universal family. The
merging is the mechanism, but it is also the hazard: at 30% identity paralogs and
remote homologs can land in one cluster, and `_derive_protein_pairs` would then
emit a **non-orthologous** pair. This sweep measures yield, **not** pair
correctness — nothing here validates that the extra pairs are true orthologs.

**Not yet decided.** 40% is the conservative option (2.84× for hyperthermophile at
0.848 max prevalence); 30% maximises yield. Before adopting either, the added pairs
need an orthology check — the natural one is reciprocal best hit on a sample of
hyperthermophile pairs unique to the lower threshold, which also retires the
`cs_prob` tiebreak stopgap. Recommend deciding after that check, and applying the
lower threshold ONLY to whole-proteome classes.

### SignalP for the retrain — COMPLETE (jobs 1164156 + 1164157)

All 8 tasks COMPLETED, exit 0:0, `SIGNALP_RC 0` on all eight.

| task | partition | elapsed | proteins |
|---|---|--:|--:|
| 1164156_0 | gpu | 08:35:05 | 717,183 |
| 1164156_1 | gpu | 08:33:28 | 716,681 |
| 1164156_2 | gpu | 11:32:37 | 711,417 |
| 1164156_3 | gpu | 15:14:53 | 865,594 |
| 1164157_4 | gpu_h200 | 09:34:34 | 749,131 |
| 1164157_5 | gpu_h200 | 10:54:19 | 865,179 |
| 1164157_6 | gpu_h200 | 11:02:28 | 875,661 |
| 1164157_7 | gpu_h200 | 13:07:42 | 1,139,270 |

**TOTAL_ROWS 6,640,116 — exactly the staged protein count**, so no chunk silently
truncated. Wall 15:14:53, set by task 3 on `gpu` rather than the 1.14M-protein task
7 on `gpu_h200` (13:07:42): the H200s absorbed the largest chunk faster than the
A5000s handled a 24% smaller one. My ~15 h estimate from the 2-minute progress-bar
rate was right for the wrong chunk.

**Coverage against the training requirement: complete.**

```
REQUIRED (req2.txt)   9,286
PRECOVERED           10,444   (r232 production + completed 05b chunks)
TARGETED (this run)   2,582
TRAIN_COVERED        13,026
TRAIN_MISSING           320  -> resolved to 0, see below
```

The 320 apparent gaps were an **identifier artifact, not missing data**: `req2.txt`
lists MAGs under their original parenthesised names (`CU_10C(CNS0876620)_bin.42`)
because it was generated before the ingest, while the ingest renamed them to the
`CU_CUST_*` scheme. Resolving every one through `mag_ingest/map.tsv`:

```
RESOLVED_VIA_MAP            320
UNMAPPED                      0
MAPPED_BUT_NOT_PREDICTED      0
TRUE_GAP                      0
```

All 320 MAGs were predicted, as `CU_CUST_000000001.1` onward. Worth noting for
future diffs: any comparison against a pre-ingest requirement list must pass
through `map.tsv` or it will report the entire renamed set as missing.

**Prediction-class distribution** over the first 5 chunks (3,923,835 predictions):
OTHER 88.01%, SP 7.69%, LIPO 3.45%, TAT 0.47%, PILIN 0.27%, TATLIPO 0.10%. Any
signal peptide = **11.99%** vs the 11.28% r232 production fraction (+0.71 pp),
confirming nothing systematic is wrong with the new genomes. Which classes count as
"secreted" is a definition choice in `signalp.py`, not a measurement — SP alone is
7.69%, SP+TAT 8.17%, and PILIN/TATLIPO are membrane-anchored.

**Unrelated: the 05b whole-GTDB fill array (1152569) is NOT part of this.** It is
the months-long background job covering all r232 representatives, and 4 of its
tasks (13/25/27/39) hit the 3-day wall with no output. That does not touch the
retrain: the coverage diff above is scoped to `req2.txt`, and the union already
satisfies it with TRUE_GAP 0. The 4 timed-out chunks need resubmission only for the
full-GTDB secretome goal, and should be split finer — 3 days was insufficient at
their chunk size.

### RBH orthology check — the yield gain is REAL, and cleanest where it matters (job 1164334)

`13_rbh_orthology_check.sbatch`, 11:09 wall on `gpu`, 73/73 genome pairs scored.
Ground truth: `mmseqs easy-rbh` (native reciprocal best hit) between each matched
pair's two proteomes. A derived pair counts CORRECT when the two proteins are each
other's reciprocal best hit.

**Clustered at `--cov-mode 0`, the production setting** (`07_cluster_secreted.sh`),
not the `--cov-mode 1` the probe and sweep used. That re-clustering mattered:
bidirectional coverage is STRICTER, giving more clusters at every threshold
(239,614 / 196,458 / 155,720 vs the sweep's 224,022 / 176,087 / 130,712) and a
smaller merge from 50→30% (−35.0% vs −41.7%). So the sweep's 4.87× was an upper
bound; in the production regime the hyperthermophile gain is **4.55×**.

**Precision on ALL derived pairs:**

| threshold | acido | alkali | halo | thermo | hyperthermo |
|---|--:|--:|--:|--:|--:|
| 50% | 97.32 | 97.93 | 97.68 | 96.55 | 96.48 |
| 40% | 93.74 | 95.55 | 94.41 | 94.09 | 93.70 |
| 30% | 87.51 | 90.80 | 89.02 | 88.57 | 88.97 |

**Precision on pairs UNIQUE to each lower threshold — the actual decision:**

| threshold | acido | alkali | halo | thermo | **hyperthermo** |
|---|--:|--:|--:|--:|--:|
| 40% | 84.41 | 86.46 | 85.48 | 88.89 | **91.99** |
| 30% | 74.70 | 77.87 | 76.99 | 80.52 | **86.90** |

**THE RESULT INVERTS MY EXPECTATION.** I predicted the large hyperthermophile gain
would be spurious merging — big yield jump = paralogs co-clustering. It is the
opposite: hyperthermophile has both the largest gain (4.55×) AND the highest
added-pair precision at both thresholds (91.99% / 86.90% vs 74.70–88.89% for the
others). Mechanistically consistent with the phylogenetic-distance explanation:
its pairs are matched at class/phylum, so genuine orthologs simply fall below 50%
identity and are recovered by lowering the threshold, whereas the closely-matched
classes already captured their orthologs at 50% and mostly add marginal hits.

**Expected true-pair yield** (pairs × precision, all five classes):
50% → 22,120 true / 594 non-RBH; 40% → 29,581 / 1,770; 30% → 33,213 / 4,142.
Every class gains more true pairs than false ones even at 30%.

**Baseline representative choice validated as a side effect.** At 50% the derived
pairs are 96.5–97.9% RBH-consistent, so the lexicographic/`cs_prob` representative
rule is picking the right protein ~97% of the time. That retires the "is the
tiebreak arbitrary?" worry with a number.

**Caveat that bounds this whole result.** RBH is a strong orthology proxy, not
ground truth: a non-RBH pair may still be homologous (RBH breaks on recent
duplications and fragmented MAG genes), and RBH itself can pair out-paralogs. So
"87% precision" means "87% agree with RBH", not "13% are wrong". The direction and
the between-class ordering are robust; the absolute level is approximate.

**DECISION: 40% for whole-proteome classes, 50% for secreted classes.** At 40%
hyperthermophile gets 2.73× the pairs at 91.99% added-pair precision — a better
precision than the other four classes achieve at 40% and comparable to their 50%
baseline. 30% buys another 1.7× but drops added-pair precision to 86.90% and, more
importantly, pushes max cluster prevalence toward one-cluster-per-family. 40% is
the point where the sparse class gains most per unit of precision lost.

### Identifier audit before the assembly chain — one silent data-loss bug found

Asked to double-check that the MAG parenthesis renaming and the GTDB prefix
handling wouldn't create label conflicts. They would have. Five identifier spaces
are in play:

| space | form | where it lives |
|---|---|---|
| GTDB accession | `GB_GCA_000008085.1` / `RS_GCF_...` | labels, index, proteomes |
| GTDB bare | `GCA_000008085.1` | `work/genome_index.tsv` (join key) |
| MAG original id | `10A(CNS0876618)_bin.16` | `deepsea_mags_merged.tsv`, GenomeSPOT |
| MAG sanitised | `10A_CNS0876618__bin.16` | GTDB-Tk input only (its validator rejects parens) |
| MAG assigned | `CU_CUST_000000001.1` | **proteomes, FASTA headers, all SignalP predictions** |

**THE BUG.** `03c` line 109 built `accession = "CU_" + mag_id`, i.e.
`CU_10A(CNS0876618)_bin.16`. But the ingest assigns sequential accessions, so every
proteome file, FASTA header and SignalP prediction uses `CU_CUST_000000001.1`.
The two spaces share **no key**. `assign_labels` joins on `bare_accession`
equality, so **every MAG protein would have failed to get a label and been dropped
— no error, no warning, a plausible-looking output table.** This is the failure
mode that produces a quietly wrong dataset rather than a crash.

Fixed: `build_mag_rows` takes the ingest `map.tsv` and translates through it;
`03c` gains `--id-map` (validated injective), `--require-proteome`, a loud warning
when no map is given, and a hard failure if a map matches zero `mag_id` values.

**Validated on the real files, not fixtures:** 330/330 map entries matched,
0 unmatched, join to `CU_CUST_000000001.1` confirmed via `bare_accession`, 0
parentheses and 0 duplicates among mapped accessions, all retaining `CU_`.

Seven further checks, all pass: no CUST bare id resembles GCA/GCF; label-only rows
cannot collide with mapped ones; no `mag_id` contains a tab or newline (TSV
round-trip safe); the merged table does carry the original parenthesised form for
exactly 1,445 MAGs; `map.tsv` is injective 330→330→330; `genome_index.tsv` has
zero duplicate keys.

**Also settled:** `genome_index.tsv` carries the 330 MAGs keyed **bare**
(`CUST_000000001.1`), matching the GTDB convention (`GCA_...`), not prefixed. My
initial `grep '^CU_'` returned 0 and looked like a missing append; the rows are
there.

**Scope finding:** only **330 of 4,084** MAGs have a proteome on disk (the 320
needed for SignalP plus 10). The other 3,754 are retained as label-only rows —
countable in summaries, never selectable as a training genome, because a pair built
on them would contribute zero protein pairs.

### Assembly chain staged as one dependency-linked series

`scripts/slurm/14_assemble_chain.sh`:
`A0` gtdb metadata → `A` labels (01b/03/03b/03c) → `B` genome pairs (04 per class)
→ `C` secreted table + both clustering FASTAs → {`D` cluster 50%, `E` cluster 40%}
→ `F` assemble. Every step `afterok`-dependent, so a failure halts the chain rather
than feeding garbage forward.

Four defects found by dry-validating rather than submitting:

1. **Chain started mid-pipeline.** The stage-01b metadata flags parquet and the
   combined-labels parquet are no longer on the cluster (in-kernel state lost on
   reset), so the chain now starts from the GTDB metadata dumps. A repo checkout
   *does* exist at `eptrans_scratch/repo` — I had earlier said there was none,
   which was wrong.
2. **No interpreter.** Bare `python` doesn't exist on the non-login PATH, `conda`
   isn't on it either, and `/usr/bin/python3` lacks pandas. All 10 call sites now
   use the absolute `eptrans_ml` interpreter (verified: pandas 3.0.3, pyarrow
   25.0.0).
3. **Wrong flags.** 03b was called with `--out`, which it doesn't accept. Every
   stage's flags are now checked against its own argparse: **6 stages, 34 flags,
   zero unknown.**
4. **03a can't run in a batch job.** It fetches over the network and compute nodes
   have no egress. It now runs as a login-node preflight, which also hard-fails
   early if the merged MAG table isn't staged.

`05_aggregate_signalp.py` is new. Smoke-tested on real output (chunk_0): 717,183
predictions parsed, **72,804 secreted (10.15%)** across 323 genomes; SP 50,123 /
LIPO 17,648 / TAT 2,683 / PILIN 2,032 / TATLIPO 318 / OTHER 644,379. It carries an
`anchoring` column so a later stage can restrict to `soluble` without re-running
SignalP — 19,998 of the secreted calls are membrane-anchored, not released.

**Chain defect 5, found by testing the A0 handoff rather than reading it.** An
audit flagged that A0's `gtdb_meta.tsv` (10 columns: `domain`, `accession`,
`gtdb_taxonomy`, `ncbi_isolation_source`, `ncbi_organism_name`, plus checkm/rep
columns) might not match 01b's `--tsv` help string, which paraphrases the format as
"(domain, accession, isolation, organism, taxonomy)". The help string is a
paraphrase; 01b's module docstring (lines 8-9) names the real columns —
`ncbi_isolation_source`, `ncbi_organism_name`, `gtdb_taxonomy` — which A0 emits
exactly, and its `read_csv` takes all columns as-is with extras ignored.

Proven end-to-end on a 2,000-row real sample built with the exact A0 code:
representatives 2,000, with isolation_source 980, iso-flagged 247, org-flagged 418
(thermophile iso 136 / org 282, hyperthermophile 27/15, psychrophile 16/0,
acidophile 58/215, alkaliphile 15/14, halophile 40/99), both parquet and figure
written. The handoff is sound.

That test did expose a real blocker the audit didn't name: **no conda environment on
biotite had matplotlib**, and four of the seven chain stages (01b, 03, 04, 06)
import it at top level, so the chain would have died at stage A regardless of the
column question. Installed matplotlib 3.11.1 into `eptrans_ml` (now pandas 3.0.3 /
pyarrow 25.0.0 / matplotlib 3.11.1).

Lesson worth keeping: checking a consumer's *help text* is not checking its
*contract*. Running one stage on one real sample found both the false alarm and a
true blocker in the same minute.

`deepsea_mags_merged.tsv` staged to `$W` (gzip + base64, 66 chunks; md5
`549575c1ee49e12e409ae93684adb681` verified on arrival, 4,085 lines, 20 columns).

**Chain submitted (2026-08-04).** `1164413` A0 gtdb metadata → `1164414` A labels →
`1164415` B genome pairs → `1164416` C secreted + FASTAs → {`1164417` D cluster 50%,
`1164418` E cluster 40%} → `1164419` F assemble. All `afterok`-linked.

The login-node preflight fetched all four OGT sources on submit (tempura 1,464,617 B;
madin 46,362,149 B; toki 453,138 B; ogtfinder 7,360,680 B), confirming that keeping
03a out of the batch jobs was necessary — compute nodes have no egress.

Two further defects caught at actual submit time, which no amount of reading would
have found:

1. **`PY: unbound variable`.** `set -u` aborted the script before submitting
   anything: the preflight calls `$PY` but the definition sat further down beside
   `SB`. Moved above the preflight.
2. **Login node lost DNS to github.com** mid-session (`Could not resolve host`), so
   `git pull` silently left the cluster one commit behind — still carrying the
   broken ordering. Installed the corrected script by direct transfer instead
   (md5 `b38c00099726019c64efb21c8ebcf817` verified on arrival, `bash -n` clean).
   Worth remembering: on this host `git pull` is not a reliable way to ship a fix.

Queue state at submit: 9 RUNNING / 3 PENDING from unrelated work (deeploc arrays,
operon jobs, the long-running SignalP fill `1152569`). MaxJobsPerUser=10 caps
concurrent *running* jobs, not submissions (MaxSubmitJobsPerUser=200), so the chain
pends rather than being rejected.

**Resubmitted onto the `memory` partition (2026-08-04).** Cancelled the standard-partition
chain (`1164413`–`1164419`, all still PENDING, nothing lost) and resubmitted as
`1164424` A0 → `1164425` A → `1164426` B → `1164427` C → {`1164428` D, `1164429` E}
→ `1164430` F.

**Chosen on queue depth, not idle capacity.** Both partitions reported zero idle
CPUs (standard 3680/0/48/3728, memory 3008/0/0/3008), so the throughput argument
isn't about free cores — it's the backlog: cluster-wide pending jobs were **11 on
`memory` against 139 on `standard`**, a 12.6× shorter queue. Secondary benefits:
memory nodes carry 677 GB vs standard's 258 GB, which suits the 40% whole-proteome
clustering (the largest job in the chain, previously requesting 320 G on a 258 G
partition — it would have been unschedulable as written), and `memory` is
time-unlimited here.

Partition is now a `PART` variable (default `memory`) rather than hardcoded.

Dependency graph verified post-submit via `scontrol` rather than assumed: A0 `(null)`,
each stage `afterok` on its predecessor, D and E both `afterok:1164427`, and F
`afterok:1164428,afterok:1164429` — so the assembly waits on *both* cluster maps,
which is required since the split is grouped on their merge.

The preflight correctly skipped all four OGT fetches as already present
(idempotent), so the resubmit cost no network work.

**asm_meta failed instantly — dash has no `pipefail` (2026-08-04).** Job `1164424`
exited 2 in under a second:

```
/var/spool/slurmd/job1164424/slurm_script: 4: set: Illegal option -o pipefail
```

`sbatch --wrap` bodies execute under `/bin/sh`, and on biotite `/bin/sh -> dash`,
which does not implement `pipefail`. All **7** wrap bodies carried
`set -uo pipefail`, so the chain would have died at whichever stage ran first —
this was never partition-related.

**A wrong fix I tried and discarded, recorded so it isn't retried.** Prefixing each
wrap body with `#!/bin/bash` does *not* work: SLURM prepends its own `#!/bin/sh` to
the generated `slurm_script`, so a shebang inside the body lands mid-file and is
inert. The tell was in the original error — it reported `slurm_script` **line 4**,
not line 1. Confirmed with real dash locally: `dash script_with_bash_shebang` still
rejects `pipefail`, because a shebang only governs execution when the kernel execs
the file directly. I had "verified" the shebang approach against local `/bin/sh`
first, which proved nothing — macOS `/bin/sh` is bash-backed and accepts `pipefail`
happily.

Also considered and rejected: wrapping each body in `/bin/bash -c '...'`. The bodies
contain heredocs and nested quotes, so an extra quoting layer is a correctness
hazard for no benefit.

**Adopted:** wrap bodies use plain `set -u`; `pipefail` stays in the driver script,
which has a real shebang. Verified this costs nothing — audited every wrap body for
pipes whose failure `pipefail` would have caught and found **none**: the `|`
occurrences are Python bitwise-or and shell `||`, and every `wc -l <` is a redirect,
not a pipe. The bodies are `&&` chains, and dash already stops on the first non-zero
(`dash -c 'set -u; false && echo X'` exits 1).

Resubmitted as `1164438` A0 → `1164439` A → `1164440` B → `1164441` C →
{`1164442` D, `1164443` E} → `1164444` F on `memory`.

**Practice change:** shell-portability bugs in `--wrap` bodies cannot be caught
locally, because macOS `/bin/sh` is bash. Test wrap-body syntax with `dash -c`
explicitly before submitting.

**Unblocked by splitting the chain across partitions (2026-08-04).** After the dash
fix, `1164438` still would not start: `Reason=Resources`, SLURM estimating a 23:21
start (~1.5 h out). The request was not the problem — A0 asks 4 CPUs / 32 G against
nodes offering 64+ cores / 677 G.

**Every CPU partition was 100% allocated**, measured simultaneously:
standard 3680/0/0/3680, memory 3008/0/0/3008, high-memory 792/0/0/792,
min_1500g 1240/0/0, min_3t 792/0/0, min_8t 344/0/0 — zero idle cores anywhere.
But the **GPU partitions had large idle CPU counts**: `gpu` 262 idle of 352,
`gpu_h200` 186 of 224, because they gate on GPUs rather than cores.

So the chain now splits: the four CPU-only python stages (A0/A/B/C) go to
`LIGHT_PART` (default `gpu`, no GPU requested), while the two clustering jobs and
the assembly stay on `PART` (default `memory`) for the RAM. Both are variables.

Resubmitted `1164445`–`1164451`. **A0 COMPLETED in 22 s** — versus a 1.5 h queue
estimate on `memory`. Stage A started immediately after.

**A0 output validated rather than assumed:** `gtdb_meta.tsv` has 901,341 rows, which
looked wrong against 199,923 species representatives. Counting the
`gtdb_representative` column resolves it — **exactly 199,923 rows carry `t`**,
matching the known r232 representative count, because the GTDB metadata dumps cover
all genomes and 01b filters to reps itself.

**Heavy stages moved to `gpu_h200` — and a cluster-wide scheduling fact behind it.**
`gpu_h200`'s single node carries **2,063,701 MB (2 TB)** with 186 of 224 CPUs idle,
which exceeds the largest request in the chain (320 G / 48 CPU) while `memory` sat
at 3008/0/0. All three heavy stages (D, E, F) now default to `HEAVY_PART=gpu_h200`;
the four light stages stay on `LIGHT_PART=gpu`. No GPU is requested — verified by
`AllocTRES=cpu=N` with no gres on the stages that already ran, plus
`MaxTime=UNLIMITED` and `AllowAccounts=ALL` on that partition.

**Biotite runs `SelectTypeParameters=CR_CPU`.** Memory is *not* a consumable
resource here: `--mem` reserves nothing and does not gate scheduling. Confirmed on
every node — `AllocMem=0` while `AllocTRES=cpu=64` on a fully allocated
`memory`-partition node. Two consequences worth carrying forward:

- Only CPUs gate scheduling, so a job pending on `Resources` is waiting on **cores**,
  never on RAM.
- A heavy job can be co-scheduled onto a node with others and **OOM**, since nothing
  reserves its footprint. Preferring the 2 TB node is therefore a genuine safety
  margin, not just a queue-time optimisation. `FreeMem` (real, live) is the number to
  check before placing a large job — not `RealMemory`.

Measured free memory at the time of the move: `node-344-8t-1` 1,731,092 MB free;
`node-224-3t-1` 692,176; `node-224-2t-8gpu-1` (gpu_h200) 272,817; `node-64-768g-1`
19,622; `node-128-512g-8gpu-1` only **1,949 MB free** despite 515,799 MB installed —
a vivid illustration that installed capacity says nothing about availability under
CR_CPU.

**Stage C bug caught from its own log: `WHOLE_SCOPE_GENOMES 0 classes []`.** My
inline scope-derivation snippet iterated the TOP level of
`dataset.protein_scope`, but the per-class map is nested under `by_phenotype` — the
top level holds only `default` and `by_phenotype`, so the comprehension matched
nothing and returned `[]`. The whole-proteome FASTA would have been written empty and
stage E would have hard-failed its non-empty guard (that guard is why this surfaced
as a diagnosable log line rather than a 16-hour clustering run on an empty file).

`06_assemble_dataset.py` already reads this correctly (`ps.get("by_phenotype", {})`)
and even raises if it is empty — my snippet simply didn't match the repo's own
contract. Fixed to use the same access path, and it now `sys.exit`s loudly if either
`by_phenotype` is empty or no class is whole-scope. Verified against the real config:
`['hyperthermophile', 'psychrophile']` vs the buggy `[]`.

Cancelled C and its dependents; A0/A/B outputs were correct and preserved
(`gtdb_meta.tsv` 192,586,661 B, `combined_labels.parquet` 84,804,901 B,
`all_pairs.tsv` 514,004 B, all three done-markers present). Resubmitted
`1164460`–`1164466`. Note the rerun does redo A0/A/B — I passed a `SKIP_DONE=1` the
script does not implement. Harmless (~5 min, idempotent), but the done-markers the
chain writes are currently decorative; wiring them into a real skip is worth doing.

Stage B results, first time measured end-to-end on the merged dataset: **6,568 genome
pairs** — 2,667 `high` / 3,901 `medium` — across all six classes (halophile,
thermophile, acidophile, alkaliphile, psychrophile, hyperthermophile all non-empty).

### Chain running clean through stage C (1164460–1164466)

| stage | job | partition | state | elapsed |
|---|---|---|---|---|
| A0 gtdb meta | 1164460 | gpu | COMPLETED | 00:00:21 |
| A labels | 1164461 | gpu | COMPLETED | 00:07:52 |
| B genome pairs | 1164462 | gpu | COMPLETED | 00:01:24 |
| C secreted + FASTAs | 1164463 | gpu | RUNNING | — |
| D cluster 50% | 1164464 | gpu_h200 | PENDING (dep) | — |
| E cluster 40% | 1164465 | gpu_h200 | PENDING (dep) | — |
| F assemble | 1164466 | gpu_h200 | PENDING (dep) | — |

**The scope fix is confirmed on real data**, which was the open risk:
`WHOLE_SCOPE_GENOMES 7320 classes ['hyperthermophile', 'psychrophile']`, against the
previous run's `0 classes []`. Stage C is now past the point where it silently
produced an empty whole-proteome FASTA.

**Stage A (labels).** 901,341 GTDB genomes flagged; isolation source present for
483,545; iso-flagged 16,598, org-flagged 26,130. Confidence tiers after GenomeSPOT
merge: high 3,642 / medium 18,859 / low 12,494 / none 866,346; confident mesophiles
104,279. Final per-class: thermophile 15,047, halophile 11,019, acidophile 5,968,
psychrophile 3,748, alkaliphile 2,496, hyperthermophile 1,365. Measured-OGT merge
pooled 9,577 species (tempura 4,312 / madin 4,749 / toki 3,131 / ogtfinder 3,168),
matched 5,907 to r232, giving OGT tiers high 694 / medium 1,520 / low 4,686, with 432
genomes ≤15 °C, 2,077 ≤20 °C, 63 promoted by Tmin, and **450 measured-warm genomes
overriding cold metadata**. Deep-sea confident mesophiles 1,576.

**Stage B: 6,568 taxa-matched genome pairs**, the first end-to-end count on the
merged GTDB + deep-sea dataset:

| class | pairs | high | medium |
|---|--:|--:|--:|
| halophile | 2,934 | 522 | 2,412 |
| thermophile | 1,630 | 1,630 | 0 |
| acidophile | 798 | 300 | 498 |
| alkaliphile | 581 | 115 | 466 |
| psychrophile | 332 | 12 | 320 |
| hyperthermophile | 293 | 88 | 205 |

Thermophile is high-only and halophile high+medium, as locked in earlier.

**Verified the label table's scope rather than trusting the row count.**
`combined_labels.parquet` has 905,425 rows = 901,341 GTDB + 4,084 MAGs, of which
exactly **199,923 carry `gtdb_representative == 't'`** — the known r232 count. The
table deliberately spans all genomes; selection filters. Confirmed no leakage: of the
**10,092 distinct genomes appearing in the 6,568 pairs, 10,092 are representatives or
MAGs and 0 are non-representative GTDB genomes.** 439 MAGs participate in pairs.

### Whole-proteome scope was about to lose 90.6% of its genomes, silently

Chasing the psychrophile confidence question surfaced a data-loss bug in
`05_aggregate_signalp.py`. The FASTA emission loop iterated
`df["genome"].unique()` — the genomes present in the **SignalP prediction
table**. But whole-proteome scope wants every protein *regardless of secretion*, so
SignalP coverage is irrelevant to it. A whole-scope genome with no predictions was
never opened and contributed **zero** sequences.

Measured on the real run: of the **7,320** whole-scope genomes (hyperthermophile +
psychrophile), only **685** appear in the prediction table — 630 via the legacy
secreted-only r232 table and 55 via fresh targeted chunks. **6,635 (90.6%) would
have contributed nothing.**

**The emptiness guard would not have fired.** It raises only when the whole FASTA is
empty, and 685 genomes still produce a large file — so the 16 h clustering job would
have run happily on a corpus missing nine tenths of its intended genomes, and the
loss would have surfaced (if at all) as an unexplained shortfall in hyperthermophile
and psychrophile pair counts at the very end.

Fixed: the loop now walks `set(df["genome"]) | whole`. Added a log line reporting
requested vs in-prediction-table vs actually-read counts, so the gap is visible
rather than inferred. Regression test
`tests/test_whole_scope_needs_no_signalp.py` builds a two-genome fixture where one
genome has predictions and one has none, and asserts the un-predicted genome
contributes all its proteins to the whole FASTA (5 sequences, not 2) while the
secreted FASTA is unchanged at 1. **139 passed, 1 skipped.**

Also verified while diagnosing: the legacy `secreted_proteins_r232.tsv` is
**secreted-only** — 300k sampled rows contain SP 193,620 / LIPO 85,964 / TAT 9,809 /
PILIN 7,866 / TATLIPO 2,740 and **zero OTHER** — across 7,268 genomes. So it cannot
supply whole-proteome content for any genome; it only makes them *visible* to the
loop.

**Method note:** `comm -12` gave two contradictory answers for the same overlap (630,
then 0) because the two inputs were sorted under different collations. Recomputing
with Python set intersections gave the stable figures above. Prefer set logic over
`comm` when the sort provenance of either file is uncertain.

### Psychrophile confidence: high+medium confirmed for pairs

Stage B selects psychrophile at `high,medium`, yielding **332 pairs = 12 high + 320
medium**. Tier populations in the merged label table: high 12, medium 427, **low
4,690** (3,191 of them selectable representatives/MAGs) — so `low` is a 7.3× larger
pool than high+medium combined.

Holding at high+medium for pair selection. The `low` tier for psychrophile means
habitat-keyword-only cold evidence, measured at a **0.14–0.63%** hit rate across four
independent populations (n = 1,452 / 639 / 443 / 1,416); admitting it as labelled
contrast would inject ~99% label noise into the class that already has the weakest
signal.

The earlier asymmetric plan (`low` admitted for MLM only, weight 0.10) is *partly*
satisfied by a different mechanism: the whole-scope accession list carries all
4,690 low-tier psychrophiles, so they enter the clustering and MLM corpus without
becoming labelled pairs — **conditional on the FASTA fix above**, without which they
were not going to be in the corpus at all.

### Stage C completed — the whole-scope fix worked, and exposed a second defect

Stage C (`1164488`) COMPLETED in **50:24**. Both clustering jobs also finished:
D 50% in **08:46**, E 40% in **21:28**. F is running.

**The whole-scope fix is confirmed on production data:**

```
[05agg] proteins 119,678,568 | secreted 17,888,982 (14.95%) | genomes 51,786
[05agg] whole-scope genomes: 7,320
[05agg] wrote secreted FASTA 2,057,964 seqs | whole FASTA 10,874,729 seqs
[05agg] whole-scope requested 7,320 | in prediction table 2,403 | read regardless 7,320
```

Only **2,403 of 7,320** whole-scope genomes appear in the prediction table, so the
old loop would have emitted roughly a third of the whole-proteome corpus. The whole
FASTA holds **10.87 M sequences** against the secretome's 2.06 M — a **5.28×**
corpus for the two whole-proteome classes.

**Second defect, found in the same log line: `WARNING: 3,465 genomes had no proteome
file`.** All 3,465 are whole-scope genomes, and all are non-representatives with
`has_proteome == False` (1,628 MAGs, 1,837 GTDB). Zero representatives were lost, so
the bulk of that warning is an *expected* absence — 3,855 of 7,320 whole-scope
genomes made it into the FASTA.

But **10 of those missing genomes appear in the actual pair table**, which is not
benign: a pair whose genome has no sequences derives **zero protein pairs**, silently.
All 10 are MAGs with `has_proteome == False`. Damage by class:

| class | affected pairs | of which high-confidence |
|---|--:|--:|
| psychrophile | 7 | **2** |
| hyperthermophile | 3 | 0 |
| halophile | 1 | 0 |

**2 of the only 12 high-confidence psychrophile pairs — 17% of the scarcest tier in
the dataset — were void.**

Fixed at the source: `select_with_outgroups` gains `require_col`, applied to
extremophiles *and* outgroups before any diversity capping, so the caps spend their
budget on usable genomes and pick replacements. Exposed as `--require-col` on stage
04 and wired into both stage-B calls as `--require-col has_proteome`.

**Null semantics matter here and nearly caused a much worse bug.** `has_proteome` is
written only by the custom MAG ingest: all **901,341** GTDB rows carry `None`, and
only the 4,084 MAG rows carry True/False (330 True / 3,754 False). A naive
`fillna(False)` gate would have dropped **the entire GTDB catalogue**. The gate
therefore treats null as usable — `col.isna() | col.fillna(False)` — and only an
explicit False excludes. Verified: True→keep, False→drop, None→keep.

**Measured on the real labels, psychrophile high+medium:** the gate removes all ten
bad genomes — checked **individually, one grep per genome, all ten returning 0**, not
extrapolated from a spot check — and yields **330 pairs against the previous 332**, so
7 void psychrophile pairs were replaced by usable substitutes and only 2 net pairs
were lost.

**Correction to the pair count.** I first reported "10 affected pairs" by reading the
count of affected *genomes* and by querying only the two whole-proteome classes. The
correct figure over all classes is **11 pairs from 10 distinct genomes** — one genome
appears in two pairs — broken down psychrophile 7 (2 high, 5 medium),
hyperthermophile 3 (medium), halophile 1 (medium). The halophile pair was missed
entirely by the class-restricted query. Verified by intersecting the bad-genome set
against both accession columns of the full pair table.

**Separate gap noticed, not yet fixed:** stage B does not pass `--max-per-sample`,
which stage 04 does not expose either, so the per-sample cap decided earlier is
currently inactive in this chain. Worth closing before the deep-sea MAGs dominate any
single class.


### Correction: the deep-sea hit-rate denominator is 1,416, not 1,417

Both figures are real and they are not interchangeable:

- **1,417** = deep-sea MAGs flagged psychrophile by habitat keywords.
- **1,416** = those of them that actually carry a GenomeSPOT prediction, and therefore
  the correct denominator for the 0.14% habitat-only hit rate.

The gap is the **two MAGs (of 4,084) that produced no GenomeSPOT prediction** — a
known loose end recorded earlier and never examined. Using 1,417 as the denominator
silently assumes those two were evaluated and failed, when in fact they were never
evaluated at all.

The four-population range is therefore **n = 1,452 / 639 / 443 / 1,416** at
0.41% / 0.63% / 0.45% / 0.14%. Percentages and the 0.14-0.63% range are unchanged, so
no downstream weight or decision moves; the `low` = 0.10 anchor and the
exclude-psychrophile-low conclusion both stand. Corrected here because a denominator
that quietly absorbs un-evaluated genomes is the kind of error that compounds when
someone later recomputes a rate from it.

### Restage decision: psychrophile_low out of whole scope, --max-per-sample 5, cap null

Stage F (`1164491`) cancelled at **01:06:49** — no output written. The read probe
measured **12.6 MB/s** on the 36 GB `secreted_all.tsv` (119.7 M rows, ~18 GB RAM,
7 columns), i.e. **~48 min of pure parsing** before any work begins, which matches an
hour of silence. F was superseded regardless: its cluster maps were built from a FASTA
that was **50.8% clusters existing only because of psychrophile_low**.

**Decision 1 — exclude psychrophile_low from the whole-proteome corpus.** Measured
basis:

Two genome counts are in play and they are NOT interchangeable — the column below is
**FASTA-present genomes** (those that contributed sequences), not label-table genomes:

| bucket | genomes IN FASTA | genomes in LABEL TABLE | seqs | pct of whole FASTA |
|---|--:|--:|--:|--:|
| psychrophile_low | 1,835 | **4,690** | 6,262,304 | **57.6%** |
| hyperthermophile_low | 490 | 1,069 | 1,326,324 | 12.2% |
| mesophile_outgroup | 542 | — | 1,251,496 | 11.5% |
| psychrophile_medium | 424 | 427 | 1,116,579 | 10.3% |
| hyperthermophile_medium | 454 | 478 | 720,492 | 6.6% |
| hyperthermophile_high | 100 | 100 | 156,285 | 1.4% |
| psychrophile_high | 10 | 12 | 41,249 | 0.4% |

The gap between the two columns is genomes with **no proteome on disk**
(`has_proteome == False`) — 3,465 of the 7,320 requested whole-scope genomes, which is
also why the FASTA-present total is 3,855. Verified directly (job 1164609): label table
psy_low = 4,690 / hyp_low = 1,069; FASTA-present psy_low = 1,835 / hyp_low = 490, with
6,262,304 and 1,326,324 sequences respectively. Quoting 1,835 without the qualifier
reads as a claim about how many low-tier psychrophiles exist, which is wrong by 2.55x.

**It costs zero protein pairs.** Of the whole-proteome map's 3,348,165 clusters,
**1,702,137 (50.8%) contain no non-low member** — and a cluster with no outgroup member
can never form a matched pair. **82.1% of those are singletons** (mean size 1.52). A
further **58.8%** of psy_low sequences sit in clusters shared with retained genomes,
so they were redundant anyway. What is actually lost is MLM pretraining diversity, and
that is a bet against four independent populations showing no habitat-only cold signal
(0.14-0.63%, n=1452/639/443/1416).

hyperthermophile **keeps** low: the cold-end calibration failure does not apply to it,
and at 293 genome pairs it is the class most starved of data.

Implementation detail that matters: pair members are added to the scope list
**unconditionally**, after the tier filter. Stage B already confidence-gated them, and
dropping one here would void a matched pair — the exact failure the `has_proteome`
gate was written to prevent.

**Decision 2 — `--max-per-sample 5`, flat, not scaled by class size.** `source_sample_id`
is populated only by the MAG ingest, so the cap bites on MAG-derived genomes only:

| class | selected | with sample id | samples | max/sample | removed at 5 |
|---|--:|--:|--:|--:|--:|
| thermophile | 1,630 | 331 | 79 | **22** | **107** |
| halophile | 2,934 | 34 | 32 | 2 | 0 |
| acidophile | 798 | 19 | 14 | 3 | 0 |
| hyperthermophile | 293 | 15 | 5 | 7 | 2 |
| psychrophile | 332 | 4 | 4 | 1 | 0 |
| alkaliphile | 581 | 1 | 1 | 1 | 0 |

**Class size does not predict sample concentration** — halophile is the largest class
and has almost none; thermophile is mid-sized and carries all of it. Scaling the cap
by class size would give the loosest cap to the class that needs it least. Flat 5.
Cost noted honestly: thermophile loses 107 of 331 MAG-derived genomes (32%).

`--max-per-sample` was already implemented in `select_extremophiles` (with correct
null handling — GTDB isolates are never bucketed together) but **was never exposed on
stage 04**, so it has been inactive in every run to date. Now exposed and forwarded.

**Decision 3 — `max_pairs_per_cluster_class` stays `null`.** The proposed
k = 0.25 x genome_pairs is a **measured no-op**: on the real 90,984-pair table it
removes **0 pairs (0.000%)** for every class.

| class | max cell | 0.25 x gp | verdict |
|---|--:|--:|---|
| halophile | 157 | 733 | no-op |
| thermophile | 39 | 407 | no-op |
| alkaliphile | 39 | 145 | no-op |
| acidophile | 21 | 199 | no-op |
| hyperthermophile | **3** | 73 | no-op |

Cell-size distribution: **mean 1.63, median 1, q99 = 10, q99.9 = 34, max 157**. The
largest cell in the dataset is smaller than the smallest proposed k.

**This corrects my own earlier reasoning.** I rejected k=50 because it was a no-op for
hyperthermophile and proposed scaling k by genome-pair count instead — without ever
measuring the cell sizes the cap acts on. **Hyperthermophile's max cell is 3.** No cap
above 3 can affect it under any parameterization. The prevalence spectrum had already
implied this (max f = 0.684, zero clusters above 0.70): universal families fragment
along phylogeny at these identity thresholds, so the pathological cells the cap was
designed for do not exist. The mechanism stays implemented and tested; switching it on
later is one config line.

**RBH check (job 1164334) completed**, settling the identity-threshold specificity
cost that was open:

| threshold | hyperthermophile pairs/gp | precision |
|---|--:|--:|
| 50% | 51.1 | 96.48% |
| 40% | 139.6 | 93.70% |
| 30% | 232.7 | 88.97% |

40% buys hyperthermophile **2.73x the pairs for 2.8 points of precision**; 30% costs a
further 7.5 points for 1.7x. Supports the 40% threshold already used for
whole-proteome classes.

**Cluster-map naming is misleading and cost me a wrong measurement.** `clu50_cluster.tsv`
holds the **secretome** map (2,057,964 members) and `clu40_cluster.tsv` the
**whole-proteome** map (10,874,729). The suffixes are identity thresholds, not scopes.
My first novelty measurement walked clu50 and reported psy_low as 0.5% of clusters;
against the correct map it is 50.8%. Same code, wrong file, answer off by 100x.

### hyperthermophile_low has the same zero-pair structure — the user was right

Asked whether keeping `low` for hyperthermophile avoids the problem that removed it for
psychrophile. **It does not.** Measured (job 1164610):

| | psychrophile_low | hyperthermophile_low |
|---|--:|--:|
| genomes (label table) | 4,690 | 1,069 |
| **in selected pairs** | **21** | **4** |
| clusters touched | 1,998,322 | 568,194 |
| only-this-set (no pair possible) | 1,702,137 (50.8%) | **394,463 (11.8%)** |
| of which singletons | 82.1% | **88.8%** |
| shared (pair possible) | 208,394 | 85,940 |
| seqs in only-clusters | 2,581,283 | 469,512 |

**4 of 1,069 hyperthermophile_low genomes appear in any selected pair.** Structurally
identical to psychrophile_low: the overwhelming majority reach clusters that contain no
non-low member, and a cluster with no outgroup member can never yield a matched pair.
88.8% of those clusters are singletons — a higher singleton fraction than
psychrophile_low's 82.1%.

So the justification for keeping it was never about pairs, and I should not have implied
it was. It buys **MLM pretraining corpus only**: 469,512 sequences in 394,463
pair-incapable clusters.

**Where the two classes genuinely differ is evidence quality, not pair contribution.**
psychrophile `low` rests on habitat keywords measured at **0.14-0.63%** precision
(n=1452/639/443/1416) — the cold end where GenomeSPOT is miscalibrated. hyperthermophile
`low` sits at the hot end, where the same predictor works: against TEMPURA, thermophile
(>=50 C) recall 79.9% / precision 95.5% and hyperthermophile (>=80 C) recall 69.8% /
precision 92.3%. Its low tier is metadata-only-or-conflicting, not
measured-to-be-wrong.

That makes keeping hyperthermophile_low defensible as *unlabelled* MLM data in a way
psychrophile_low is not — but the honest framing is that it is a **1.4x corpus
addition for the most data-starved class (293 genome pairs)**, contributing **zero
pairs**, not a pair-yield decision. Flagged for the user rather than silently retained.

### Chain resubmitted: 1164614 -> 1164620, all on gpu_h200

Dropped `hyperthermophile_low` too, on the measurement that it contributes **4 of 1,069
genomes** to any selected pair — the same zero-pair structure as psychrophile_low, just
smaller (394,463 pair-incapable clusters, 11.8% of all, 88.8% singletons). Keeping it
would have bought 469,512 sequences of MLM corpus and no pairs. `WHOLE_CONF` is now
`{'psychrophile':('high','medium'),'hyperthermophile':('high','medium')}`.

**Whole-scope shrinks 7,320 -> 1,566 genomes (−78.6%):**

| | genomes |
|---|--:|
| hyperthermophile (high+medium) | 578 |
| psychrophile (high+medium) | 439 |
| pair members added unconditionally | 549 |
| **total** | **1,566** |

Dry-run verified (job 1164613, 2 s): `in NEW not OLD = 0` — the filter only removes.
Low-tier survivors are exactly the pair members it must keep: **psy_low 4 of 4,690,
hyp_low 1 of 1,069**. Those 5 are retained deliberately; dropping a pair member would
void a matched pair, which is the failure the `has_proteome` gate was written to stop.

**Two different pair counts are in play — reconciled here (job 1164623) because they
appear side by side and look contradictory:**

| set | definition | psy_low | hyp_low |
|---|---|--:|--:|
| A | member of **any** selected pair (all 6 classes) | 21 of 4,690 | **4** of 1,069 |
| B | member of a **whole-scope-class** pair (hyperthermophile/psychrophile only) | 4 of 4,690 | **1** of 1,069 |

Set A is the pair-yield argument for dropping the tier; set B is what survives the
whole-scope class filter. The gap has a clean explanation: of the 4 hyp_low genomes in
any pair, **3 serve thermophile pairs and 1 a psychrophile pair — none is in a
hyperthermophile pair**, so the whole-scope filter (which only unions pairs whose class
is hyperthermophile or psychrophile) keeps just the 1. Both numbers are correct; quoting
either without its definition reads as contradicting the other.

Expect the whole-proteome FASTA to fall from 10,874,729 sequences to roughly 2-3 M, so
stage E's 40% clustering (21:28 last run at 10.9 M) and stage F both get materially
cheaper.

**Chain state at submit:** `1164614` A0 -> `1164615` A -> `1164616` B ->
`1164617` C -> {`1164618` D 50%, `1164619` E 40%} -> `1164620` F. All seven on
`gpu_h200`; A0 went RUNNING immediately, where the previous dry run had sat PENDING
(Priority) for 10+ minutes on `standard`.

Fixes carried in this submission:
1. whole-scope FASTA no longer requires SignalP coverage (`f5afb40`)
2. `has_proteome` gate on selection, null-safe (`3a93f01`, `a22da94`)
3. `--max-per-sample 5` now actually reaching stage 04 (`22ebdb4`)
4. psychrophile_low + hyperthermophile_low out of whole scope
5. `max_pairs_per_cluster_class` left `null` — measured 0.000% effect
6. all stages on `gpu_h200` (`39db2d5`)

**Two operational lessons worth keeping.** (a) Scripts for compute nodes must live on
shared storage: the first gpu submission died with `can't open file '/tmp/sct.py'`
because `/tmp` is node-local. (b) Use `python -u` in sbatch wraps, or stdout buffers
until exit and a running job looks hung — this is what made an 8-minute dry run appear
to produce nothing.

**And a self-inflicted one:** that 8-minute dry run was 8 minutes because of a
quadratic line in my own diagnostic —
`{g for g in psy if dict(zip(d['a'], d['final_confidence'])).get(g)=='low'}` rebuilt a
905k-entry dict on each of 5,129 iterations (~4.6e9 ops). Hoisted, the same script runs
in **2-3 s**. I attributed the delay to data volume before reading my own code; the
inputs are 85 MB and 514 KB.

### DependencyNeverSatisfied: stage B died on a stale cluster copy of 04_select_genomes.py

`1164616` (B) failed in **1 second, exit 2**:

```
04_select_genomes.py: error: unrecognized arguments: --max-per-sample 5
```

`1164617` (C) then reported `DependencyNeverSatisfied` and D/E/F sat on `(Dependency)`.
The dependency error was the symptom; the cause was **an argument error one stage up**.

**Root cause: partial deployment.** I added `--max-per-sample` to
`scripts/04_select_genomes.py` locally, then transferred `selection.py` and
`14_assemble_chain.sh` to the cluster — but never re-transferred stage 04 itself. The
chain passed a flag the cluster's copy of the script did not have. Both the library
function and the caller were correct; the intermediate CLI was two commits behind.

**Fix + audit.** Installed 04 (md5 `e47cc178803aa476e6158fc048a36b22`, verified
`--help` registers the flag), then checksummed **every script and module the chain
invokes** against local:

| status | file |
|---|---|
| ok | 01b_flag_metadata, 03_combine_bins, 03a_fetch_ogt, 03b_merge_ogt, 03c_merge_mags |
| ok | 04_select_genomes, 05_aggregate_signalp, 06_assemble_dataset |
| ok | selection.py, dataset.py, binning.py, gtdb.py |

**0 stale, 0 missing** after the fix. This check should run before every chain
submission — it costs one command and catches exactly this class of failure.

**Added `START_AT` resume support**, because A0 (00:00:25) and A (00:07:19) had both
COMPLETED with outputs on disk and reruns were about to repeat them:

```
START_AT=B bash scripts/slurm/14_assemble_chain.sh
```

Valid values `A0` (default, full chain) `| A | B | C | D | F`. A skipped stage yields no
job id, so dependents route through a `dep()` helper that emits
`--dependency=afterok:<id>` only for a non-empty id — the first submitted stage
therefore starts unconstrained rather than depending on a blank. Skip logic tested for
all six start points before use, and `dep()` verified to emit the flag for `12345` and
nothing for `""`.

**Resubmitted `START_AT=B`: `1164631` -> `1164632` -> {`1164633`, `1164634`} ->
`1164635`.** A0/A correctly skipped and their outputs reused.

**Both fixes confirmed live in B's log:**

```
[selection] require_col=has_proteome: dropping 3,754 of 905,425 genomes explicitly
flagged as having no proteome (nulls retained as unannotated)
[04] extremophiles: 1,555 | outgroups: 1,112 | pairs matched: 1,112 (unmatched 443)
```

The null-retention clause is the load-bearing part: 901,341 GTDB rows have
`has_proteome = None`, and a naive gate would have dropped every one of them. Only
**3,754 explicitly-False** genomes are excluded.

### Chain B->F on gpu_h200: stage timings, and stage F needs 128 GiB

Resumed run `START_AT=B`, all on `gpu_h200`:

| job | stage | state | elapsed |
|---|---|---|--:|
| 1164631 | B genome pairs | COMPLETED | 00:01:29 |
| 1164632 | C secreted table | COMPLETED | 00:43:25 |
| 1164633 | D cluster 50% | COMPLETED | 00:09:17 |
| 1164634 | E cluster 40% | COMPLETED | 00:05:51 |
| 1164635 | F assemble | RUNNING | — |

**The scope reduction landed as designed.** C reports:

```
WHOLE_SCOPE_TIER hyperthermophile ('high','medium') 578
WHOLE_SCOPE_TIER psychrophile     ('high','medium') 439
WHOLE_SCOPE_PAIR_MEMBERS_ADDED                      547
WHOLE_SCOPE_GENOMES 1564
[05agg] wrote secreted FASTA 2,057,964 seqs | whole FASTA 3,293,163 seqs
```

Whole FASTA **10,874,729 -> 3,293,163 sequences (-69.7%)**, from 7,320 -> 1,564
whole-scope genomes. Stage E's 40% clustering fell to **05:51** (it was 21:28 at 10.9 M
sequences), and D 50% to 09:17. Note the 1,564 vs the dry run's 1,566: B reran with the
gate and produced a marginally different pair set, so 547 pair members were unioned
rather than 549. Consistent, not contradictory.

**29 genomes had no proteome file** — diagnosed (job 1164652) rather than assumed:
all 29 are `CU_` MAGs, all `has_proteome == False`, all non-representative, and
**0 are pair members**. So the `has_proteome` gate is doing exactly its job: void
genomes reach the corpus FASTA loop but never a matched pair. These 29 are absences by
construction, not losses.

**Stage F's real memory footprint is 128 GiB, not the ~18 GB I estimated.** Measured
live via `srun --overlap` at 17 min: RSS **134,742,960 kB = 128.5 GiB**, CPU time
00:17:23 against 17:22 elapsed (100% utilisation, so it is streaming, not thrashing).
My earlier estimate of ~18 GB for 119.7 M rows x 7 columns was low by ~7x -- pandas
object-dtype string columns carry far more per-cell overhead than the on-disk bytes
suggest.

**Peak is higher still: 210.8 GiB at 46 min** (later sample), so the footprint is
~12x my estimate, not 7x. The RSS trace also locates the phase boundary:

| t (min) | RSS (GiB) |
|--:|--:|
| 17.1 | 128.5 |
| 45.9 | **210.8** (peak) |
| 47.4 | 204.7 (-4.1 GiB/min) |

RSS peaking at ~46 min and then declining marks the end of the 36 GB read -- which lands
exactly where the probe predicted (36.2 GB / 12.6 MB/s = 48 min) -- and the start of the
compute phase, where intermediates are released.

Consequences worth keeping:
* On `standard` (258 GB nominal, shared, and NOT reserved -- see below) a 211 GiB peak is
  at or over the line. F is safe only on `memory` (677 GB) or `gpu_h200` (2 TB).
* biotite runs `SelectTypeParameters=CR_CPU`, so `--mem` does **not** reserve RAM and
  does not gate scheduling. On a shared node that is a real hazard; on this 2 TB node it
  is not.

**Do not read `FreeMem` as headroom -- I did, and it overstates the risk.** On
`node-224-2t-8gpu-1` while F held 211 GiB:

```
RealMemory 2,063,701 MB (2 TB) | AllocMem 0 (CR_CPU) | FreeMem 129,353 MB
MemAvailable 1,499 GiB | Cached 1,337 GiB
```

`FreeMem` excludes 1,337 GiB of **page cache**, which is reclaimable on demand and is
largely the 36 GB table and cluster maps F had just streamed. The figure that matters is
`MemAvailable = 1,499 GiB`. F's 211 GiB peak against ~1.5 TB available is not an OOM
risk on this node; quoting FreeMem made it look like one.
* F re-reads the full 36 GB `secreted_all.tsv` (12.6 MB/s measured => ~48 min of
  parsing before work begins). The secretome table did **not** shrink -- only the
  whole-proteome FASTA did -- so this cost is unchanged by the scope reduction.

Stage F's sbatch wrap does not use `python -u`, so its log stays empty until exit; a
silent F is expected, not a hang. Verified progress by process inspection instead.

### Observability fix: unbuffered stages + phase markers in 06

Stage F (`1164635`) ran **58+ minutes with a zero-byte log**, so I had to infer its phase
from RSS growth and CPU time via `srun --overlap` rather than read it. Two causes, both
fixed for future runs (neither affects the run in flight):

1. **`PY` is now `... /python -u`.** A single variable feeds every `$PY` call in the
   chain, so one edit makes all stages unbuffered. Without `-u`, python block-buffers
   stdout when it is a file, so progress prints accumulate until process exit and a
   running job is indistinguishable from a hung one.
2. **Phase markers in `06_assemble_dataset.py`.** `_phase()` prints
   `[06] t+MM:SS <msg>` with wall-clock since process start, at the three boundaries that
   carry the cost: inputs loaded, `assemble_dataset` done (with row count), parquet
   written. A log tail alone now locates the phase.

Stage 06's three costs, for reference: the 36 GB `secreted_all.tsv` read (~48 min
measured at 12.6 MB/s), the cluster-map union-find merge (258 MB + 163 MB -> 195,237
groups), and pair derivation + leakage-aware splitting. Only the first was ever
measurable from outside.

Verified: marker format renders as `[06] t+03:07 ...`, `bash -n` clean, 139 passed /
1 skipped.

**F's trace, for the record.** RSS 128.5 GiB at 17 min -> **210.8 GiB peak at 46 min**
-> 204.7 and falling at 47 min; CPU time tracked elapsed exactly (57:48 / 57:47) so it
was compute-bound throughout, never swapping. The peak at ~46 min coincides with the
predicted end of the 36 GB read. The previous attempt was cancelled at 1:06:49 having
written nothing, which bounds the post-read phase from below (>20 min) but not above --
`to_parquet` is a single terminal write, so an absent output file says nothing about
progress within the phase.

### Stage F RSS trace — correct units, and where the cost actually is

`ps -o rss` reports **kilobytes**, so GiB = KB / 1024^2. I twice divided by 1e6 and
labelled the result GiB, inflating every figure by 4.9%. Corrected trace for
`1164635`:

| t (min) | rss (KB) | GiB |
|--:|--:|--:|
| 17.1 | 134,742,960 | 128.5 |
| 45.9 | 221,024,256 | 210.8 |
| 95.0 | 280,902,656 | 267.9 |
| 96.2 | 283,988,992 | 270.8 |
| 97.5 | 287,049,728 | 273.8 |
| 104.0 | 295,504,624 | **281.8** |

Per-interval growth over the last four samples: **2.45, 2.25, 1.24 GiB/min** (mean
1.98, and decelerating). Projections from the mean rate: **551 GiB at 4 h**, 789 GiB at
the 6 h wall. My earlier "~700 GiB by 4 h" did not follow from my own stated rate --
2.3 GiB/min from 281.8 GiB gives ~595 GiB, not 700. Conclusion unchanged: on a 2 TB node
with MemAvailable ~1,421 GiB there is no memory risk.

**The cost is MultiIndex `.loc`, not dict accumulation.** I blamed the per-pair dict
append in `_derive_protein_pairs` and proposed a parallel-lists rewrite. Benchmarked at
realistic scale (1.2 M pairs, job 1164764) that hypothesis fails:

| accumulation | build + frame | peak accumulator |
|---|--:|--:|
| dicts (current) | 5.9 s | 0.31 GiB |
| parallel lists | 2.6 s | 0.06 GiB |
| speedup | **2.24x** | 4.7x |

5.9 s and 0.31 GiB for the full projected output cannot explain hours of runtime or
hundreds of GiB. The real term (job 1164765, 4 M-entry index):

| operation | unsorted | sorted | dict-of-dicts |
|---|--:|--:|--:|
| partial `best.loc[e]` | 3.61 ms | 0.94 ms | **0.4 us** |
| scalar `best.loc[(e,cl)]` | 420 us | 49.2 us | — |

`best` comes from `groupby([...]).first()` and is **not lexsorted**
(`is_monotonic_increasing = False`), so each lookup degrades to a scan. The loop does 2
partial lookups per genome pair and **2 scalar lookups per emitted pair**: ~1.2 M pairs
x 2 x 420 us = **~17 min per million pairs**, which is the dominant cost and scales with
output size.

Measured fixes: **`sort_index()` -> 8.6x** on the scalar path (one line);
**dict-of-dicts -> ~9,000x** on partials for a 6 s one-time build. The parallel-lists
change is worth doing but is the 2.2x, not the fix.

**Parallel-F safety, checked before proposing it:** stage F's only writes are
`--out $W/labeled_dataset.parquet` and `--fig $W/dataset_splits.png`. All five inputs
(`secreted_all.tsv`, `combined_labels.parquet`, `all_pairs.tsv`, `clu50_cluster.tsv`,
`clu40_cluster.tsv`) are read-only and concurrent readers are safe, so an optimised F
writing `*_v2` paths cannot corrupt the incumbent.

Recurring failure mode, third instance this session: attributing cost to the code I had
just read instead of the code I had measured (the 8-min dry run's quadratic dict rebuild,
then this). Profile first.

### Optimised pair derivation, validated byte-identical, running as a parallel F

Applied both measured fixes to `_derive_protein_pairs`:

1. **Dict-of-dicts replaces the MultiIndex Series.** `groupby(["_bare", gc]).first()`
   returns a Series whose index is not lexsorted, so every `.loc` degrades to a scan.
   Replaced with `{bare: {cluster: tagged_id}}`, built once. Measured on a 4 M-entry
   index: partial `.loc` **3.61 ms -> 0.4 us (~9,000x)**, and `sort_index()` alone would
   have given 8.6x on the scalar path.
2. **Parallel per-column lists replace one dict per row** (2.24x, 4.7x less peak
   accumulator memory) and the cluster intersection now iterates the smaller map and
   probes the larger, `O(min)` instead of building two key sets.

**Equivalence verified, not assumed** (job 1164778). Reference implementation = the
original `.loc` code verbatim, run against a synthetic table with paralogs, an unmatched
pair, an absent genome, and both scopes active:

```
old rows 179 | new rows 179
IDENTICAL: True
time old 0.20s | new 0.09s | speedup 2.2x
cap k=2 rows 179 | deterministic True
```

Byte-identical output on all 8 columns after sorting, cap path intact and reproducible.
139 passed / 1 skipped. (The 2.2x here is the accumulator effect only -- this table is
too small for the lookup fix to show, since its cost scales with pairs emitted.)

**Running as a parallel job, safely isolated.** Verified before submitting that stage F
writes only `--out` and `--fig`; all five inputs are read-only and concurrent readers are
safe.

| | incumbent 1164635 | optimised 1164779 |
|---|---|---|
| tree | `repo/` (dataset.py md5 `3fd4b118`) | `repo_v2/` (md5 `4baa0cb0`) |
| out | `labeled_dataset.parquet` | `labeled_dataset_v2.parquet` |
| fig | `dataset_splits.png` | `dataset_splits_v2.png` |
| marker | `.F_done` | `.F_v2_done` |
| mem | 256G | 400G |

Separate source trees, so editing the module could not disturb the 2 h+ incumbent --
confirmed `repo/src/eptrans/dataset.py` still hashes to `3fd4b118` after staging v2.
Whichever finishes first wins; the other is insurance. Incumbent was at **02:12:17** when
v2 was submitted.

### Correction: the RSS/bottleneck commit is `9eb8066`, not `9febf3b`

I cited `9febf3b` for the corrected-units + bottleneck work. That hash is the
**pre-amend** commit and is no longer reachable from HEAD. Sequence:

1. Committed with `git commit -m "...backticks around a variable name..."`. The message
   was in double quotes, so **bash executed the backticks as command substitution**
   (`best: command not found` on stderr) and silently deleted that clause from the
   message body — line 14 read `The real cost (1164765):  from groupby().first() ...`.
2. Amended with `git commit --amend -F -` and a quoted heredoc, which restored the text
   **and produced a new commit object**: `9eb8066`.
3. I then quoted the old hash from memory instead of re-reading it. `9febf3b` is orphaned
   — checking it out shows the corrupted message.

Verified: `9eb8066` line 14 now reads `the "best" Series from groupby(...).first() is not
lexsorted`, and `9febf3b` is confirmed NOT an ancestor of HEAD. The wrong hash never
reached a tracked file (`grep` across md/py/sh/yaml: no hits), and all twelve other SHAs
cited this session — 2f21af6, c0167e3, c3a9439, 7397b46, dcfa9f4, 22ebdb4, 39db2d5,
8a652f1, f1e8c76, a22da94, 3a93f01, f5afb40 — are real and reachable.

Two habits this argues for:
* **Never put backticks in a `-m` message.** Use `-F -` with a quoted heredoc
  (`<<'MSG'`), which is literal, or avoid backticks entirely. The failure is silent apart
  from one stderr line, and it mutates durable history.
* **Any `--amend` invalidates the hash.** Re-read `git log -1 --format=%h` after
  amending; never carry the pre-amend SHA forward.

### Run-1 model artifacts archived to `*_old` before the retrain

**They were on a collision course.** All four training/inference stages write to fixed
paths with **no run identifier**:

| stage | output path |
|---|---|
| 08 MLM | `$PERSIST/models/mlm_adapt` |
| 08 classifier | `$PERSIST/models/clf_${PHENO}` |
| 09 embed | `$PERSIST/embeddings/secretome_r232` |
| 10 cached probe | `$PERSIST/models/cached_probes` |

So re-running stage 08 would have overwritten the run-1 adapter in place -- **~10.7 h of
H200 time**, with the embedding cache (~3 h) invalidated behind it.

**Also settles an open question from earlier in the session.** I had only been able to
establish at *directory* level that the cached heads persisted rather than landing in
swept scratch, and explicitly declined to assert the files were intact. They are:
**all five `head_best.pt`, 5,249,120 B each.**

Inventory: `mlm_adapter_best/adapter_model.safetensors` 94,413,200 B; five
`clf_*/clf_ckpt.pt` at ~299,214,922 B each; five cached-probe heads; eight embedding
shards totalling 9.6 G. **69 files, 12,249,122,328 bytes.**

**Moved, not copied.** `mv` within one filesystem is a rename -- instant and no extra
space. Verified after the move: same 69 files, same 12,249,122,328 bytes,
`adapter_model.safetensors` md5 `0771ed96a8a32744e19daeac2ab27c79` and
`clf_thermophile_cached/head_best.pt` md5 `8fcfad0aabd0dd6d0aede8416ba7e329`, both
unchanged. Checked first that no running job reads these paths (stage F touches only
`assemble/`).

An earlier `cp -al` hardlink snapshot (`models_run1_r232_20260711`) was removed once
`stat` confirmed it shared inodes with `*_old` -- link count 2 before removal, 1 after,
byte total and checksum unchanged. It was a second name for the same data, not a second
copy.

`ARCHIVE_README.md` written into both directories recording the inventory, the checksums,
the provenance (trained 2026-07-11/12 on the **secreted-only** dataset, before the MAG
ingest, per-phenotype scope, measured-OGT rubric, `has_proteome` gate, `--max-per-sample`,
and the low-tier exclusions), and the restore command.

Two things the README makes explicit:
* **Run-1 weights are not comparable head-to-head with the retrain.** Label definitions,
  genome set and protein scope all changed. They are a fallback and a reference for the
  reported run-1 numbers (thermophile AUPRC 0.862 on 1,495 val pairs, pair-level AUC
  0.905), not a baseline.
* The **MLM->classifier key-remap trap** still applies to this adapter: stage 08 trains
  `EsmForMaskedLM`, the classifier is a bare `EsmModel`, and `peft.load_adapter` silently
  drops mismatched keys. Anything loading it must go through
  `load_mlm_adapter_into_classifier` (288 tensors, 0 unmatched) or it reads zeros.

**Worth fixing properly:** the fixed output paths are the underlying defect. Stage 08/09/10
should take a run tag (or refuse to start when the target exists and no `--overwrite` is
given), so the next retrain cannot depend on someone remembering to archive first.

### F_v2 has no progress output — the markers were never deployed

Asked whether stage F_v2 reports progress. It does not, and the reason is mine: the
`_phase()` markers committed in `c0167e3` existed **only in the local tree**. On the
cluster both `repo/` and `repo_v2/` had `_phase(` count **0**. So v2 ran under `python -u`
(which was deployed, via the chain) against a script with nothing to emit until its first
`print` at line 102 — which sits *after* the 36 GB read. Blind for the same reason as the
incumbent, by a different route.

**Second partial-deployment failure this session**, after the stale `04_select_genomes.py`
that killed stage B. Same shape: edited locally, transferred some files, assumed the rest.

**A measurement error inside the diagnosis, worth recording.** My first
`srun ... pgrep -f 06_assemble_dataset | head -1` returned **the wrong process** — both F
jobs run on the same node, and `head -1` picked the older one, so v2's "progress" was
actually the incumbent's. Fixed by mapping PID to job through
`/proc/<pid>/cgroup`:

| job | pid | elapsed | CPU | RSS |
|---|--:|--:|--:|--:|
| 1164635 incumbent | 2196564 | 148.6 min | 02:28:35 | 223.8 GiB |
| **1164779 v2** | 2277830 | **16.3 min** | 00:15:57 | **80.9 GiB** |

At **matched wall clock** (v2 at 16.3 min vs the incumbent's own 16.5-min sample) v2 holds
**80.9 GiB against 128.5 GiB — 1.59x less memory**. Not yet a time claim: neither had
reached the phase where the lookup fix dominates.

Incidental: the incumbent's RSS **fell** from its 281.8 GiB peak (t=104 min) to 223.8 GiB
(t=148.6 min), so it released the input table and is into derivation — the phase the
optimisation targets.

**Deployed the marker version to both trees** (md5 `defde8fbe20063184597b709f368adda`,
4 markers, compiles). No effect on the running jobs, which imported their module at start.

**Added `scripts/check_deployed.sh`** so this cannot recur silently: it md5s the 13
scripts and modules the chain invokes against the cluster copy and exits non-zero on any
mismatch. Run before every submission. On its first run it immediately found two real
divergences:

| file | local | remote | cause |
|---|---|---|---|
| `14_assemble_chain.sh` | `17a40f74` | `7a2392de` | the `-u` change (`c0167e3`) not deployed |
| `src/eptrans/dataset.py` | `4baa0cb0` | `3fd4b118` | **intentional** — `repo/` holds the unoptimised module the incumbent is running |

The chain has now been deployed (`17a40f74`, syntax checked). The `dataset.py` divergence
stays until the incumbent finishes: `repo/` must keep the module its running job started
with, while `repo_v2/` carries the optimisation. Anyone running the check mid-experiment
should expect that one line and no others.

### Code defect behind the RSS unit error: a hardcoded number in an f-string

The unit error corrected in `9eb8066` had a specific cause worth recording, because the
cell that reported it **contradicted itself** and I did not say so.

That cell converted the trace correctly for the table (`kb/1048576`) but the trailing
note interpolated a **literal string** for the peak:

```python
print(f"NOTE incumbent RSS FELL from 295.5->{inc['rss_kb']/1048576:.0f} GiB ...")
#                                    ^^^^^ hardcoded, never converted
```

So the same `print` mixed a correctly-converted current value with a hand-typed peak that
was `295504624/1e6` — the decimal-million shortcut — labelled GiB. Its output read
`295.5->224 GiB` while the prose alongside it said 281.8 GiB. Both cannot be right.

Recomputed with every figure derived from the raw KB, no literals:

| t (min) | rss (KB) | GiB |
|--:|--:|--:|
| 17.1 | 134,742,960 | 128.5 |
| 45.9 | 221,024,256 | 210.8 |
| 95.0 | 280,902,656 | 267.9 |
| 96.2 | 283,988,992 | 270.8 |
| 97.5 | 287,049,728 | 273.8 |
| 104.0 | 295,504,624 | **281.8 (peak)** |
| 148.6 | 234,667,572 | 223.8 |

`295504624/1024^2 = 281.8`; `/1e6 = 295.5`, a **4.9% inflation**. Released since peak:
**58.0 GiB**. The reported conclusion (281.8) was the correct one and no tracked file ever
contained 295.5 — verified by grep across md/py/sh — so nothing durable needs changing.

The lesson is narrower than "unit confusion": **do not hand-type a number into an
f-string that the same cell already has in a variable.** A literal cannot be checked
against the data it claims to describe, and it silently survives the correction of every
computed value around it.

### Fixed the underlying defect: run tags + clobber guards on stages 08/09/10

Archiving run 1 by hand was a workaround. The defect was that all three stages wrote to
**fixed paths with no run identifier**, so a retrain overwrote the previous run in place
— putting ~10.7 h of H200 MLM training, and the ~3 h embedding cache behind it, one
accidental resubmission away from destruction.

Two mechanisms added to `08_train_backbone.sbatch`, `09_embed_secretome.sbatch` and
`10_train_cached_probe.sbatch`:

* **`RUN_TAG`** — empty by default, so historical paths (`models/`, `embeddings/`) are
  preserved and existing tooling keeps working. `RUN_TAG=run2` redirects to
  `models_run2/` + `embeddings_run2/`.
* **`refuse_clobber`** — a non-empty target with `OVERWRITE != 1` is a **hard stop before
  any GPU time is spent**, printing the three ways forward (set a RUN_TAG, archive, or set
  OVERWRITE=1).

**Verified by extracting the guard from the real script and exercising all five paths:**

| case | result |
|---|---|
| empty target | proceeds (rc=0) |
| existing artifacts, no OVERWRITE | **refuses, rc=1, before GPU work** |
| `RUN_TAG=run2` with run1 present | proceeds into the parallel tree |
| `OVERWRITE=1` | proceeds with a warning |
| classifier mode | guards `clf_$PHENO`, not the adapter |

Placement matters more than syntax here: the guard fires *before* the python invocation,
so a mistake costs one second rather than one wasted allocation.

One bug found and fixed by that test: the error text printed `$d` literally because it sat
in an escaped double-quoted string. It now emits a copy-pasteable command with the real
path — `mv '/…/models/mlm_adapt' '/…/models/mlm_adapt_old'`.

Not yet deployed to the cluster: stage F holds `repo/`, and `check_deployed.sh` will flag
these three as STALE until the training stages are next staged. That is the check working
as intended.


---

## Composition-only baseline vs ESM-embedding classifier — how much signal is holistic?

**Motivation.** Extremophile phenotype classifiers are known to lean on bulk amino-acid
composition (charged-residue fraction, etc.). Before trusting the fine-tuned ESM
classifier as a re-ranker in the MPNN-in-the-loop generator — where a per-residue
`bias_AA` already moves composition — we need to know whether the classifier adds
*holistic* signal beyond composition, or is effectively a composition detector (which
would create a self-reinforcing loop: bias moves composition → classifier rewards it).

**Design (matched comparison).** Same rows, same split, same classifier family — only the
feature set differs, so ΔAUC is purely the holistic-embedding contribution:
- Substrate: the **full r232 secretome**, 1,985,058 proteins matched across the cached
  embeddings, the mature-chain FASTA, and `labeled_dataset_r232_clustered.parquet`.
- Split: the parquet's own `split` column (train 1,590,361 / val 198,277 / test 196,420).
  Evaluated on **val** (the ESM cached-probe headline AUPRCs were val-selected) and **test**.
- **Composition-LR**: logistic regression on the 20-d AA-frequency vector.
- **ESM-LR**: logistic regression on the 2560-d MLM-adapted masked mean-pool embedding
  (the same frozen features the cached-probe heads were trained on).
- Both: `class_weight="balanced"`, `C=1.0`, `StandardScaler`, all negatives.
- Job `0692175b` (biotite `memory` partition, 16 CPU / 96 GB; data-load ~21 min, fits ~18 min).

**Results (matched estimator — LR — plus the production nonlinear head for context).**

Three classifiers, all on val AUPRC (the ESM-MLP column reports val only — the production
head was model-selected and reported on val). The **comp-LR vs ESM-LR** columns are the
matched comparison (same linear estimator, only features differ → clean ΔAUPRC). The
**ESM-MLP** column is the production cached-probe head (512-hidden MLP + matched-pair margin
loss + best-epoch selection) — the same numbers as the `run1_classifier_performance.png` /
`run1_auc_table.csv` AUC plot from today.

| phenotype | split | prev. | comp-LR AUPRC | ESM-LR AUPRC | **ESM-MLP AUPRC** (prod.) | ΔAUPRC (LR) | comp-LR ROC-AUC | ESM-LR ROC-AUC | ΔAUC (LR) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hyperthermophile | val | 0.006 | 0.065 | 0.332 | **0.898** | +0.267 | 0.899 | 0.975 | +0.076 |
| hyperthermophile | test | 0.006 | 0.066 | 0.323 | — | +0.258 | 0.896 | 0.975 | +0.079 |
| thermophile | val | 0.130 | 0.308 | 0.732 | **0.862** | +0.424 | 0.749 | 0.917 | +0.168 |
| thermophile | test | 0.132 | 0.305 | 0.727 | — | +0.421 | 0.744 | 0.914 | +0.170 |
| halophile | val | 0.350 | 0.598 | 0.763 | **0.818** | +0.165 | 0.745 | 0.855 | +0.110 |
| halophile | test | 0.349 | 0.597 | 0.759 | — | +0.162 | 0.745 | 0.852 | +0.108 |
| acidophile | val | 0.045 | 0.221 | 0.627 | **0.745** | +0.407 | 0.861 | 0.952 | +0.091 |
| acidophile | test | 0.045 | 0.219 | 0.628 | — | +0.409 | 0.860 | 0.953 | +0.094 |
| alkaliphile | val | 0.038 | 0.133 | 0.319 | **0.674** | +0.186 | 0.774 | 0.886 | +0.111 |
| alkaliphile | test | 0.037 | 0.132 | 0.330 | — | +0.198 | 0.774 | 0.891 | +0.117 |

comp-LR/ESM-LR: val ≈ test throughout (≤0.01 drift) — stable, no overfitting.

**Why ESM-LR (0.33/0.73/0.76/0.63/0.32) differs from the AUC-plot ESM-MLP
(0.90/0.86/0.82/0.75/0.67).** They are two different classifiers on the *same* embeddings.
ESM-LR is a plain **linear** logistic regression, single fit, weighted log-loss only — used
here *on purpose* so that comp-LR vs ESM-LR holds the estimator fixed and varies only the
feature set (otherwise "better features" and "better head" would confound). ESM-MLP is the
production head: **nonlinear** (512-hidden MLP), trained with an added **matched-pair margin
loss**, over 30 epochs with **best-epoch selection on val**. The gap is largest for the
tiny-positive thermal/pH phenotypes (hyperthermophile 0.33→0.90, alkaliphile 0.32→0.67),
where nonlinearity + the pair loss help most, and smallest for **halophile** (0.76→0.82),
whose signal is largely linear/compositional — a consistent, expected pattern.

**Floor/ceiling framing (the key point).** The matched LR-vs-LR ΔAUPRC is a **conservative
floor** on the embedding's advantage over composition: an MLP on a 20-d composition vector
cannot manufacture signal absent from the features, while an MLP on embeddings extracts
*more* than LR does. So the true "holistic signal beyond composition" is **at least** the
LR deltas shown and almost certainly larger — the ESM-MLP AUPRCs are the **ceiling** the
embedding reaches once you stop handicapping it with a linear head. The two ESM columns are
therefore not contradictory: ESM-LR is the floor, ESM-MLP is the ceiling, and both sit far
above composition-only.

**Metric note.** ROC-AUC here is standard one-vs-rest over the whole val/test set. It is
**not** the cached-probe **pair-AUC** (0.924 hyperthermophile, etc.), which is computed only
on taxonomy-matched ortholog pairs and is the anti-taxonomy control. The two are different
quantities and must not be compared across tables; the valid comparisons above are
within-metric (comp-LR AUPRC vs ESM-LR AUPRC; comp-LR ROC-AUC vs ESM-LR ROC-AUC).

**Reads.**
1. **The ESM embedding carries the majority of the discriminative signal; composition
   alone is weak.** On AUPRC (the metric that matters at these prevalences), the embedding
   roughly doubles-to-quadruples composition-only: thermophile 0.308→0.732, acidophile
   0.221→0.627, hyperthermophile 0.065→0.332 (≈5×). The classifier is **not** a mere
   composition detector.
2. **Halophile is the exception that proves the rule.** It is the one phenotype where
   composition-only is already decent (AUPRC 0.598) — consistent with haloadaptation being
   genuinely a bulk-composition phenomenon (surface acidic enrichment, matching the laccase
   structural analysis). Even there the embedding still adds +0.165.
3. **The embedding's edge is largest where the biology is subtlest** — the thermal and pH
   phenotypes, where adaptation is contextual/positional rather than a simple composition
   shift. This is exactly the "holistic" signal the concern was about, and it is real.

**Implication for MPNN-in-the-loop scoring.**
- The classifier re-ranker adds substantial signal beyond `bias_AA`'s composition nudge —
  so it is worth keeping as an oracle, *not* redundant with the bias.
- **But** because `bias_AA` moves composition and the classifier partly rewards it (halophile
  especially, where composition-only already reaches 0.598, and recall the clf-vs-mutation-count
  ρ=+0.54), a residual self-reinforcement risk remains on the halophile track. Mitigation
  (already in the spec): weight the composition-orthogonal terms — extremophilic pseudo-LL and
  the coupling-consistency oracle — comparably to the classifier, and consider orthogonalizing
  the composition component out of the halophile classifier score before re-ranking.

Artifacts: `composition_vs_embedding_auc.csv` (this three-way table: comp-LR, ESM-LR, ESM-MLP),
job `0692175b` output `matched_comparison.json`. The ESM-MLP (production) column is sourced
from `run1_auc_table.csv` / `run1_classifier_performance.png` (cached-probe heads). A
composition-only baseline restricted to the ordinary val/test split (job `9ef578a4`, no
embedding) reproduced the composition numbers on the 420k MLM subsample as a cross-check.

---

## 2026-08-06 — Scope defect chain: INV-SCOPE-D → INV-EMIT-A

Traced why the psychrophile head reached only val AUPRC 0.572 / pair-AUC 0.635 while
thermophile reached 0.909 / 0.931. The proximate hypothesis was biological (cold
adaptation is local/structural, mean pooling dilutes it). The actual first-order cause
turned out to be a data defect: **the psychrophile arm never saw whole proteomes at all.**

### Defect 1 — INV-SCOPE-D (commit `f811de5`)

`_apply_corpus_scope` decided extremophile scope from each genome's **single corpus
label** (`lab.isin(whole_classes)`) while the mesophile branch keyed on **pair
membership**. Polyextremophiles (cold+saline deep-sea, cold+alkaline soda lake) carry one
label but serve several classes, so every whole-scope pair-extremophile was silently
reduced to its secretome.

Measured before the fix: psychrophile EXT 19 genomes / 4,888 rows / frac_secreted
**1.000**, labelled {halophile 2776, alkaliphile 2100, acidophile 12}; **zero of the 1,286
psychrophile-labelled genomes form ext pairs**. hyperthermophile EXT 27 / 3,677 / 1.000.
The corpus as a whole was correctly scoped (psychrophile label holds 4,158,435 proteins) —
the defect was confined to pair-forming genomes.

Fix mirrors the mesophile union rule onto the extremophile side; accepts both `ext_acc`
(pairs output) and `extremophile_acc` (pairs input, the real schema) since a bare guard
would have made the union a silent no-op. 145 tests + 2 regressions.

Follow-on: INV-SCOPE-A then asserted "no whole-scope class genome has non-secreted
proteins" — true only under the bug. Exempted pair-serving extremophiles the same way
INV-SCOPE-C already exempts whole-scope outgroups (commit `9bb3b5e`, 146 tests).

Stage F re-run (job `1165537`, COMPLETED 29:13): corpus 18,064,818 → **18,080,119
(+15,301)**, protein pairs 66,759 → **66,764 (+5, all hyperthermophile)**.

### The +5 was the tell — my prediction of substantial pair growth was wrong

Diagnosis required discarding three explanations, each refuted by measurement:

| Hypothesis | Refuted by |
|---|---|
| Novel MAG proteins cluster alone | The genomes are GTDB (`GB_` 198,268 / `RS_` 245), not MAGs |
| Missing from `whole_scope_accessions.txt` | All present (`list=1`) |
| Absent from `clu40_cluster.tsv` | Present — 1,315 lines for `GCA_002167555.2` |

The decisive number: psychrophile ext non-secreted proteins land in **161,020 clusters,
100.0% singletons**. A 100.0% rate cannot be a biological gradient — core housekeeping
genes (ribosomal proteins, EF-Tu, GroEL, RNAP subunits) are >40% identical across phyla
and must cluster with any mesophile outgroup. This was the user's objection and it was
correct; it forced the search to plumbing.

### Defect 2 — INV-EMIT-A (commit `950ab49`), the actual root cause

`05_aggregate_signalp.py` has two emission paths with **different scope rules**:

* **FASTA** — iterates `set(df.genome) | whole`, i.e. every whole-scope genome's full
  proteome regardless of SignalP coverage (an earlier fix, for the mirror-image bug).
* **TSV** — built from SignalP prediction rows **only**. No prediction → no row, ever.

Stage 06 reads the **TSV**. So a whole-scope genome scanned in secreted mode had its
cytoplasmic proteins clustered but unreachable:

| `GB_GCA_002167555.2` | count |
|---|---|
| rows in `secreted_all.tsv` | **51** (all `is_secreted=True`) |
| records in `wholeproteome.faa` | **1,045** |
| lines in `clu40_cluster.tsv` | **1,315** |
| non-secreted corpus rows after INV-SCOPE-D | **0** |

All 19 psychrophile pair-extremophiles had 0 non-secreted rows. INV-SCOPE-D could not take
effect because the rows it would have kept **did not exist** — the scope filter is
removal-only and never fabricates a protein.

Fix: collect whole-scope rows during the existing FASTA pass (no extra I/O), concat with
SignalP rows **ordered first** so real calls win the `tagged_id` dedupe and only genuinely
unscanned proteins land as `OTHER`/non-secreted. Hard assertion fails the stage if any
whole-scope genome still has fewer TSV rows than proteome records. New test asserts
`fasta_ids <= set(tsv_ids)`. 147 tests pass.

Baseline preserved: `preemit_secreted_all.tsv`; FASTA checksums before re-run
`wholeproteome.faa` `877aa811c6f77df8ad8c60061437841c`, `secretome.faa`
`0eee629ec2182cb1c59e78795bc20acd` (if unchanged, clustering stages D/E can be skipped).

### Outgroup matching: exhaustion, not phylum rarity

Question raised: are unmatched extremophiles from obscure phyla? Measured — **no**.

* Matching **does** reach phylum: `selection.py:142` defaults to
  `[genus, family, order, class, phylum]` while `config.yaml` lists only
  `[genus, family, order, class]`. **Config/code disagreement, unreconciled.**
  Realised `matched_rank`: family 1,775 · order 1,222 · class 939 · **phylum 781** ·
  genus 743 · unmatched 1,029.
* In **all 98** (class, phylum) groups holding unmatched extremophiles, `matched` equals
  `supply` **exactly** — demand > supply in every case, zero anomalies. Thermoproteota
  (516 genomes in the selection set, 4th-largest): 243 thermophiles demanded, 147
  mesophiles available, 147 matched, 96 unmatched. Halobacteriota: 120 / 102 / 102 / 18.
* Mechanism: `find_outgroup(erow, pool, used_this_class)` + `used_this_class.add(oidx)` —
  greedy assignment **without replacement**, so a phylum's capacity is capped at its
  confident-mesophile count. 3,728 distinct outgroups serve 5,460 matched pairs (reuse
  happens across classes, never within one).
* 364 of 1,029 unmatched (35.4%) sit in phyla with **zero** confident mesophiles — not
  recoverable by any policy change. ~665 are recoverable.
* `reuse_outgroups` exists (default False). **Decision: left as-is**, so the whole-proteome
  effect stays attributable to one variable.

### `confident_mesophile` is a global three-way conjunction

`is_confident_mesophile` requires temp 20–40 **and** pH 6–8 **and** salinity ≤3, all
present and non-NaN — one global flag serving all six classes, with no per-class variant.
Among the 203,406 genomes carrying all three predictions: temp alone passes 157,793
(77.6%), pH alone 164,652 (80.9%), salinity alone 164,063 (80.7%), **all three 108,390
(53.3%)**. Failures spread evenly (28,049 temp-only · 20,787 pH-only · 20,890
salinity-only · 25,290 multi-fail), so no single axis dominates. A thermally-valid control
is rejected for a pH 5.9 prediction. Temp-only would give **1.46×** the pool — directly
relevant to the exhaustion above.

Coverage caveat: only **22.5%** of the 905,425 genomes have predictions at all, because
GenomeSPOT was run on 199,923 (representatives + deep-sea MAGs), not all of GTDB. This is
coverage, **not** a confidence filter: `03_combine_bins.py:105` uses a pure presence test
and the shipped `*_optimum_error` / `*_optimum_warning` columns are **never read** — an
unused lever, notable given the 2-unit-wide pH window and GenomeSPOT's shrinkage
(slope 0.846).

### Consequences

* Psychrophile pair-AUC 0.635 was measured on **secreted proteins of halophile/
  alkaliphile-labelled genomes** — the whole-proteome hypothesis is still untested.
* On rebuild, psychrophile and hyperthermophile λ-sweep numbers become **stale**;
  thermophile, halophile, acidophile, alkaliphile are secreted-scope and unaffected.
* Stage C re-running as job `1165625` (gpu_h200, 200G, no wall cap).

### Stage-2 scaffolding written and deployed (all 10 copies md5-verified)

* `09b_embed_perresidue.py` — top-k residue cache (k=32) **plus** the identical masked
  mean, so mean/attention/MIL arms share one forward pass and any delta is attributable to
  the operator. Storage forced the design: all-residue fp16 at 2560-d would be ~30 TB for
  the corpus vs ~22 GB for the psychrophile-relevant top-32 subset. Residue selection is
  label-free (`norm`, with `stride` as a null control) since no active-site annotations
  exist and any label-dependent rule would leak.
* `pooling.py` — mean / gated-attention / top-k-MIL. Verified on biotite that
  zero-initialised attention reproduces masked-mean weights exactly, so attention
  **strictly subsumes** mean and a loss would indicate optimisation, not a wrong
  hypothesis.
* `10b_train_pooling_ablation.py` — crossover over psychrophile **and** thermophile
  (locality predicted to help only the former), identical loss/weights/seeds; dumps
  attention α at the best epoch as the interpretable artefact.
* `16_ec_constrain.py` — KOfam route. Swiss-Prot ruled out **by measurement**: 0 of
  572,970 FASTA headers carry `EC=` (EC lives in `.dat`, absent here). `ko_list` has
  **10,736 KOs with `[EC:...]`**; `kofam.all.hmm` pressed and `exec_annotation` present.

### Process note

Four wrong claims this session, each stated before being computed: predicted pair growth
that didn't happen; three successive root causes refuted by the next measurement; and
"all deployed and md5-verified" when 2 of 5 files had actually been hash-compared. The
working correction each time came from either the user's biological objection or an
explicit check — not from further reasoning over the same unverified premises.

### INV-EMIT-A verified in production (job `1165625`, COMPLETED 1:08:53, peak RSS 287 GiB)

Three independent confirmations that the fix did what was intended:

| Check | Result |
|---|---|
| Whole-scope non-secreted rows added | **+3,184,192** (3,293,163 candidates − 108,971 that already had a SignalP call) |
| `GB_GCA_002167555.2` rows in `secreted_all.tsv` | **1,045** = 994 non-secreted + 51 secreted, matching its 1,045 FASTA records exactly (was 51) |
| FASTA checksums vs pre-run baseline | **both unchanged** — `wholeproteome.faa` `877aa811c6f77df8ad8c60061437841c`, `secretome.faa` `0eee629ec2182cb1c59e78795bc20acd` |

Corpus input **125,221,498 → 128,405,690 proteins (+3,184,192, +2.5%)**; secreted count
constant at **18,614,380**, so the secreted fraction falls 14.87% → 14.50% purely by
denominator growth. The 51 real SignalP calls on the diagnostic genome survived the
dedupe, confirming the SignalP-rows-first ordering works.

**Only 108,971 of 3,293,163 whole-scope proteins (3.3%) had ever been in the TSV** — the
scale of the defect. Because both FASTAs are byte-identical, `clu40_cluster.tsv` and
`clu50_cluster.tsv` remain valid and **clustering stages D/E were skipped** (verified by
checksum, not assumed): the chain is C→F, not C→D→E→F.

Stage F resubmitted as job `1165687` (gpu_h200, 900G, no wall cap), reusing both cluster
maps. Baseline preserved as `scopeD_labeled_dataset.parquet` /
`scopeD_labeled_dataset_protein_pairs.tsv` (the INV-SCOPE-D-only state: 18,080,119 rows,
66,764 pairs) for a like-for-like comparison.

**Correction logged:** an earlier statement of this growth cited 119,678,568 / 14.95% as
the pre-fix baseline. That is the line from a *previous* stage-F run, not job 1165625.
The correct figures are 125,221,498 / 14.87%, which reconcile exactly
(125,221,498 + 3,184,192 = 128,405,690) against the constant secreted count.

### Stage F rebuild on the corrected corpus (job `1165687`, COMPLETED 45:30)

Corpus scope **90,455,200 → 21,580,199** rows (was 18,080,119 under INV-SCOPE-D alone).
Leakage invariant holds (max splits per group = 1); all 412,925 pairs same-split.
Splits 17,262,603 / 2,160,044 / 2,157,552.

| Class | Pairs before | Pairs after | Factor | Genome pairs | Pairs / gp |
|---|---|---|---|---|---|
| **psychrophile** | 906 | **309,989** | **342×** | 19 → **329** | 942.2 |
| **hyperthermophile** | 277 | **37,355** | **135×** | 27 → **232** | 161.0 |
| thermophile | 13,136 | 13,136 | **1.0×** | 474 | 27.7 |
| halophile | 43,796 | 43,796 | **1.0×** | 965 | 45.4 |
| acidophile | 3,901 | 3,901 | **1.0×** | 253 | 15.4 |
| alkaliphile | 4,748 | 4,748 | **1.0×** | 230 | 20.6 |
| TOTAL | 66,764 | **412,925** | 6.2× | | |

**The four secreted-scope classes are byte-identical (exactly 1.0×)** — the strongest
evidence the fix is surgical: it touched only whole-scope genomes, so the thermophile
(0.909) / halophile (0.844) / acidophile (0.808) / alkaliphile (0.752) λ-sweep results
remain valid and comparable. Only psychrophile and hyperthermophile numbers are stale.

Label counts: mesophile 11,057,153 · psychrophile 5,213,973 · hyperthermophile 1,882,764 ·
halophile 1,386,214 · thermophile 1,125,420 · acidophile 585,569 · alkaliphile 329,106.

**Singleton anomaly resolved** (the finding that broke the investigation open):

| Metric | Before | After |
|---|---|---|
| psychrophile ext cluster singleton rate | **100.0%** | **72.1%** |
| hyperthermophile ext cluster singleton rate | 100.0% | 67.3% |
| psychrophile shared clusters via NON-secreted ext proteins | **0** | **122,968** |
| hyperthermophile shared clusters via NON-secreted ext proteins | 0 | 12,480 |
| the 19 originally-broken genomes: non-secreted corpus rows | 0 | 49,787 |

A ~70% singleton rate is what 40%-identity clustering should give (most proteins are
lineage-specific). 100.0% was arithmetically impossible for genomes carrying ~300 core
housekeeping genes — the objection that redirected the search from biology to plumbing.

Psychrophile at 942 pairs per genome pair vs 15–45 for the secreted classes is the
whole-proteome effect the scope decision was made for, finally realised.

### Per-class training composition, rebuilt corpus (supersedes the pre-fix table)

Recomputed from `labeled_dataset.parquet` (21,580,199 rows) and the 412,925-pair table.
Genome-quality tiers come from the corpus `label_confidence`; paired-genome tiers from the
selection files' `final_confidence` (per the earlier decision that paired quality keys off
selection, not the corpus label).

| Class | Scope | Corpus genomes | Ext. proteins | Paired genomes | Genome pairs | Paired clusters | Protein pairs |
|---|---|---|---|---|---|---|---|
| thermophile | secreted | 4,993 → 5,159 | 1.11M → 1.13M | 474 → **474** | 474 | 8,999 → **8,999** | 13,136 |
| hyperthermophile | whole | 632 → 864 | 1.12M → **1.88M** | 27 → **232** | 232 | 193 → **6,232** | 37,355 |
| psychrophile | whole | 1,286 → 1,597 | 4.16M → **5.21M** | 19 → **329** | 329 | 634 → **97,733** | 309,989 |
| halophile | secreted | 4,259 → 4,317 | 1.30M → 1.39M | 965 → **965** | 965 | 30,409 → **30,409** | 43,796 |
| acidophile | secreted | 2,727 → 2,780 | 539k → 586k | 253 → **253** | 253 | 3,119 → **3,119** | 3,901 |
| alkaliphile | secreted | 1,176 → 1,181 | 290k → 329k | 230 → **230** | 230 | 3,215 → **3,215** | 4,748 |
| TOTAL | — | 15,073 → **15,898** | 8.51M → **10.52M** | 1,968 → **2,483** | 2,483 | 46,569 → **149,707** | **412,925** |

**Paired genomes and paired clusters are byte-identical for all four secreted classes** —
the same surgical signature as the pair counts.

Quality composition of the gain (this matters for whether the 342× is real signal):

* **psychrophile paired genomes 0 high / 19 med → 10 high / 319 med.** The increase is
  medium-tier (OGT ≤ 20 °C evidence), **not** low-tier — no class has any low-tier paired
  genome, reflecting the psychrophile_low and hyperthermophile_low exclusions.
* **hyperthermophile 5 high / 22 med → 71 high / 161 med** — a large absolute gain in
  high-confidence pairs, consistent with GenomeSPOT's hot end being well calibrated.
* Corpus genome counts rose (15,073 → 15,898) because pair-serving polyextremophiles now
  contribute full proteomes, so genomes that previously had zero surviving rows appear.

Caveat: the paired-clusters column is not comparable across scopes — psychrophile's 97,733
counts clusters at 40% identity over whole proteomes, halophile's 30,409 at 50% over
secretomes.

Artifacts: `perclass_training_table.png` / `.csv` **v2**.

### Retraining submitted: job `1166233`, RUN_TAG=emitfix

Full clean re-run chosen over cache reuse. Cache-reuse accounting (measured, and it
corrects an earlier wrong statement of "90.4% needs re-embedding" — that counted corpus
rows, but stage 09's input is `corpus_all.faa`, unchanged at 5,261,647 sequences because
both FASTAs are byte-identical):

| Quantity | Count |
|---|---|
| Corpus rows | 21,580,199 |
| Rows with no sequence (never embeddable) | 16,424,069 (76.1%) |
| **Embeddable** (corpus ∩ FASTA) | **5,156,130** |
| — already cached | 2,067,342 (40%) |
| — new to embed | **3,088,788 (60%)** |

**All 412,925 pairs have both members embeddable (100.0%, every class)** — nothing in the
pair term is blocked.

New embedding work by label: mesophile 1,180,771 · psychrophile 1,010,516 ·
hyperthermophile 759,938 · halophile 65,441 · alkaliphile 37,335 · acidophile 34,787 —
concentrated exactly where the fix landed.

**Why the 40% cache was discarded rather than reused:** those vectors came from an MLM
adapter trained on an extremophile-only subsample of the *pre-fix* corpus, which contained
no psychrophile cytoplasmic proteins. Reusing them would embed a corrected corpus through
an adapter fitted to the defective one, mixing two feature distributions in one cache and
confounding the very effect being measured. ~4.5 h saved is not worth an uninterpretable
result.

Sizing: cache 26.4 GB fp16 on disk, 52.8 GB fp32 in stage 10, against `--mem=400G`.
Stage 00 (contacts) completed in 41 min; stage 08 (MLM adapter, ~10 h) is the long pole.
`models_scoped/` and `embeddings_scoped/` are untouched under their own tag, preserving a
three-way run-1 → scoped → emitfix comparison.

**Read the sweep against this expectation:** the four secreted classes should reproduce
their λ-sweep numbers closely (thermophile 0.909, halophile 0.844, acidophile 0.808,
alkaliphile 0.752) since their pairs are byte-identical — a material shift there would
indicate adapter retraining moved the shared feature space, not the class data.
Psychrophile is the actual test.

### CORRECTION: `reuse_outgroups` is ON, and outgroup concentration measured

**Correction to two earlier statements in this notebook and in session prose**: I recorded
`reuse_outgroups` as "False by user decision". That is wrong. `scripts/slurm/14_assemble_chain.sh`
line 193 passes `--reuse-outgroups` on **all six** selection calls (thermophile high-only,
then the five high+medium classes). Confirmed in the data: 1,122 outgroups serve >1 class,
max exactly 6 = the class count.

**The flag's semantics are narrower than the name suggests.** `selection.py:237`:

```python
used_this_class = set() if reuse_outgroups else used_outgroups
```

It governs **cross-class** reuse only. Within a class, `used_this_class.add(oidx)` still
excludes after each use — verified: **zero** cases of one outgroup used twice within a
single class. The reuse ceiling is therefore 6 (one per phenotype), not unbounded.

**Concentration measured at both levels:**

| Level | Units | Distinct outgroups | Mean | Top 10% share | Gini |
|---|---|---|---|---|---|
| Genome pairs | 5,460 | 3,728 | 1.46 | 23.8% | **0.245** |
| Protein pairs | 412,925 | 2,080 | 199 | **63.9%** | **0.771** |

The 0.245 → 0.771 gap is the finding, and **it is not caused by `reuse_outgroups`**. A
genome reused 6× contributes at most 6× more *genome* pairs, but its *protein* contribution
scales with proteome size and cluster density: one large well-clustered outgroup supplies
thousands of protein pairs from a single genome pairing. Concentration is driven by
proteome heterogeneity, not by the flag. Top 1% of outgroup genomes (21) supply 11.7% of
all protein pairs; the single most-used supplies 3,183 = 0.77%.

Per class (protein-pair level):

| Class | Outgroups | Top-1 share | Gini |
|---|---|---|---|
| psychrophile | 329 | 1.03% | **0.324** |
| hyperthermophile | 232 | 2.26% | **0.333** |
| halophile | 965 | 1.00% | 0.636 |
| thermophile | 474 | 1.81% | 0.622 |
| alkaliphile | 230 | 3.83% | 0.613 |
| acidophile | 253 | **7.49%** | 0.639 |

**The whole-proteome classes are the most EVENLY distributed** (Gini 0.32–0.33), because
every outgroup contributes its full proteome. Secreted classes are more concentrated
(0.61–0.64) — secretome size varies far more between genomes than proteome size does.
Worst single case: acidophile, one outgroup supplying 7.5% of that class's 3,901 pairs.

**Implication.** Turning reuse off would cost ~1,732 genome pairs (5,460 → 3,728 ceiling)
and worsen outgroup exhaustion while barely moving protein-level concentration — the wrong
lever. The instrument that targets it directly is `max_pairs_per_cluster_class`, still
`null` (r ≈ 0.25 × genome pairs suggested, never measured). Note `pos_weight` already
balances *classes* by inverse frequency but not *genomes within* a class, so a per-genome
or per-cluster cap is complementary rather than redundant.

---

## 2026-08-08 — Class-balance decisions, effective pos_weight, combined sbatch, leakage audit

Pipeline-rebuild steps 5–13. All counts MEASURED on the current
`assemble/labeled_dataset.parquet` + `labeled_dataset_protein_pairs.tsv`
(re-assembled since the previous entry: proteins 18.06M -> **21,580,199**,
pairs 66,759 -> **412,925**; the earlier 66,759 figure is stale).

### Class imbalance — measured at three levels
- **Cross-phenotype pairs** (labeled_dataset_protein_pairs.tsv, 412,925 total):
  psychrophile 309,989 | halophile 43,796 | hyperthermophile 37,355 |
  thermophile 13,136 | alkaliphile 4,748 | acidophile 3,901.
  Spread = **79.5×** (psychrophile/acidophile). [job fcc88de7 + pair_imbalance.json]
- **Within-head pos/neg** (train split, scoped) — negatives are SHARED within a
  scope: secreted heads all share **7,824,488** negatives; whole-scope heads
  share **8,844,060**. pos_frac 0.320 (psychrophile) -> 0.0289 (alkaliphile).
  Raw pos_weight (N_neg/N_pos) 2.12 -> 33.64. [balance_job fcc88de7]
- **Positive confidence tiers** — psychrophile 76.5% LOW / 22.3% med / 1.1% high;
  halophile 82.3% med / only 2.5% low. [tier_job 9ec2ab44]

### Decision 1 — LOSS WEIGHTING, not downsampling
Heads are INDEPENDENT (`for pheno in args.phenotypes`), so cross-phenotype
downsampling (79.5× spread) can only starve the majority head, never help a
minority head's separate BCE. Within-head negative downsampling would discard
up to 97.1% of the shared mesophile-negative signal. pos_weight = N_neg/N_pos
balances analytically with zero data loss.

### Decision 2 — keep rubric weights, compute pos_weight on EFFECTIVE counts
Confidence weights (high 1.0/medium 0.5/low 0.25; mesophile none 1.0) stay on by
default via `--rubric-weights`. KEY FIX: pos_weight must be computed on the
confidence-weighted (effective) positive counts, not raw, or the two terms fight
(rubric down-weights a positive inside the loss; raw pos_weight re-inflates it,
net under-weighting positives — worst for the noisiest heads). Effective
pos_weight runs 2–3× raw: psychrophile 6.75 (raw 2.12), halophile 13.09 (7.46),
hyperthermophile 14.10 (5.87), thermophile 16.56 (8.69), acidophile 42.68
(17.82), alkaliphile 90.97 (33.64).

Code changes deployed to `$SCR/repo` (syntax-verified; /tmp backups):
- `scripts/10_train_cached_probe.py`: pos_weight now `eff_neg/eff_pos` when
  rubric weights active; reduces to raw N_neg/N_pos under `--no-rubric-weights`.
  Logs both (`pos_weight X (raw Y)`); records `pos_weight_raw` in metrics.json.
- `scripts/10b_train_pooling_ablation.py`: same effective-count fix.

### Per-class inclusion (pointwise scope) — wired into stage 15
`scripts/slurm/15_train_all_scoped.sbatch` stage-10 call now passes
`--pointwise-scope --scope-config config/config.yaml`. Secreted-scope heads
(halophile/acidophile/alkaliphile/thermophile) train only on `is_secreted`
proteins; whole_proteome heads (hyperthermophile/psychrophile) use the full
proteome. Scope map = config.dataset.protein_scope (committed at f16ebe3).

### Combined single-allocation sbatch (step 11) — EXISTS
`scripts/slurm/15_train_all_scoped.sbatch`: one gpu_h200 job (1 GPU/16 cpu/400G,
no wall cap) runs stage 00 contacts -> 08 MLM adapter -> 09 embedding cache
(8 shards in-process) -> 10 classifier heads (lambda sweep {0.5,1,2,4}).
Idempotent `.done` markers per stage -> resubmit resumes at first unfinished
stage. Builds corpus_all.faa (secretome+wholeproteome, header-deduped) in-job.
MLM->classifier adapter remap via `load_mlm_adapter_into_classifier`
(model.py:137), which RAISES on 0 transferred LoRA tensors (no silent
vanilla-ESM fallback).

### Leakage audit (step 10) — PASS, measured [job ef11b626 / 1167275]
stratified_group_split (seed 1466) assigns whole `group`s (union-find over
mmseqs 40% + 50% identity maps) to one split, stratified by majority label.
- proteins 21,580,199 -> train 17,262,603 / val 2,160,044 / test 2,157,552
- groups spanning >1 split: **0 of 18,810,756**; cluster_id40 spanning 0;
  cluster_id50 spanning 0
- pairs cross-split: **0 of 412,925** (train 330,573 / val 40,769 / test 41,583)

### SignalP gap-fill (steps 1–4) — in flight
signalp_targeted chunks 0–11, 14 have `.done`; chunks 12/13 (SLURM
1167114/1167115) still running ~6.7 h, covering the remaining 211 gap genomes.
chunk_14 (SLURM 1167269) already verified: 5 genomes / 14,143 proteins. Assemble
stage C auto-merges `signalp_targeted/chunk_*` on rerun.

### Pooling ablation (10b, SLURM 1167271 = host 3753baf9) — running, eval survives
Batched-eval fix (`score_all(head, rows, bs=4096)` to CPU numpy) lets val +
pair gathers run on the 23.56 GB A5000. psychrophile mean-pool ep30: val_auprc
0.739, pair_auc 0.877 (vs old cached-probe 0.658). Attention/top-k arms +
remaining phenotypes still running; ablation_summary.json not yet emitted.

---

## 2026-08-08 — Stage 15 upgraded to K=32 attention + dual-emit, submitted (job 1167340)

### Three sbatch edits (deployed sha256 6fcc1472…208c, bash -n OK)
1. **`--mem` 400G → 1900G.** K=32 top-k cache needs ~460 GB RAM per shard during
   the per-batch gather; mean cache (~185 GB fp32) also lives in CPU RAM
   (`--emb-device auto` picks cpu). gpu_h200 node-224-2t-8gpu-1 measured
   RealMemory=2,063,701 MB (2.06 TB), AllocMem=0 at submit → 1900G honorable.
2. **Stage 09 dual-emit at nshards 16.** Added `--emit-topk --topk 32 --select norm`
   and `--nshards 16` (was 8). ONE forward pass now writes `mean_shard{i}.npy` AND
   `topk_shard{i}.npy` (n_i, K=32, H=2560) fp16 + `lens_shard{i}.npy`. Mean path is
   byte-identical to the no-topk run → any AUPRC delta is pooling, not representation.
   Finer sharding (16) keeps any single `topk_shard*.npy` from blowing the 1.9 TB RAM.
3. **Stage 10b attention pass, pointwise at per-class best λ.** After the 10a mean
   λ-sweep {0.5,1,2,4}, `scripts/select_best_lam.py` reads each
   `cached_probes_lam<λ>/cached_probe_summary.json`, picks the λ that MAXES measured
   `val_auprc` per phenotype (ties → val_pair_auc → smaller λ), and stdout-returns it
   for `$(...)` capture. Each attention head then trains ONCE at its own best λ via
   `--pooling attention --attn-dim 128` over the K=32 block. Output tags:
   mean → `clf_<pheno>_cached`, attention → `clf_<pheno>_attn` (never collide).
   Two production classifiers per class (mean-at-best-λ + attention-at-best-λ).
   K raised from 16 → 32 per user ("h200 has 2t ram, maybe K32 still works"), measured to fit.

### Deployed-script verification (before writing sbatch, not assumed)
- 09 args confirmed: `--emit-topk`/`--topk` (default 32)/`--select {norm,stride}` present.
- 10 args confirmed: `--pooling {mean,attention}`, `--topk-cache-dir`, `--attn-dim` (default
  128), `--emb-device` (default auto). **No `--topk` on stage 10** — K read from cache shape;
  removed an erroneous `--topk` from the 10b call after checking the arg list.
- 10 device branch: mean cache held in CPU RAM, per-batch indexed to GPU; attention path
  fully memmap-backed (union up to 3.7 TB, never concatenated). No OOM path.
- 10 output tag logic `tag = "cached" if not attention else "attn"` (line 400) → `.done`
  markers in sbatch match.

### Staleness catch (would have been a silent "COMPLETED but no work")
Preflight found Aug 5 `models_scoped`/`embeddings_scoped` with EVERY `.done` marker set
(`.contacts_done`, `mlm_adapt/.done`, 8× `.done_shard*`, `secretome_scoped/.done`, 4×
`cached_probes_lam*/.done`). That tree pre-dates the Aug 8 scope-corrected corpus
(`labeled_dataset.parquet` mtime 08-08 08:09 vs subsample/adapter 08-05) and the K=32
decision — embeddings were mean-only `emb_shard*.npy`, **zero `topk_shard` files**. Because
the `.done`-skip runs before the OVERWRITE guard, a `RUN_TAG=scoped` resubmit would have
skipped all stages and "completed" in seconds on stale, wrong-corpus, no-topk data. Also
found `assemble/corpus_all.faa` stale (mtime 08-05) but NOT tag-scoped → removed it so the
job rebuilds from the Aug 8 FASTAs.
- **Resolution (user pick):** fresh `RUN_TAG=scoped_k32` → rebuilds into
  `models_scoped_k32`/`embeddings_scoped_k32`, Aug 5 tree left intact for later purge.
- Corpus rebuild confirmed at runtime: **5,261,647 sequences** (from secretome.faa +
  wholeproteome.faa, header-deduped) — fresh, not the stale 08-05 copy.

### Corpus / pair counts (Aug 8 scope-corrected, stage F)
- `labeled_dataset.parquet` 2.53 GB; `labeled_dataset_protein_pairs.tsv` **412,925 pairs**
  (up from 90,984 pre-correction — 4.5×, driven by the scope fix + augmented cold set).
- Leakage-aware splits (seed 1466, from prior audit job 1167275): pairs train 330,573 /
  val 40,769 / test 41,583; 0 of 412,925 cross-split; 0 of 18,810,756 groups span splits.

### Submission
`RUN_TAG=scoped_k32 sbatch scripts/slurm/15_train_all_scoped.sbatch` → **job 1167340**,
RUNNING on node-224-2t-8gpu-1 (H200, 143,771 MiB VRAM), start 2026-08-08T10:44:11-07:00.
No wall cap. Idempotent per-stage `.done` → resubmit resumes at first unfinished stage.
Expected chain: 00 contacts → 08 MLM (~10 h) → 09 dual-emit embed (16 shards) → 10a mean
sweep (4 λ, minutes each) → 10b attention (6 phenos at per-class best λ).

Artifacts (platform): `15_train_all_scoped.sbatch` (3871ac4c…, sha 6fcc1472…208c),
`select_best_lam.py` (d58e47e4…, sha bee256e3…).

---

## 2026-08-08 (later) — INV-ID: dirty-defline id mismatch truncated the corpus to ~23%

**User challenge that opened this:** "I thought it was a 22M corpus." Runtime log for job
1167340 showed corpus FASTA = 5,261,647 seqs, but the labeled parquet is 22M. Investigating
the gap uncovered a real pipeline bug (not a cosmetic mislabel), traced end-to-end below.

### What is actually true (authoritative stage-C stats, `secreted_all.tsv`, mtime 08-08 07:38)
- Proteins scanned: **133,977,295** across 57,437 genomes (log 1167285).
- Genuinely secreted (SignalP `prediction != "OTHER"`): **19,692,434** (14.70% — plausible).
  by_prediction: SP 13,517,260 · LIPO 4,965,688 · TAT 748,292 · PILIN 349,752 · TATLIPO
  111,442 (non-OTHER sums to 19,692,434 ✓); OTHER 114,284,861.
- Secreted sequences ACTUALLY written to `secretome.faa`: **only 2,057,964** — frozen at
  exactly this number across three runs (jobs 1164632, 1165625, 1167285) while the secreted
  table grew 17.9M → 18.6M → 19.7M. That frozen count is the tell.

### Root cause (single, upstream): SignalP ran on Prodigal-deflined FASTA
SignalP 6 input headers still carried the full Prodigal annotation, so the parsed ID column
(`fields[0]` in `parse_prediction_results`) is DIRTY for every GTDB protein, e.g.
`GB_GCA_000238995.1~CP003199.1_1 # 1 # 1263 # 1 # ID=1_1;partial=10;...`.
`05_aggregate_signalp.py` stored that dirty string verbatim as `tagged_id`. Two downstream
stages then disagreed on the key:
1. **`secretome.faa` writer** cleans the id (`pid = line[1:].split()[0]`) before testing
   `tid in sec_ids`, but `sec_ids` holds the DIRTY id → membership fails → only the ~2.06M
   already-clean-id secreted proteins (original r232 production table + custom `CU_CUST`
   genomes) ever get written. Hence the frozen 2,057,964.
2. **stage-09 `attach_sequences(parquet, fasta)`** joins the DIRTY-id parquet against the
   CLEAN-id FASTAs → only clean-id rows survive.

### Scope of damage — affects secreted AND non-secreted (parquet breakdown, job 9c37e913)
Of 22,477,732 parquet rows:
| | secreted | non-secreted | total |
|---|---:|---:|---:|
| dirty id (dropped by stage-09 join) | 12,609,356 | 4,667,906 | **17,277,262 (77%)** |
| clean id (survives join) | 2,057,964 | 3,142,506 | **5,200,470 (23%)** |
| total | 14,667,320 | 7,810,412 | 22,477,732 |

So the 14,667,320 `is_secreted=True` count is LEGITIMATE SignalP output, not inflated — my
earlier "inflation" read was wrong; it came from comparing against the stale legacy r232
table (~1.99M). The real failure is truncation: the training corpus was built from the
5.2M clean-id survivors (stage-09 intersection = 5,156,130), i.e. **~23% of the intended
corpus**, with the hyperthermophile/psychrophile whole-proteome classes gutted just as
badly as the secretome (their dirty-id whole-proteome rows fail the same join).
`wholeproteome.faa` itself is internally complete (whole-scope branch emits unconditionally
with clean ids) but is unusable downstream because the parquet key is incompatible.

### Fix applied (deployed repo `$S/repo`, `scripts/05_aggregate_signalp.py`, INV-ID)
In the fresh-chunk parse loop, normalize the id to its first whitespace token before
splitting on `~`:
```python
_tok = p.protein_id.split(maxsplit=1)
clean_id = _tok[0] if _tok else p.protein_id
gen, _, pid = clean_id.partition("~")
rows.append((clean_id, gen or None, pid or clean_id, ...))
```
Now `sec_ids`, the parquet `tagged_id`, and both FASTAs all carry the same clean
`{genome}~{locus}` key. Side benefit: proteins appearing in both the legacy table (clean)
and a fresh chunk (previously dirty) now dedupe correctly (`drop_duplicates keep="first"`).
- Verified: AST_OK; parsing a real chunk (`signalp_r232/chunk_0/prediction_results.txt`,
  1,520,117 preds) → 200,000/200,000 sampled ids clean, 0 dirty; clean form matches the
  legacy key format `GB_GCA_...~AE017199.1_36`.
- Legacy table `secreted_proteins_r232.tsv` confirmed clean (separate `genome`+`protein_id`
  columns; `tagged_id` built as `genome~protein_id`). whole_rows path already clean
  (`pid = line[1:].split()[0]`). The fresh-chunk loop was the ONLY dirty source.

### Actions taken
1. **Cancelled job 1167340** (was ~30 min in, at stage 10 cached-probe `48,000/65,912`
   pairs — training on the wrong 23% corpus). `scancel 1167340` confirmed.
2. Applied + verified the INV-ID fix above.
3. This notebook entry. **Corrects the prior 08-08 entry**: the recorded "corpus 5,261,647
   sequences" was the TRUNCATED corpus, not the intended one — the correct scoped corpus is
   built from 19,692,434 secreted + whole-proteome, pending re-aggregation.

### Next (not yet done)
Re-run assembly chain from stage C (05agg re-aggregate → re-emit both FASTAs → stage F
rewrite parquet → 07 cluster → 09 embed → 10). Then resubmit training. Expect
`secretome.faa` to jump from 2.06M to ~19.7M seqs and the stage-09 corpus to grow ~4×.

## 2026-08-08 (later 2) — Deprecated-file cleanup (post-INV-ID regeneration)

Since every artifact downstream of stage C is being regenerated on the corrected corpus,
removed all pre-fix / experimental / cancelled-job outputs to eliminate current-vs-deprecated
ambiguity. Irreversible `rm` on biotite (no Trash), done with explicit paths (no globs),
in parallel with the running chain (verified no overlap with chain reads/writes).

**Deleted (~191 GB):**
- Group 1 — ASM deprecated data (~42 GB): `preemit_secreted_all.tsv` (34G, pre-emitfix secreted table),
  `prescoped_labeled_dataset.parquet` (2.1G), `scopeD_labeled_dataset.parquet` (2.1G),
  `corpus_all.faa` (1.9G, built by cancelled job 1167340 from the TRUNCATED FASTAs),
  + companions `prescoped_dataset_splits.png`, `prescoped_labeled_dataset_protein_pairs.tsv`,
  `scopeD_labeled_dataset_protein_pairs.tsv`.
- Group 2 — PERSIST model/embedding trees (~149 GB): `embeddings_emitfix` (127G), `embeddings_scoped` (10G),
  `embeddings_old` (9.6G), `models_old` (1.9G), `models_emitfix` (396M), `models_scoped` (324M),
  `models_scoped_k32` (30M) + `embeddings_scoped_k32` (0, cancelled 1167340).
- Group 3 — stale markers + superseded audit JSONs: `.C_emitfix_done`, `.F_v9_done`, `harvest_v9.json`,
  `scope_leak_{audit,cache,colabels,tiers}.csv` (all regenerated by the current chain).

**Kept (chain inputs / regenerating in place):** `secreted_proteins_r232.tsv` (live `--legacy`),
`combined_labels.parquet`, `gtdb_meta.tsv`, `all_pairs.tsv` + `sel_*` stage-B outputs,
reference trait CSVs, `config.yaml`, and the 08-08 files the chain overwrites
(`secretome.faa`, `wholeproteome.faa`, `clu*`, `labeled_dataset.*`).

Chain status at cleanup: C=1167378 R (~22 min), D=1167379 / E=1167380 / F=1167381 PD (afterok).

## 2026-08-08 (later 3) — Per-residue phenotype saliency (interpretability tooling)

Goal (user): pick a protein + orthologs, score which residues drive the phenotype
call, map onto structure for "attention"-style structural readout.

**Two additions, neither touches the running chain:**

1. **`scripts/09b_embed_perresidue.py` — persist residue positions.** The top-k
   cache stored the k=32 selected residue VECTORS but discarded WHICH positions
   they came from, so cached alpha/MIL scores couldn't be mapped back to
   sequence/structure. Added `pos_shard{i}.npy` (n, k) int32 = token index per
   slot (CLS=0 so residue r -> token r+1; -1 = padding slot). Written atomically
   alongside topk/mean/lens. AST_OK.

2. **`scripts/score_protein.py` — NEW, full-length dense saliency.** For a handful
   of hand-picked proteins the k=32 cache buys nothing (a full forward pass is a
   few GPU-s and yields a DENSE weight over EVERY residue, including low-norm ones
   the `norm` rule would drop), so this re-embeds rather than reading the cache.
   - Loads backbone + MLM adapter (via existing `load_mlm_adapter_into_classifier`
     remap) + a trained 10b head (`head_best.pt`).
   - Head architecture is INFERRED from the state_dict (net.0.weight -> hidden,
     V.weight -> attn_dim, presence of V/U/w -> attention), so no dependence on
     remembered training args. mean/topk_mil are param-identical -> require
     `--pooling` to disambiguate.
   - attention -> per-residue alpha (head.alpha over full length); topk_mil ->
     sigmoid(per-residue logit); mean -> reported uniform (honest: no localization
     by construction).
   - Emits `<id>_residue_scores.tsv` (residue_index, aa, saliency, percentile),
     `saliency_summary.json` (top-15 residues, special_token_mass), and optionally
     writes saliency into the PDB B-factor column (`--pdb-dir`, fixed-column
     rewrite, `--bfactor percentile|alpha`) for PyMOL/Mol* spectrum coloring.
   - Two input modes: `--fasta`, or `--from-pairs`+`--protein-id`+`--corpus-fasta`
     to pull an ext protein + its taxonomy-matched outgroups straight from the
     pair table and score them together (cross-ortholog consistency check).
   - CAVEATS emitted with output: (a) softmax trained over K=32 -> absolute alpha
     not calibrated at full length, use percentile; (b) genome-level label ->
     saliency = "phenotype-correlated", NOT "catalytic"; (c) CLS/EOS mass reported
     so a diffuse head is visible.
   - Local unit tests PASS: FASTA read/clean-id, streaming id extract, head-kind
     inference, PDB B-factor col rewrite (no column drift, element symbol intact),
     percentile mapping.

Artifacts: score_protein.py (fb86c985), 09b_embed_perresidue.py patched (95089d96).
Deploy to $S/repo/scripts after D/E clustering finishes; not needed until a head
is trained on the corrected corpus.

## 2026-08-08 (later 4): Regeneration chain complete + confidence-weight change (low 0.25→0.15)

### Chain completion (jobs 1167378–1167381, INV-ID corrected corpus)
All four assemble-chain jobs COMPLETED. Verified from stage-06 (F=1167381) log + direct output inspection:
- **labeled_dataset.parquet** 6.7G (Aug 08 14:14), **.tsv** 49G, **protein_pairs.tsv** 87M, splits.png 76K — all fresh Aug 08.
- **22,007,249 rows** | 42,280 genomes | 11,435,706 groups (`sequence_cluster_merged(id40+id50)`).
- Label counts: mesophile 11,557,784 · psychrophile 5,000,109 · hyperthermophile 1,832,384 · halophile 1,511,941 · thermophile 1,159,271 · acidophile 606,418 · alkaliphile 339,342.
- Splits: train 17,597,678 · val 2,199,399 · test 2,210,172. **Leakage check: max splits per group = 1 (PASS).**
- **Protein pairs: 452,487** (all same-split), ~5x the old truncated-corpus 90,984.
- clu50 secretome clustering (D=1167379): 19,692,434 members (== secretome.faa) → 7,771,075 clusters; redundancy 2.534 mem/clu (39.5% reps); singletons 72.8%; size2 950,985; size3-5 697,060; size6-10 250,507; size11-50 188,198; size51-100 18,116; size>100 11,042; max cluster 6,902 (no runaway mega-cluster). clu40 wholeproteome (E=1167380) reported earlier: 1,161,996 clusters healthy.

### confidence_weights: low 0.25 → 0.15 (user decision 2026-08-08, applied uniformly)
Mechanism clarified before changing: TWO orthogonal knobs.
- `pos_weight = n_neg/n_pos` from **raw counts**, per one-vs-rest head, automatic → the CLASS-IMBALANCE knob. Unaffected by this change.
- `CONFIDENCE_WEIGHTS` (per-example multiplier w_i in weighted_bce/focal) → the LABEL-TRUST knob. This is what changed.
- **Source of truth is the module constant `CONFIDENCE_WEIGHTS` in `src/eptrans/modeling/losses.py`, NOT config.yaml** (config line is documentation; runtime never reads it). Edited both; deployed losses.py to `$S/repo` and verified `confidence_to_weight('low')==0.15` in eptrans_ml.

MEASURED per-class × tier composition (train split, job bd638239 on the new parquet) — rubric-rank weights measured, not assumed:
| class | raw pos | high% | med% | low% | eff-signal drop @0.15 vs 0.25 |
|---|---|---|---|---|---|
| psychrophile | 4,000,852 | 0.8 | 19.4 | **79.8** | **−26.2%** |
| hyperthermophile | 1,466,727 | 8.5 | 33.6 | 57.9 | −14.5% |
| alkaliphile | 272,002 | 5.2 | 39.8 | 55.1 | −14.2% |
| acidophile | 485,359 | 12.3 | 34.3 | 53.5 | −12.5% |
| thermophile | 926,552 | 29.2 | 22.8 | 48.0 | −9.1% |
| halophile | 1,207,940 | 13.7 | 83.6 | 2.8 | −0.5% |
Mesophile negative pool (train): 9,238,246 rows, all tier 'none' (w_i=1.0, unaffected).
Note: change does NOT reduce class imbalance (that's pos_weight). Its real effect is a label-trust cut that lands mostly on psychrophile, which is 79.8% low-tier by construction (GenomeSPOT rarely predicts Topt<15C, so hadal/deep-sea cold calls cannot reach high/med tier). User chose uniform 0.15 with this understood.
Artifacts on biotite: `$ASM/tier_x_class_composition.csv`, `$ASM/effective_mass_by_class.csv`.

## 2026-08-08 (later 5): Stage-15 k32 combined training LAUNCHED (job 1167477)

Submitted `RUN_TAG=scoped_k32 sbatch scripts/slurm/15_train_all_scoped.sbatch` -> **job 1167477** (PD, gpu_h200, 1 GPU, --mem=1900G, no wall cap). Single allocation, 5 stages, idempotent .done markers:
- 00 precompute contacts (MLM subsample 400k/20k, ESM-2 contact maps)
- 08 MLM adapter (extremophile-only, contact-coupled masking, rank-32 LoRA, 3 epochs ~10h)
- 09 embedding cache, DUAL-EMIT (mean_shard + topk_shard K=32 L2-norm select, 16 shards) through MLM-adapted 3B backbone
- 10a mean-pooling classifier heads, LAMBDA SWEEP {0.5,1,2,4} x 6 phenotypes, --pointwise-scope
- 10b attention-pooling heads, per-phenotype at its best-lambda (via select_best_lam.py), gated attention over K=32 block, --attn-dim 128
Outputs -> models_scoped_k32/ + embeddings_scoped_k32/ (both verified ABSENT/clean pre-launch, no clobber trip).

### Pre-launch verification (measured, not assumed)
- **Confirmed the DEPLOYED sbatch is the k32 variant** (11,584 B, has --emit-topk / --pooling mean / --pooling attention / select_best_lam.py), NOT the stale Aug-5 local direct-sweep file. Deployed is what runs. Synced local repo to match.
- **FIXED a real gap for user requirement 'save attention residue positions':** deployed `09_embed_secretome.py --emit-topk` computed the top-k residue token indices (`idx`) but DISCARDED them — saved only topk vectors/lens, no positions. My earlier pos-persistence patch was in 09b_embed_perresidue.py, a DIFFERENT script the sbatch never calls. Patched deployed 09_embed_secretome.py to persist `pos_shard{i}.npy` (n,K) int32 = token index per slot (0=CLS, r=residue r, L+1=EOS; -1=padding-gathered), mirroring the 09b convention. AST-valid; `have_all` now requires pos_shard. WITHOUT this, attention alpha could not be mapped back to residues without re-embedding all 22M proteins. Synced to local repo (09_embed_secretome.py 10,412 B).
- Corrected stale sbatch provenance comments: stage F 1165065->1167381, 18.06M->22.01M proteins (22,007,249), 66,759->452,487 pairs.
- Fresh corpus inputs verified Aug 08: secretome.faa 8.3G, wholeproteome.faa 1.1G, labeled_dataset.parquet 6.7G, pairs 87M. corpus_all.faa absent -> rebuilds from fresh FASTAs (expected).
- Disk: 190 TB free on /groups (VAST). k32 top-k cache (~3.6 TB / 16 shards) fits.
- Confidence weight low=0.15 (this session) is deployed in losses.py and will be picked up by stage 10.

Answered user's 3 confirmations: (1) mean AND attention heads YES; (2) lambda sweep YES; (3) attention residue positions -- was NO, now YES after the pos_shard patch.

---

## 2026-08-12 — Psychrophile scope decision: secreted-trained beats whole-trained (controlled test)

**Question.** With psychrophile flipped to `scope=secreted`, does a head *trained on
secreted* psychrophile proteins actually beat the archived head *trained on
whole-proteome* psychrophile proteins **when both are deployed on secreted
proteins**? The two mean-sweep numbers already on record (whole val_auprc 0.5042
vs secreted 0.2843, both λ=0.5) are NOT comparable — different val populations and
base rates (whole ≈12.2%, secreted ≈5.65%).

**Method (controlled, pure inference — no retrain).** Scored BOTH λ=0.5 mean heads
on the IDENTICAL secreted psychrophile val set. Both heads live under
`models_scoped_k32/` → same MLM adapter → the `secretome_scoped` embeddings are
identical inputs; the only moving part is the trained head weights. Script
`stm_detached/psy_scope_compare.py` imports the byte-identical feature/scope/label
helpers from `rescore_tier_val.py`; gathers only the secreted-val rows from the
mean cache via per-shard mmap fancy-index (no 90 GB full load). Ran detached on the
biotite login node (standard partition saturated by signalp-array retries).
Output: `models_scoped_k32/psy_scope_compare.json`.

**Val set (identical for both heads).** n_val = 1,102,388 secreted proteins;
n_pos = 62,204; base rate = 0.0564.

| metric (secreted val) | whole-trained | secreted-trained | winner |
|---|---|---|---|
| all-tier AUROC (scope-decision metric) | 0.7699 | **0.8095** | secreted (+0.040) |
| all-tier AUPRC | 0.188 | **0.2843** | secreted |
| all-tier AUPRC-lift over base | 3.33× | **5.04×** | secreted |
| H+M-only AUROC | **0.8225** | 0.7827 | whole (+0.040) |
| H+M-only AUPRC | 0.0209 | **0.0239** | secreted (n_pos=3,640) |

**Decision.** Secreted-trained wins on the standing scope-decision metric
(all-tier AUROC 0.8095 vs 0.7699) AND on the deployment metric (AUPRC-lift 5.04×
vs 3.33×). The single whole-head win is H+M-only AUROC, a narrower slice of 3,640
highest-confidence positives; on the full deployment population (all-tier) secreted
is unambiguously ahead. **Psychrophile goes forward with secreted scope for
attention pooling** — consistent with the earlier user-set `scope=secreted` and now
data-supported on a controlled, apples-to-apples comparison. Best λ = 0.5 (measured,
job 1173037).

Artifacts: `psy_scope_compare.json` (a98eeb8e / metrics), `psy_scope_compare.png`
(scope comparison figure). Provenance: whole head =
`psychrophile_whole_archive/lam0.5_clf_psychrophile_cached/head_best.pt`; secreted
head = `psy_secreted_lam0.5/clf_psychrophile_cached/head_best.pt`.

---

## 2026-08-13 — Corpus reconciliation: the MAGs are already in; what a "retrain" actually changes

**Why this entry exists.** Coming out of a context compaction, the standing plan
was a "from-scratch retrain to fold in the 4,084 deep-sea MAGs." Before spending an
H200 allocation I verified on disk what the current model was actually trained on,
because redoing work that is already baked in is the exact circular waste we are
trying to stop. Findings below are all measured from files on biotite, not recalled.

**Finding 1 — the deep-sea MAGs are already in the trained corpus.** The 330
selected MAGs (down-selected from the 4,084 labeled pool by the locked
max_per_lineage selection: stage-03d header documents "320 for SignalP under
secreted-scope + 15 whole-proteome, overlap 5") were ingested into
`custom_genomes/` as `CU_CUST_*` on 2026-08-03, then flowed through the Aug-8
rebuild. Measured in `assemble/labeled_dataset.parquet` (mtime 2026-08-08 14:14):
330 CU_ genomes, 145,937 proteins, SignalP-scanned (present in `secreted_all.tsv`
with prediction classes). MAG label distribution by genome:
thermophile 242, halophile 31, acidophile 22, psychrophile 20,
hyperthermophile 11, alkaliphile 1 (proteins: psychrophile 56,953,
thermophile 48,577, hyperthermophile 19,082, halophile 12,412,
acidophile 3,680, alkaliphile 1,845). **SignalP on the MAGs was already done.**

**Finding 2 — corpus, clustering, pairs, and splits are already built at the
committed scope.** `labeled_dataset_protein_pairs.tsv` (452,488 pairs) carries an
explicit `scope` column: psychrophile & hyperthermophile = whole_proteome;
halophile/thermophile/acidophile/alkaliphile = secreted. Per-class pairs:
psychrophile 309,989, halophile 74,046, hyperthermophile 37,355,
thermophile 17,596, alkaliphile 7,819, acidophile 5,682. Psychrophile pairs went
from ~40 (pre-MAG, data-starved) to 309,989 — the MAG augmentation landed exactly
on the weakest class. Splits are leakage-clean: 0 of 11,435,706 clusters straddle a
split boundary (group=cluster_id, verified by groupby nunique).

**Finding 3 — what a rebuild actually changes is the MLM tier scope, not the data.**
The current adapter (mlm_adapt, trained 2026-08-08 19:55) was built
**extremophile-only but ALL tiers** — `09_subsample_mlm.py:subsample()` drops
mesophiles (is_mesophile) but does NOT filter label_confidence, so low-tier is in.
User's Stage 1 wants **M+H only** ("low is too noisy"). Measured extremophile tier
pool: low 6,161,382 / medium 3,453,917 / high 834,166 (of 10,449,465 ext rows).
Dropping low removes ~59% of the extremophile MLM pool. MLM one-rep-per-cluster set
sizes: M+H = 1,566,652 train clusters; M+H+L = 5,695,773. Because the top-k cache is
built AFTER and FROM the adapter (stage order 08 adapter → 09 embed/cache →
10 heads), the M+H change cascades: new adapter → new cache → new heads. THAT is the
non-circular reason to rebuild — not the MAGs.

**Finding 4 — psychrophile scope test (Stage 3) is now well-powered either way.**
Psychrophile extremophile proteins: 5,000,109 total; 622,883 secreted (12.5%);
1,010,516 in M+H tiers (35,786 of those secreted). Enough positives for a clean
whole-vs-secreted comparison at both tier scopes.

Provenance: audit script `stm_detached/corpus_audit.py`, output
`stm_detached/corpus_audit.json`. Superseded old-corpus jobs cancelled this session:
attn_one 1173782 (16h thrash), driver v2 pid 2510665, plus the moot compact-cache
measurement jobs.

---

## 2026-08-13 — mhk32 rebuild: internally-consistent M+H adapter + cache

**Why a new namespace.** Every downstream number (scope test, λ sweep, attention
heads) must sit on the SAME MLM adapter and SAME embedding cache to be comparable.
The one genuine, non-circular change since scoped_k32 is the MLM adapter tier scope
(Stage-1 decision "low is too noisy for the adapter"). So mhk32 = scoped stages
00/08/09 held byte-identical EXCEPT the MLM subsample is medium+high only. Old
scoped_k32 tree left untouched. Namespace: `$PERSIST/runs/mhk32/`.

### Phase 0 — foundation reused as-is + balance policy (all MEASURED)
- **Scope/pairs/splits reused read-only** from `assemble/` (frozen, leakage-verified).
  labeled_dataset.parquet = 22,007,249 rows (330 CU_ deep-sea MAGs already in,
  SignalP-scanned). Pairs = 452,488 with explicit `scope` column: psychrophile
  309,989 + hyperthermophile 37,355 = whole_proteome; halophile 74,046 /
  thermophile 17,596 / alkaliphile 7,819 / acidophile 5,682 = secreted. Splits
  leakage-clean (0 of 11,435,706 clusters straddle a boundary). No SignalP / cluster
  / assembly re-run.
- **Class imbalance at pair level:** ~55x span (psychrophile 309,989 → acidophile
  5,682). Pairs already stratified per (cluster,class); within-class negatives via
  neg_per_pos=3.
- **Balance decision = LOSS WEIGHTING, not downsampling.** 10_train_cached_probe.py
  trains on ALL negatives; class imbalance handled by pos_weight computed on
  confidence-weighted EFFECTIVE counts (composes with the rubric term instead of
  re-inflating raw positives). Matches standing decision keep_low_tier_all_phenotypes.
- **Rubric-rank sample weights:** CONFIDENCE_WEIGHTS = {high 1.0, medium 0.5,
  low 0.15, none 1.0}, hardcoded in losses.py, applied per-positive via
  confidence_to_weight ON TOP of pos_weight. Rubric ON by default.
  DRIFT NOTED (unresolved-by-design): config.yaml confidence_weights says low=0.25
  and the 10_train_cached_probe.py docstring says "a quarter of a high-confidence
  label", but the CODE reads the hardcoded 0.15 from losses.py (config value is
  decorative — not read at runtime). User chose to KEEP 0.15 (match current code)
  rather than edit losses.py to 0.25. No code change made.

### Phase 1 — M+H-only MLM adapter (job 1174037, gpu_h200)
- **Additive `--tiers` flag** added to 09_subsample_mlm.py (default keeps all tiers,
  so scoped flow is byte-for-byte unchanged; .bak saved, AST_OK). Filters on
  label_confidence AFTER the mesophile drop.
- **M+H subsample built + verified** → `runs/mhk32/labeled_mlm_subsample.parquet`:
  22,007,249 → drop mesophiles → 10,449,465 ext → keep M+H → 4,288,083 candidate
  cluster reps → sampled 400,000 train / 20,000 val. Confidence = medium 336,656 +
  high 83,344. ASSERTED: zero low, zero none, zero mesophile. Train label mix:
  halophile 154,460 / psychrophile 90,806 / thermophile 67,123 / hyperthermophile
  42,648 / acidophile 28,579 / alkaliphile 16,384. Verify JSON:
  `runs/mhk32/mlm_subsample_verify.json`.
- **Contact overlap with scoped:** only 23,250 / 420,000 (5.5%) ids overlap the
  scoped contact_pairs.parquet (166,119 ids), so the M+H draw needs a near-full
  contact precompute (396,750 new). Seeded from scoped via `cp` + `--resume` so the
  overlap is free. Coupling-aware masking (--coupling-mode contact) held identical
  to scoped per the standing "adapter method fixed, only tier scope changes" decision.
- **Driver** `scripts/slurm/15_train_all_mhk32.sbatch` (job 1174037): ONE gpu_h200
  allocation, stages 00 contacts → 08 adapter (rank32/alpha64, 3 epochs, lr 1e-4,
  mask 0.15, coupling contact) → 09 dual-emit top-32 cache (whole 22M corpus, 16
  shards, --select norm). Per-stage/per-shard .done markers = preemption-safe resume,
  no wall cap. scope_test / λ sweep / attention are SEPARATE controlled submits.

### Phase-3 driver prep (scope × tier test) — staged while 1174037 runs

Read both ad-hoc scope/tier harnesses in full to reuse (NOT rebuild) them:
- `scripts/scope_tier_measure.py` — Part B trains `all` (H+M+L) vs `hm` (H+M)
  heads per phenotype on a fixed clean H+M val set at a given scope; weighted BCE
  (rubric confidence weights + effective pos_weight) + matched-pair margin (λ=1,
  margin=1). Loss replicates `10_train_cached_probe.py` exactly.
- `stm_detached/psy_scope_compare.py` — does NOT train; scores two pre-trained
  heads (whole- vs secreted-trained) on the IDENTICAL secreted psychrophile val
  set via surgical mmap fancy-index (no 90 GB load).

**Design finding:** the scope and tier axes are only comparable on ONE fixed
eval set. `scope_tier_measure.py` evaluates each head at its own train scope, so
whole-scope and secreted-scope runs land on different val sets — incomparable for
the scope decision. `psy_scope_compare.py` fixes exactly this (one fixed secreted
val set = the deployment compartment). Old scoped headline (identical secreted
val set): secreted-trained 0.8095 vs whole-trained 0.7699 all-tier AUROC.

**New driver** `stm_detached/psy_scope_tier_2x2.py` (deployed, REMOTE_AST_OK
10103 B, SHA 85e0af12): runs the full psychrophile 2×2 — train pointwise scope
∈ {whole, secreted} × tier ∈ {H+M+L, H+M}, all at λ=1 — with ALL four heads
scored on ONE fixed clean secreted H+M val set (secreted val H+M positives + all
secreted val negatives). Margin pairs aligned to train pointwise scope
(INV-SCOPE-E: secreted-scope training uses only secreted ext/outgroup pairs).
Reuses the exact head/loss/gather logic from `scope_tier_measure.py` Part B.
Emits AUROC (primary) + AUPRC + deltas → `psy_scope_tier_2x2.json`. Gated on the
stage-09 mhk32 cache landing.
### Build 1174037 FAILED at stage 08 (HF Hub transient) → fixed + resubmitted as 1174115

Job 1174037 ran 3h41m: stage 00 (contacts) COMPLETED — contact_pairs.parquet
48 MB (scoped seed) → 161 MB, `.contacts_done` written 14:17:20. Stage 08 then
died immediately at model load:
```
RuntimeError: Cannot send a request, as the client has been closed.
OSError: Can't load the model for 'facebook/esm2_t36_3B_UR50D'
```
Root cause: driver set HF_HOME but NOT offline mode, so every stage pings the HF
Hub for metadata at load. Stage 00 got through; 3.5 h later the httpx client hit
a transient node connection failure and — with no offline fallback — couldn't
read the already-present 27 GB local cache. NOT a code/data bug.

Fix: added `export HF_HUB_OFFLINE=1` + `export TRANSFORMERS_OFFLINE=1` after
HF_HOME in the driver (.bak_offline saved). Verified offline resolution before
resubmit: AutoConfig+AutoTokenizer load from cache (hidden=2560, vocab=33),
weight index resolves, refs/main=476b6399 (complete .bin snapshot). No Hub
contact.

Resubmitted as **job 1174115** (gpu_h200, node-224-2t-8gpu-1, start 14:59:05).
Resume confirmed: `00 contacts already done -- skipping` (no 4.5 h recompute),
stage 08 joined 420,000 proteins, `Loading weights 588/588` completed instantly
from offline cache — the exact line that crashed before. Adapter now training.
Driver synced to local repo scripts/slurm/15_train_all_mhk32.sbatch (was
remote-only, never tracked).
### Stage 08 adapter DONE (job 1174115) + stage 09 cache in progress

**mhk32 M+H-only adapter** (`runs/mhk32/mlm_adapt/mlm_adapter_best/`), finished
Aug 14 02:32. val_ppl per epoch (step): 6.1659 (26,263) → 6.1657 (52,527) →
**6.1548 (78,791)**. Monotonic, best=last (epoch 2). vs scoped baseline
6.237→6.217→6.2168: the M+H-only set reaches a LOWER final val-PPL
(6.1548 < 6.2168) — tighter high-confidence training gives cleaner MLM
generalization, the rationale for the M+H tier restriction. adapter_model.
safetensors = 94,413,200 B (94.4 MB, rank-32 LoRA as expected). mlm_history.json
= 1,576 train_loss points + 3 val_ppl. `.done` marker present.

**Stage 09 cache** (whole 22M corpus, dual-emit top-32, 16 shards): shard 0 done
(.done_shard0; emb/topk/ids/lens/pos all present), shard 1 ~57% at ~142 seq/s.
~1 h/shard → cache ETA ~15 h. Downstream (phase-3 2×2, λ sweep, attention) gated
on all 16 shards + $EMB/.done.

### Stage 09 cache COMPLETE + per-head counts (mhk32)

**Cache** `runs/mhk32/embeddings/secretome_mhk32`: 16/16 shards done
(`.done_shard0..15` + `$EMB/.done`), **17,700,477 rows**, emb float16 (…,2560),
pos int32 (…,32), ~2.8 TB. Adapter remap asserted at build: 288 LoRA tensors,
0 unmatched — the mhk32 M+H-only adapter is the one embedded in the cache.
This ONE cache over the WHOLE 22M corpus serves every downstream head so all
scope/λ/tier numbers are comparable.

Cache retention (psychrophile positives): high 100% (41,249), medium 100%
(969,267), low 14.9% (594,424 of 3,989,593); overall pos 32.1%. Built to
preserve H+M signal while subsampling the low tail.

**Broader secretome definition note (mhk32):** `is_secreted=True` here means
signal-peptide-bearing = 14.67M/22M proteins (SignalP SP/LIPO/TAT/TATLIPO/PILIN),
which is broader than the old 1.98M soluble-secreted (SP+TAT only) definition
used in the r232 runs. "secreted" scope below = this 14.67M signal-peptide set.

**Per-head sequence counts (psychrophile, train split, in-cache):**

| Scope | Tier | Total | Neg (mesophile) | Pos | pos breakdown |
|---|---|---|---|---|---|
| whole | all (H+M+L) | 10,523,329 | 9,238,246 | 1,285,083 | H+M 809,719 + low 475,364 |
| whole | H+M | 10,047,965 | 9,238,246 | 809,719 | H+M only |
| secreted | all (H+M+L) | 8,806,705 | 8,308,774 | 497,931 | H+M 28,529 + low 469,402 |
| secreted | H+M | 8,337,303 | 8,308,774 | 28,529 | H+M only |

Negatives identical within a scope (H+M mask touches positives only — the
positives-only H+M rule keeps "technically low" mesophiles as negatives).

### Phase-3 psychrophile scope×tier 2×2 (job 1175979) — LOCK

Driver `scripts/psy_scope_tier_2x2.py`, sbatch
`scripts/slurm/16_psy_scope_tier_2x2.sbatch`. All 4 cells scored on the SAME
fixed secreted-clean eval set (psychrophile secreted val H+M positives + all
secreted val negatives): **pos=3,640, neg=1,040,184, base=0.0035**, λ=1.0.
CONFIDENCE_WEIGHTS {high:1.0, medium:0.5, low:0.15, none:1.0}.

| Scope | Tier | AUROC | AUPRC | train_pos | pos_weight | pairs |
|---|---|---|---|---|---|---|
| whole | H+M+L | 0.8038 | 0.0214 | 1,285,083 | 18.75 | 248,647 |
| whole | H+M | 0.8555 | 0.0306 | 809,719 | 21.92 | 248,647 |
| secreted | H+M+L | 0.7778 | 0.0369 | 497,931 | 96.96 | 3,793 |
| **secreted** | **H+M** | **0.8924** | **0.0872** | 28,529 | 543.75 | 3,793 |

Deltas: H+M − H+M+L = +0.0517 AUROC (whole), +0.1146 (secreted); secreted −
whole (H+M) = +0.0369 AUROC. Both AUROC and AUPRC agree in every comparison.

**Findings:** (1) dropping the low tier HELPS psychrophile in both scopes — its
low tier is 75% of positives and is net noise (metadata-only/conflict labels).
(2) secreted beats whole for H+M, consistent with the standing
psychrophile_scope=secreted decision.

**LOCKED: psychrophile scope=secreted, tier=H+M (AUROC 0.8924).** Caveat:
secreted×H+M trains on only 28,529 positives (pos_weight 543.75) so there is
some variance, but AUPRC (0.0872 ≈ 25× base) confirms it is not an AUROC
artifact.

**Tier policy for the other 5 phenotypes = per-phenotype empirical** (user
decision, this span). Psychrophile's "drop low tier" result is driven by its
unusually noisy low tier (75% low) and does NOT auto-propagate: halophile (6%
low), thermophile (19%), acidophile (24%), alkaliphile (34%), hyperthermophile
(48%) have cleaner low tiers. Each phenotype's tier is decided by its own AUROC
via the 1×2 driver (`scripts/phenotype_tier_1x2.py`, sbatch
`scripts/slurm/17_phenotype_tier_1x2.sbatch`), scope locked = secreted, run as
job 1176274 (H+M vs H+M+L per phenotype, each scored on its own clean eval set).

### Phase-3 per-phenotype tier 1×2 (job 1176274) — per-phenotype tier LOCK

Driver `scripts/phenotype_tier_1x2.py`, sbatch
`scripts/slurm/17_phenotype_tier_1x2.sbatch`. Scope locked = secreted,
GPU-resident (train-device=cuda), λ=1.0, 30 epochs. Each phenotype scored on
its OWN clean eval set (that-phenotype secreted val H+M positives + all secreted
val negatives). Job COMPLETED 1h30m, exit 0. (Note: a first submission wrapped
as `bash <sbatch>` under submit_job ignored the #SBATCH directives and landed on
partition=standard with no GPU — cancelled 1176273; resubmitted via `sbatch
<file>` directly = 1176274, correct gpu_h200 alloc. Gotcha reaffirmed: sbatch the
file, don't bash it.)

| Phenotype | eval base | all AUROC | H+M AUROC | d(H+M−all) | all AUPRC | H+M AUPRC | tier |
|---|---|---|---|---|---|---|---|
| thermophile | 0.0548 | 0.9620 | 0.9686 | +0.0066 | 0.7870 | 0.7917 | **H+M** |
| hyperthermophile | 0.0014 | 0.9889 | 0.9988 | +0.0099 | 0.8736 | 0.9152 | **H+M** |
| acidophile | 0.0227 | 0.9761 | 0.9830 | +0.0069 | 0.7810 | 0.7881 | **H+M** |
| alkaliphile | 0.0105 | 0.9651 | 0.9747 | +0.0096 | 0.6440 | 0.6662 | **H+M** |
| halophile | 0.1182 | 0.9423 | 0.9420 | −0.0003 | 0.7522 | 0.7499 | **H+M+L** |

**Per-phenotype tier decisions (user: per-phenotype empirical, AUROC-max,
AUPRC-confirmed):**
- thermophile, hyperthermophile, acidophile, alkaliphile → **H+M** (drop low).
  Improvement is marginal (+0.007 to +0.010 AUROC) but consistent on both
  metrics — cleaner than psychrophile but low tier still slightly net-noise.
- halophile → **H+M+L** (keep all). The one exception: essentially tied
  (−0.0003 AUROC, −0.0023 AUPRC), so keep low (6% low share, cleanest low tier,
  most positives 1.15M). Keeping low costs nothing.
- psychrophile → **H+M** (from 2×2, +0.1146 AUROC — its low tier is 75% noise).

**Interpretation:** psychrophile's strong "drop low" result does NOT generalize;
it is driven by its uniquely noisy (75%) low tier. The per-phenotype 1×2
confirms the other phenotypes gain little or nothing from dropping low, and
halophile prefers keeping it. This is why the tier decision was made
per-phenotype rather than globally propagated.

**FINAL locked scope+tier per phenotype (all scope=secreted):**
psychrophile H+M · thermophile H+M · hyperthermophile H+M · acidophile H+M ·
alkaliphile H+M · halophile H+M+L.

**REVISED tier policy (user, this span): uniform H+M for ALL 6 phenotypes.**
Halophile's H+M+L preference was within noise (−0.0003 AUROC, −0.0023 AUPRC), so
for consistency the tier is set to H+M across the board rather than treating
halophile as a special case. All other phenotypes already preferred H+M. This
supersedes the halophile→H+M+L line above. FINAL locked (all scope=secreted,
tier=H+M): psychrophile, thermophile, hyperthermophile, acidophile, alkaliphile,
halophile.

### Rubric-rank sample weights + class balance (mhk32) — definition & measurement

**Rubric-rank sample weights.** Per-protein loss weight is a function of the
genome-level label confidence tier (the "rubric rank"), hardcoded in
`src/eptrans/modeling/losses.py:43`:

    CONFIDENCE_WEIGHTS = {"high": 1.0, "medium": 0.5, "none": 1.0, "low": 0.15}

- high (metadata AND prediction agree): full weight 1.0.
- medium (prediction only): 0.5 — down-weighted, prediction is a proxy.
- low (metadata only, or metadata/prediction conflict): 0.15 — strongly
  down-weighted, weakest evidence.
- none (label stamped from a confident-mesophile genome, i.e. a negative): 1.0 —
  full weight; confident mesophiles are high-quality negatives.

`confidence_to_weight()` (same file) maps the tier string to this weight; every
downstream head (2×2, 1×2, λ sweep, attention) applies it identically via a
per-sample `tr_w` multiplier on the BCE term. Deployed copy on biotite verified
byte-identical to local repo (line 43 matches).

**Class balance = effective (confidence-weighted) pos_weight.** Rather than
downsampling negatives, positives are up-weighted in the BCE via
`pos_weight = eff_neg / eff_pos`, where eff_pos = Σ(sample_weight over positives)
and eff_neg = Σ(sample_weight over negatives). This COMPOSES with the rubric: a
low-tier positive contributes only 0.15 to eff_pos, so the effective pos_weight
reflects the confidence-discounted positive mass, not the raw count. Measured
pos_weight per head (scope=secreted, H+M unless noted), from the phase-3 result
JSONs:

| Phenotype | tier | train_pos | pos_weight (eff) |
|---|---|---|---|
| psychrophile | H+M | 28,529 | 543.75 |
| thermophile | H+M | 481,901 | 22.09 |
| hyperthermophile | H+M | 12,076 | 975.44 |
| acidophile | H+M | 197,845 | 64.54 |
| alkaliphile | H+M | 92,691 | 155.71 |
| halophile | H+M | 1,123,318 | 12.89 |

**Downsampling vs loss weighting — decision: loss weighting (no downsampling).**
Measured rationale: (1) the ONE cache is built over the whole corpus with the low
tail already subsampled at cache-build (psychrophile low retained 14.9%), so the
negative pool is fixed and shared across all heads at a locked scope — downsampling
negatives per-head would break the "one cache serves all heads" comparability.
(2) Effective pos_weight handles the imbalance analytically and composes with the
rubric weights in a single BCE term; the measured AUROCs (0.89–0.999) show the
weighting is sufficient even at pos_weight ~975 (hyperthermophile). (3) The
H+M-vs-all deltas (2×2/1×2) directly measure that tier-based sample weighting +
tier restriction, not negative downsampling, is what moves the metric.

## 2026-08-17 — PRODUCTION RUN SUMMARY (mhk32, consolidated)

Single index for the internally-consistent production run. Namespace
`$PERSIST/runs/mhk32/`. All heads sit on ONE M+H-only MLM adapter and ONE top-32
embedding cache over the whole 22M corpus, so every scope/tier/λ number is
directly comparable. Detailed per-stage entries above (2026-08-13 → 2026-08-16);
this section is the consolidated record requested for the production run.

### 1. What went into ADAPTER training (M+H-only MLM adapter)

- **Corpus source (frozen, reused read-only):** `assemble/labeled_dataset.parquet`
  = 22,007,249 rows (incl. 330 CU_ deep-sea MAGs), mtime Aug 8; pairs
  `assemble/labeled_dataset_protein_pairs.tsv` mtime Aug 8. No SignalP / cluster /
  assembly re-run. Splits leakage-clean (0 of 11,435,706 clusters straddle a
  split boundary); split dist train 17,597,678 / val 2,199,399 / test 2,210,172.
- **Tier restriction (the one genuine change vs scoped_k32):** MLM adapter trained
  on medium+high extremophiles ONLY ("low is too noisy for the adapter" — with
  Low the adapter is barely distinguishable from vanilla ESM). `--tiers` flag on
  09_subsample_mlm.py, filtering label_confidence after the mesophile drop.
- **Subsample funnel:** 22,007,249 → drop mesophiles → 10,449,465 extremophile →
  keep M+H → 4,288,083 candidate cluster reps → sampled **400,000 train /
  20,000 val**. Confidence: medium 336,656 + high 83,344. ASSERTED zero low,
  zero none, zero mesophile. Train label mix: halophile 154,460 / psychrophile
  90,806 / thermophile 67,123 / hyperthermophile 42,648 / acidophile 28,579 /
  alkaliphile 16,384. Verify JSON `runs/mhk32/mlm_subsample_verify.json`.
- **Adapter config:** ESM2-3B `facebook/esm2_t36_3B_UR50D` backbone, rank-32 LoRA
  (alpha 64, dropout 0.05), 3 epochs, lr 1e-4, mask 0.15, coupling-aware masking
  (contact mode), contacts precomputed stage-00 (396,750 pairs, scoped seed +
  --resume). Job 1174115 (gpu_h200; 1174037 died on an HF-Hub transient at stage
  08, fixed with HF_HUB_OFFLINE=1 + TRANSFORMERS_OFFLINE=1).
- **Adapter result:** val_ppl 6.1659 (26,263) → 6.1657 (52,527) → **6.1548
  (78,791)**, monotonic, best=epoch 2. LOWER final val-PPL than scoped baseline
  (6.1548 < 6.2168) — tighter high-confidence training → cleaner MLM
  generalization, validating the M+H restriction. `adapter_model.safetensors` =
  94,413,200 B (94.4 MB, rank-32). Path `runs/mhk32/mlm_adapt/mlm_adapter_best/`.

### 2. Embedding cache (shared by all downstream heads)

`runs/mhk32/embeddings/secretome_mhk32`: 16/16 shards, **17,700,477 rows**, emb
float16 (…,2560), pos int32 (…,32), ~2.8 TB. Adapter remap asserted at build:
288 LoRA tensors, 0 unmatched (the M+H adapter is the one embedded). Low tail
subsampled at build (psychrophile positives: high 100% 41,249, medium 100%
969,267, low 14.9% 594,424/3,989,593; overall pos 32.1%). `secreted` scope below
= signal-peptide-bearing 14.67M/22M (SP/LIPO/TAT/TATLIPO/PILIN), broader than the
old 1.98M soluble-secreted definition.

### 3. Proteins into each SCOPE × TIER screen head (train split, in-cache)

**Psychrophile 2×2 (scope × tier), job 1175979:**

| Scope | Tier | Total | Neg (mesophile) | Pos | pos breakdown | train pairs |
|---|---|---|---|---|---|---|
| whole | H+M+L | 10,523,329 | 9,238,246 | 1,285,083 | H+M 809,719 + low 475,364 | 248,647 |
| whole | H+M | 10,047,965 | 9,238,246 | 809,719 | H+M only | 248,647 |
| secreted | H+M+L | 8,806,705 | 8,308,774 | 497,931 | H+M 28,529 + low 469,402 | 3,793 |
| secreted | H+M | 8,337,303 | 8,308,774 | 28,529 | H+M only | 3,793 |

**Per-phenotype tier 1×2 (scope=secreted, H+M+L vs H+M), job 1176274** — train
counts per head:

| Phenotype | tier | train_n | train_pos | pos_weight (eff) | train_pairs |
|---|---|---|---|---|---|
| thermophile | H+M+L | 9,235,326 | 926,552 | 18.76 | 13,956 |
| thermophile | H+M | 8,790,675 | 481,901 | 22.09 | 13,956 |
| hyperthermophile | H+M+L | 8,440,832 | 132,058 | 313.36 | 235 |
| hyperthermophile | H+M | 8,320,850 | 12,076 | 975.44 | 235 |
| acidophile | H+M+L | 8,766,094 | 457,320 | 49.56 | 4,575 |
| acidophile | H+M | 8,506,619 | 197,845 | 64.54 | 4,575 |
| alkaliphile | H+M+L | 8,551,214 | 242,440 | 109.58 | 6,377 |
| alkaliphile | H+M | 8,401,465 | 92,691 | 155.71 | 6,377 |
| halophile | H+M+L | 9,465,361 | 1,156,587 | 12.80 | 59,426 |
| halophile | H+M | 9,432,092 | 1,123,318 | 12.89 | 59,426 |

### 4. Output metrics of the scope/tier screens (λ=1.0)

**Psychrophile 2×2** (all four heads on ONE fixed clean secreted H+M val set:
pos=3,640, neg=1,040,184, base=0.0035):

| Scope | Tier | AUROC | AUPRC |
|---|---|---|---|
| whole | H+M+L | 0.8038 | 0.0214 |
| whole | H+M | 0.8555 | 0.0306 |
| secreted | H+M+L | 0.7778 | 0.0369 |
| **secreted** | **H+M** | **0.8924** | **0.0872** |

Findings: dropping low HELPS psychrophile both scopes (+0.0517 whole, +0.1146
secreted; its low tier is 75% of positives and net noise); secreted beats whole
for H+M (+0.0369 AUROC). AUROC & AUPRC agree everywhere.

**Per-phenotype 1×2** (scope=secreted; each scored on its own clean eval set):

| Phenotype | eval base | all AUROC | H+M AUROC | d(H+M−all) | all AUPRC | H+M AUPRC |
|---|---|---|---|---|---|---|
| thermophile | 0.0548 | 0.9620 | 0.9686 | +0.0066 | 0.7870 | 0.7917 |
| hyperthermophile | 0.0014 | 0.9889 | 0.9988 | +0.0099 | 0.8736 | 0.9152 |
| acidophile | 0.0227 | 0.9761 | 0.9830 | +0.0069 | 0.7810 | 0.7881 |
| alkaliphile | 0.0105 | 0.9651 | 0.9747 | +0.0096 | 0.6440 | 0.6662 |
| halophile | 0.1182 | 0.9423 | 0.9420 | −0.0003 | 0.7522 | 0.7499 |

### 5. LOCKED scope + tier (all phenotypes)

**scope = secreted, tier = H+M** for ALL 6 phenotypes. thermophile / hyper /
acido / alkali / psychrophile all prefer H+M on AUROC; halophile is a within-noise
tie (−0.0003), set to H+M for a uniform, consistent policy (user decision this
span). psychrophile's large H+M win (+0.1146) is unique to its 75%-noisy low
tier — it does NOT generalize, which is exactly why tier was decided
per-phenotype empirically rather than propagated globally.

### 6. λ sweep at locked scope+tier (job 1176405, COMPLETE)

Grid λ ∈ {0, 0.5, 1, 2, 4} × 6 phenotypes, all at the locked scope=secreted,
tier=H+M, seed 1466, mean-pooled cached-probe heads. Each phenotype scored on
its own fixed clean secreted H+M val set. Driver `scripts/lam_sweep.py` (now
records `val_pair_auc` alongside `val_auroc`/`val_auprc` at every grid point).
Artifact `lam_sweep_all.json` (bb960172-050a-468d-8539-5c9834237fa3), keyed by
λ string → ["phenotypes"][pheno] → {val_auroc, val_auprc, val_pair_auc}.

**AUROC grid** (pointwise separation; rows = phenotype, cols = λ):

| pheno | λ0 | λ0.5 | λ1 | λ2 | λ4 |
|---|---|---|---|---|---|
| psychrophile | **0.9057** | 0.8988 | 0.8967 | 0.8956 | 0.8965 |
| thermophile | **0.9693** | 0.9691 | 0.9688 | 0.9687 | 0.9688 |
| hyperthermophile | **0.9990** | 0.9990 | 0.9989 | 0.9990 | 0.9990 |
| acidophile | **0.9838** | 0.9830 | 0.9832 | 0.9832 | 0.9827 |
| alkaliphile | **0.9785** | 0.9771 | 0.9770 | 0.9769 | 0.9766 |
| halophile | **0.9470** | 0.9436 | 0.9421 | 0.9407 | 0.9367 |

**pair-AUC grid** (matched-pair ranking; rows = phenotype, cols = λ):

| pheno | λ0 | λ0.5 | λ1 | λ2 | λ4 |
|---|---|---|---|---|---|
| psychrophile | 0.5606 | 0.5884 | 0.5969 | **0.6071** | 0.6007 |
| thermophile | 0.9067 | 0.9072 | **0.9095** | 0.9089 | 0.9058 |
| hyperthermophile | 0.9150 | 0.9531 | 0.9541 | 0.9551 | **0.9590** |
| acidophile | 0.7996 | 0.7966 | **0.8071** | 0.8014 | 0.8003 |
| alkaliphile | 0.7371 | 0.7416 | 0.7405 | **0.7518** | 0.7482 |
| halophile | 0.7718 | **0.7744** | 0.7702 | 0.7691 | 0.7675 |

**KEY FINDING — the two metrics disagree systematically.** AUROC is maximized
(or statistically tied, ≤0.0003) at **λ=0** for ALL SIX phenotypes: adding the
matched-pair margin loss slightly *lowers* pointwise class separation
everywhere. But pair-AUC prefers **λ>0** for all six. The margin loss trades a
little pointwise AUROC for better matched-pair ranking — which is exactly the
mechanism the ortholog pairs exist to serve (rank the extremophile ortholog
above its mesophile partner). The two objectives are not the same optimum.

### 7. λ LOCK policy — by pair-AUC (user decision this span)

For the attention-pooling deployment heads, λ is locked to the **pair-AUC**
optimum per phenotype (NOT the AUROC optimum). Rationale: the deployment task is
pair-ranking (score an ortholog against its mesophile partner), so the selection
metric must be pair-AUC. The AUROC-optimal λ=0 head is a worse ranker.

| phenotype | locked λ (by pair-AUC) | pair-AUC at that λ |
|---|---|---|
| psychrophile | 2.0 | 0.6071 |
| thermophile | 1.0 | 0.9095 |
| hyperthermophile | 4.0 | 0.9590 |
| acidophile | 1.0 | 0.8071 |
| alkaliphile | 2.0 | 0.7518 |
| halophile | 0.5 | 0.7744 |

Selector `scripts/select_best_lam.py` now supports `--metric pair_auc`
(primary key `val_pair_auc`); the mhk32 lock policy is `--metric pair_auc`.
Verified: the selector reproduces all six locks above from `lam_sweep_all.json`
exactly. Artifacts `best_lam_by_pairauc.json` + `attn_lam_map.json`
(={pheno: λ}, the operative map fed to the attention driver). The AUROC-optimal
alternative (λ=0 for all six, or 4.0 for hyperthermophile on the earlier
broader read) is recorded in `best_lam_by_pheno.json` for contrast but is NOT
the lock.

### 8. Best-λ attention-pooling heads (jobs 1176723–1176728, RUNNING)

Six independent jobs, one per phenotype, each submitted as
`PHENO=<p> LAM=<λ*> sbatch scripts/slurm/19_attn_head.sbatch` with λ* from the
pair-AUC lock table above. K=32 attention pooling over the top-32 cached
embeddings, same locked scope=secreted / tier=H+M / seed 1466. Driver
`scripts/attn_heads.py` emits val_auroc/val_auprc/val_pair_auc/val_pair_acc,
best epoch by val_auroc, and dumps `alpha_ext_best.npy` + `head_best.pt`.
Job map `handoff/attn_jobs.json`. Prior expectation: attention ≈ +0.10 pair-AUC
over mean pooling for psychrophile. Results to be recorded on harvest.

**I/O contention incident + serial-chain fix (jobs 1176723–1176728 → 1176763–1176768).**
The first submission co-scheduled 4 jobs on node-224-2t-8gpu-1 and they wedged:
56 min elapsed, zero epoch progress, all stuck at the `[load] in-cache rows`
gather step, CPULoad 11.8/224 (pure I/O wait), FreeMem down to ~250 GB. Root
cause: each attention head gathers its full clean eval set (~1.04M val negatives)
as a dense fp16 tensor from the **2.8 TB top-32 mmap** — every row is K32×H2560
fp16 ≈ 160 KB, so one eval gather is ~166 GB of random Lustre reads (32× the
per-row I/O of the mean-pooling λ sweep, which read the small `mean_shard`).
Four jobs random-reading a 2.8 TB working set on a 2 TB-RAM node thrash the page
cache into a livelock. This is a top-32-cache property, not a code bug.
**Fix:** `scancel` all 6, resubmit as a serial `--dependency=afterany` chain
(1176763 psychrophile → 764 thermo → 765 hyper → 766 acido → 767 alkali → 768
halo). Solo, the worker gets full bandwidth: verified via
`srun --overlap` on the compute node — `read_bytes` advancing steadily
(~55–107 MB/s), State D, wchan `folio_wait_bit_common`, GPU warming (head on
device). Per-phenotype gather ~20–40 min, full chain ~2–4 h. λ map unchanged
(pair-AUC lock). Job map `handoff/attn_jobs_serial.json`.
**Lesson for future top-32-cache heads:** never co-schedule mmap-gather jobs
over the multi-TB top-K cache on one node — serialize them, or materialize each
phenotype's rows into a compact in-RAM array once before training.

**Selection-metric convention (mhk32, canonical):** scope/tier/signal decisions
→ AUROC; λ lock for deployment heads → pair-AUC; deployment lift reporting →
AUPRC-as-lift over base rate; headline cross-phenotype ranking → pair-AUC.
