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
cold evidence across four populations (n=1452/639/443/1417). A calibrated weight
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
