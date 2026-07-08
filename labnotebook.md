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

**End-to-end status:** the full chain runs — GTDB metadata → flag → GenomeSPOT →
combine → select → SignalP → labeled dataset — on real data, producing a
268-protein labeled secreted-protein dataset from 14 genomes. At production
scale the same scripts run via the chunked SLURM emitters.
- **Stage 06** — labeled dataset assembly (leakage-aware splits).
- **Pilot** — end-to-end run on a small genome set + report.

_Infrastructure notes: biotite SSH + scratch dir + GitHub credential all
configured. Job submission via SLURM (`standard`/`memory`/`gpu` partitions)._
