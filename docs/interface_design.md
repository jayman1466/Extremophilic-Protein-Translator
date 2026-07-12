# Interface design — design portal

## Goal
Input an enzyme sequence → output N designs per selected extremophilic phenotype,
each with predicted structure overlaid on wild-type, active-site RMSD, classifier
score, and other folding metrics. Downloadable as TSV + multi-FASTA.

## Architecture: decouple frontend from engine
Two independent pieces talking over a **job queue**, never a direct call:

1. **Thin frontend** (`webapp/`, Flask) — validates input, records a job, renders
   results. No GPU, no large databases. Runs on **Cloud Run** (pennies).
2. **Generation engine** — MSA → coupling-aware mask-gen → MPNN gate → fold → RMSD.
   Runs where the compute + databases already live: **Biotite SLURM** now; a
   **serverless GPU** (Modal-style) + **ColabFold MSA API** for public deployment
   later. The engine never has to be packaged or exposed — the swap is a config
   change because every backend writes the same results bundle.

## Results bundle contract (every backend produces this)
```
job_dir/
  results.json          # schema in webapp/make_demo_results.py docstring
  structures/wt.pdb
  structures/<design_id>.pdb
```
`results.json`: `{wt_structure, wt_sequence, by_phenotype: {ph: [{design_id,
sequence, highlighted_seq, classifier_score, active_site_rmsd, n_mutations,
structure_file, metrics:{...}}]}}`. The frontend is a pure function of this bundle,
so Biotite-now and serverless-later render identically.

## Storage on Cloud Run
One **GCS bucket**, mounted via GCS FUSE (Cloud Run gen2):
- **SQLite** file on the mount (`store.py`, SQLAlchemy) for job metadata — genuine
  SQL, fine at demo concurrency. `$DATABASE_URL` swaps to **Cloud SQL Postgres**
  with zero code change when concurrency grows.
- Same bucket holds `jobs/<job_id>/` result files.
Google Drive is intentionally avoided — no clean programmatic Cloud Run mount.

## WT structure: fold WT and designs with the SAME method (ESMFold)
Active-site RMSD is a *relative* WT-vs-design comparison. Folding WT with AF2 and
designs with ESMFold injects cross-method bias (~1–2 Å) into the RMSD. Folding both
with ESMFold cancels the method error, so RMSD reflects the sequence change — what
we want. A "fold WT with ColabFold/AF2" toggle is exposed as a later option for an
absolute-accuracy WT reference.

## Pipeline-component sections (data-driven, `pipeline_options.py`)
Grouped by **role**, not by database, so options extend by appending a dict entry:
- **Generator (MLM):** ESM-2 3B + extremophilic adapter
- **Structure prediction:** ESMFold (folds WT + designs)
- **MSA / conservation:** MMseqs2 UniRef30 (ColabFold envDB later)
- **Active-site annotation** (multi): M-CSA, InterPro, Pfam, Swiss-Prot
- **Structural homology / Foldseek** (multi): pdb100, AlphaFold — split into its own
  section because it does double duty (catalytic-residue transfer + fold-compat)
- **Structural gate:** ProteinMPNN
- **Scoring:** per-phenotype classifiers
Disabled options render as "coming soon" to advertise the extension point.

## Frontend theme
Bootstrap 5 + custom palette: primary `#C34C62`, secondaries `#e3b8c1` (rose),
`#3b6a80` (steel), `#c3e2ee` (cyan). Results = nested accordion (phenotype →
design), Mol* overlay (WT steel / design rose), mutations highlighted inline.

## Deployment paths
- **Demo (video):** frontend local or on Cloud Run, backend Biotite SSH+SLURM, with
  a pre-warmed cache keyed by input hash for showcase enzymes (real results, instant).
- **Public:** same frontend; backend → serverless GPU inference (ESM-3B adapter +
  MPNN + ESMFold, weights in the image/volume) + ColabFold MSA API (removes the
  600 GB database from the hosting problem). Biotite never exposed to public traffic.
