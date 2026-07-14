# Extremolith

**GUI frontend at https://ept-portal-g2qcpwpcsa-uc.a.run.app/**
(WARNING: currently this just returns precomputed results of a lignin-degrading
laccase, regardless of the sequence submitted. Need to work out a cheap/scalable
GPU solution for live runs by the general public.)

*Extremophilic Protein Translator — secreted proteins from life at the edge.*

Build a labeled dataset of **secreted proteins from extremophilic organisms**
(drawn from GTDB), for fine-tuning a protein language model (e.g. ESM) and/or
training a classifier that learns the *extremophilic trait* of extracellular
proteins.

## Rationale

Proteins exposed to the extracellular environment (secreted proteins) must
function under whatever conditions the organism inhabits. By collecting
secreted proteins from organisms binned by environment (temperature, pH,
salinity), we get a signal for how sequence adapts to extreme conditions.

Two problems this pipeline is explicitly designed around:

1. **Environment labels are noisy.** An organism's isolation source is only a
   proxy — a cell can be found somewhere it is not thriving. We therefore
   combine the GTDB/NCBI metadata proxy with **GenomeSPOT** genomic predictions
   of optimal temperature, pH, and salinity, and keep a confidence label that
   records whether the two agree.

2. **Clade ≠ trait.** A naive dataset lets a model learn to recognize a
   phylogenetic clade and associate it with an environment, rather than
   learning the adaptive trait itself. We control for this by (a) selecting
   extremophiles that are **phylogenetically diverse**, and (b) pairing each
   with a **phylogenetically close mesophile outgroup**, then splitting
   train/val/test to prevent both sequence-identity and phylogenetic leakage.

## Pipeline

| Stage | Module | Script | Output |
|-------|--------|--------|--------|
| 1. Index GTDB | `eptrans.gtdb` | `01_index_gtdb.py` | `results/gtdb_reps_metadata.parquet` |
| 2. Metadata flags | `eptrans.binning` | (in 01/02) | `results/metadata_flags.parquet` |
| 3. GenomeSPOT | `eptrans.binning` | `02_run_genomespot.py` | per-genome predictions |
| 3b. Reconcile predictions | `eptrans.reconcile` | `02b_reconcile_genomespot.py` | `results/genomespot_reconciled.tsv` |
| 4. Combined labels | `eptrans.binning` | (in 02b/03) | `results/environment_labels.parquet` |
| 5. Phylo selection | `eptrans.selection` | `03_select_genomes.py` | `results/selected_genomes.parquet` |
| 6. SignalP | `eptrans.signalp` | `04_run_signalp.py` | secreted-protein table |
| 7. Dataset | `eptrans.dataset` | `05_build_dataset.py` | `results/pilot_dataset.parquet` + FASTA |

## Data location

The heavy inputs live on the **biotite** SLURM cluster (not in this repo):

```
/groups/cress/projects/jaymin/IS1111/
├── gtdb/
│   ├── ar53_metadata_r232.tsv.gz
│   ├── bac120_metadata_r232.tsv.gz
│   ├── protein_faa_reps/{archaea,bacteria}/<PREFIX>_<acc>_protein.faa.gz
│   └── gtdb_genomes_reps_r232/database/{GCA,GCF}/NNN/NNN/NNN/<acc>_genomic.fna.gz
└── work/
    ├── gtdb_reps.faa            # combined proteomes, headers >{GENOME}~{PROTID}
    ├── protein_coords.tsv.gz    # tagged_id, genome, accession, contig, start, end, strand
    └── genome_index.tsv         # headerless: <bare_acc>\t<abs_path_to_.fna.gz>
```

GTDB release **r232**; **199,923** species representatives
(189,801 bacteria + 10,122 archaea). All paths and conventions are recorded in
`config/config.yaml`.

## Conventions

- **Genome id** keeps the GTDB source prefix: `GB_` (GenBank/GCA) or `RS_`
  (RefSeq/GCF), e.g. `RS_GCF_000005845.2`.
- **Combined FASTA header:** `>{GENOME}~{PROTID}`, e.g.
  `>GB_GCA_000008085.1~AE017199.1_1`.
- **Representative flag:** metadata column `gtdb_representative == 't'`.

## Environments

- `environment/genomespot.yml` — GenomeSPOT (py3.11, `hmmlearn==0.3.0`).
- `environment/smoketest.yml` — local pipeline + DeepSig (SignalP fallback).
- `environment/translator.yml` — PLM fine-tuning (torch, transformers, ESM).

SignalP 6.0 is installed and in `PATH` on biotite (no local install required).

## Usage

```bash
pip install -e .                    # install the eptrans package
python scripts/01_index_gtdb.py     # (run on biotite; see script headers)
```

Compute-heavy stages (GenomeSPOT over the recompute delta, SignalP over
proteomes) run as SLURM array jobs on biotite — see `scripts/*.slurm`.

## Layout

```
src/eptrans/     Python package (pipeline modules)
scripts/         numbered stage runners + SLURM array scripts
config/          config.yaml (paths, thresholds, conventions)
environment/     conda env specs
tests/           unit tests (parsers, reconciliation logic)
results/         parquet tables, figures, reports (large files gitignored)
data/            local scratch / downloaded precomputed data (gitignored)
```

## License

MIT (see `LICENSE`).
