#!/bin/bash
#SBATCH --job-name=eptrans_cached
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/groups/cress/projects/jaymin/eptrans_scratch/logs/cached_probe_%j.log
#
# Frozen-backbone cached-embedding path (design #1), stage B: train ALL 5
# per-phenotype heads on the cached embeddings from 09_embed_secretome. An epoch
# is seconds (head-only on precomputed vectors), so this trains all phenotypes on
# ALL negatives (no neg_per_pos subsampling) in one short job. Needs a GPU only to
# hold the ~2M x 2560 fp32 matrix (~20 GB) for fast indexed batching; the A5000's
# 24 GB fits it. Reports pointwise val AUPRC + taxonomy-controlled pair_acc/pair_auc
# per phenotype. Submit AFTER all 8 embed shards report .done.

set -euo pipefail
PERSIST=/groups/cress/projects/jaymin/IS1111/eptrans
REPO=${REPO:-/groups/cress/projects/jaymin/eptrans_scratch/repo}
export HF_HOME=${HF_HOME:-$PERSIST/hf_cache}

source /shared/software/miniconda3/latest/etc/profile.d/conda.sh
conda activate eptrans_ml

python "$REPO/scripts/10_train_cached_probe.py" \
  --cache-dir "$PERSIST/embeddings/secretome_r232" \
  --labeled "$PERSIST/labeled_dataset_r232_clustered.parquet" \
  --pairs   "$PERSIST/labeled_dataset_r232_clustered_protein_pairs.tsv" \
  --fasta   "$PERSIST/secreted_proteins_r232.faa" \
  --epochs 30 --lr 1e-3 --batch-size 4096 --device cuda \
  --out-root "$PERSIST/models/cached_probes"
echo "[10] EXIT $?"
