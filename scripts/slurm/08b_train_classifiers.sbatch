#!/bin/bash
#SBATCH --job-name=eptrans_clf
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-4%2
#SBATCH --output=/groups/cress/projects/jaymin/eptrans_scratch/logs/clf_%A_%a.log
#
# Stage-2: five per-phenotype classifier heads on the coupling-aware backbone.
# Each array task branches from the SAME Stage-1 MLM LoRA adapter (trained with
# --coupling-mode both), so every classifier inherits the coupling-adapted
# representations. The adapter is loaded via load_mlm_adapter_into_classifier
# (08_train_backbone.py), which remaps the esm.-prefixed MLM keys onto the
# EsmModel classifier tree and VERIFIES a non-empty transfer — a plain
# peft.load_adapter silently drops the mismatched keys.
#
# Array index -> phenotype. %2 throttle keeps <=2 GPUs busy (shared node).
# Submit AFTER Stage-1 MLM completes and mlm_adapter_best is on the persistent FS.

set -euo pipefail
PERSIST=/groups/cress/projects/jaymin/IS1111/eptrans
REPO=${REPO:-/groups/cress/projects/jaymin/eptrans_scratch/repo}
export HF_HOME=${HF_HOME:-$PERSIST/hf_cache}

source /shared/software/miniconda3/latest/etc/profile.d/conda.sh
conda activate eptrans_ml

PHENOS=(acidophile alkaliphile halophile thermophile hyperthermophile)
PHENO=${PHENOS[$SLURM_ARRAY_TASK_ID]}

CLF_LABELED=$PERSIST/labeled_dataset_r232_clustered.parquet
FASTA=$PERSIST/secreted_proteins_r232.faa
PAIRS=$PERSIST/labeled_dataset_r232_clustered_protein_pairs.tsv
MLM_ADAPTER=$PERSIST/models/mlm_adapt/mlm_adapter_best

echo "[08b] phenotype=$PHENO  adapter=$MLM_ADAPTER"
test -d "$MLM_ADAPTER" || { echo "MLM adapter missing: $MLM_ADAPTER"; exit 1; }

python "$REPO/scripts/08_train_backbone.py" classifier \
  --phenotype "$PHENO" \
  --labeled "$CLF_LABELED" --fasta "$FASTA" --pairs "$PAIRS" \
  --mlm-adapter "$MLM_ADAPTER" \
  --backbone-size 3B --lora-rank 32 --lora-alpha 64 \
  --epochs 5 --lr-head 1e-3 --lr-adapter 1e-5 --lam 1.0 --margin 1.0 \
  --neg-per-pos 3.0 --batch-size 16 --max-len 1022 --device cuda \
  --out-dir "$PERSIST/models/clf_${PHENO}"
echo "[08b] $PHENO EXIT $?"
