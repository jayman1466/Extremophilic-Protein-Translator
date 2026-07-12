#!/bin/bash
#SBATCH --job-name=eptrans_hyper
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/groups/cress/projects/jaymin/eptrans_scratch/logs/clf_hyper_%j.log
#
# Stage-2 PROOF-OF-CONCEPT: a single per-phenotype classifier head for
# hyperthermophile on the coupling-aware backbone. Branches from the SAME
# Stage-1 MLM LoRA adapter (--coupling-mode both) as the full 08b array, via
# load_mlm_adapter_into_classifier (remaps esm.-prefixed MLM keys onto the
# EsmModel classifier tree and verifies a non-empty transfer).
#
# Differences from 08b_train_classifiers.sbatch:
#   * ONE job (no --array), hyperthermophile only — the smallest phenotype
#     (~40k train rows after neg_per_pos=3), so 3 epochs fit comfortably.
#   * --epochs 3 (was 5).
#   * --ckpt-every 100  -> atomic clf_ckpt.pt + flushed clf_history.json every
#     100 steps, so a growing metrics file + mtime-updating checkpoint prove the
#     job is advancing between the epoch-end AUPRC evals.
#   * --log-every 20    -> a flushed "[train_clf] epoch E step S/T loss L" line
#     every 20 steps, so the SLURM log shows live progress within ~1 min.

set -euo pipefail
PERSIST=/groups/cress/projects/jaymin/IS1111/eptrans
REPO=${REPO:-/groups/cress/projects/jaymin/eptrans_scratch/repo}
export HF_HOME=${HF_HOME:-$PERSIST/hf_cache}

source /shared/software/miniconda3/latest/etc/profile.d/conda.sh
conda activate eptrans_ml

PHENO=hyperthermophile

CLF_LABELED=$PERSIST/labeled_dataset_r232_clustered.parquet
FASTA=$PERSIST/secreted_proteins_r232.faa
PAIRS=$PERSIST/labeled_dataset_r232_clustered_protein_pairs.tsv
MLM_ADAPTER=$PERSIST/models/mlm_adapt/mlm_adapter_best

echo "[08c] phenotype=$PHENO  adapter=$MLM_ADAPTER"
test -d "$MLM_ADAPTER" || { echo "MLM adapter missing: $MLM_ADAPTER"; exit 1; }

python "$REPO/scripts/08_train_backbone.py" classifier \
  --phenotype "$PHENO" \
  --labeled "$CLF_LABELED" --fasta "$FASTA" --pairs "$PAIRS" \
  --mlm-adapter "$MLM_ADAPTER" \
  --backbone-size 3B --lora-rank 32 --lora-alpha 64 \
  --epochs 3 --lr-head 1e-3 --lr-adapter 1e-5 --lam 1.0 --margin 1.0 \
  --neg-per-pos 3.0 --batch-size 16 --max-len 1022 --device cuda \
  --ckpt-every 100 --log-every 20 \
  --out-dir "$PERSIST/models/clf_${PHENO}"
echo "[08c] $PHENO EXIT $?"
