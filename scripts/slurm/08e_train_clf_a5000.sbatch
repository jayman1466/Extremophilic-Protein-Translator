#!/bin/bash
#SBATCH --job-name=eptrans_a5000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=14-00:00:00
#SBATCH --output=/groups/cress/projects/jaymin/eptrans_scratch/logs/clf_a5000_%x_%j.log
#
# Stage-2 per-phenotype classifier on the A5000 `gpu` partition (24 GB VRAM),
# for the LARGER phenotypes that don't need the contended H200. Runs unattended
# for up to 14 days; --ckpt-every + resume make a wall-kill/preemption recover
# from the last checkpoint on resubmit, so nothing longer than ckpt-interval is
# ever lost.
#
# Phenotype is the first positional arg:  sbatch 08e_train_clf_a5000.sbatch acidophile
#
# MEMORY: A5000 has 24 GB (vs the H200's 143 GB, where bs=16 grabbed ~42 GB).
# --batch-size 6 keeps the working set well under 24 GB at full --max-len 1022
# (no truncation, so results stay comparable to the H200 runs). If a phenotype
# with longer sequences OOMs, drop to --batch-size 4.
#
# This build (repo >= 13aa099) logs val_pair_acc/val_pair_auc per epoch
# (taxonomy-controlled matched-pair eval) and writes a per-epoch matched
# (adapter, head) snapshot under clf_epoch<E>/ — both requested for these runs.

set -euo pipefail
PHENO=${1:?usage: sbatch 08e_train_clf_a5000.sbatch <phenotype>}

PERSIST=/groups/cress/projects/jaymin/IS1111/eptrans
REPO=${REPO:-/groups/cress/projects/jaymin/eptrans_scratch/repo}
export HF_HOME=${HF_HOME:-$PERSIST/hf_cache}

source /shared/software/miniconda3/latest/etc/profile.d/conda.sh
conda activate eptrans_ml

CLF_LABELED=$PERSIST/labeled_dataset_r232_clustered.parquet
FASTA=$PERSIST/secreted_proteins_r232.faa
PAIRS=$PERSIST/labeled_dataset_r232_clustered_protein_pairs.tsv
MLM_ADAPTER=$PERSIST/models/mlm_adapt/mlm_adapter_best

echo "[08e] phenotype=$PHENO  partition=gpu(A5000)  adapter=$MLM_ADAPTER"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
test -d "$MLM_ADAPTER" || { echo "MLM adapter missing: $MLM_ADAPTER"; exit 1; }

python "$REPO/scripts/08_train_backbone.py" classifier \
  --phenotype "$PHENO" \
  --labeled "$CLF_LABELED" --fasta "$FASTA" --pairs "$PAIRS" \
  --mlm-adapter "$MLM_ADAPTER" \
  --backbone-size 3B --lora-rank 32 --lora-alpha 64 \
  --epochs 3 --lr-head 1e-3 --lr-adapter 1e-5 --lam 1.0 --margin 1.0 \
  --neg-per-pos 3.0 --batch-size 6 --max-len 1022 --device cuda \
  --ckpt-every 200 --log-every 20 \
  --out-dir "$PERSIST/models/clf_${PHENO}"
echo "[08e] $PHENO EXIT $?"
