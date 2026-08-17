#!/bin/bash
#SBATCH --job-name=attn_mhk32
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=1900G
#SBATCH --time=24:00:00
#SBATCH --output=/groups/cress/projects/jaymin/eptrans_scratch/logs/train/attn_mhk32_%j.log
set -euo pipefail
source /shared/software/miniconda3/latest/etc/profile.d/conda.sh
conda activate eptrans_ml
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PY=/home/jayminp/miniconda3/envs/eptrans_ml/bin/python
REPO=/groups/cress/projects/jaymin/eptrans_scratch/repo
RUN=/groups/cress/projects/jaymin/IS1111/eptrans/runs/mhk32
ASM=/groups/cress/projects/jaymin/eptrans_scratch/assemble
EMB=$RUN/embeddings/secretome_mhk32
: "${PHENO:?set PHENO}"
: "${LAM:?set LAM}"
echo "[attn] pheno=$PHENO lam=$LAM node=$(hostname) $(date)"
$PY $REPO/scripts/attn_heads.py \
  --cache-dir $EMB \
  --labeled $ASM/labeled_dataset.parquet \
  --pairs $ASM/labeled_dataset_protein_pairs.tsv \
  --out $RUN/phase5_attn \
  --scope secreted \
  --phenotypes "$PHENO" \
  --default-lam "$LAM" \
  --neg-per-pos 3.0 \
  --epochs 30 --lr 1e-3 --batch-size 4096 --eval-batch-size 8192 \
  --hidden 512 --dropout 0.1 --attn-dim 128 --margin 1.0 \
  --pair-batch-size 256 --seed 1466 --device cuda
echo "[attn] EXIT $? $(date)"
