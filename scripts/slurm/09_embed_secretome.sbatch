#!/bin/bash
#SBATCH --job-name=eptrans_embed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/groups/cress/projects/jaymin/eptrans_scratch/logs/embed_%A_%a.log
#SBATCH --array=0-7
#
# Frozen-backbone cached-embedding path (design #1), stage A: embed the whole
# secretome ONCE through the MLM-adapted 3B backbone. 8 shards (one GPU each) on
# the A5000 gpu partition; each shard embeds rows i::8 and writes emb_shard{i}.npy
# + ids_shard{i}.txt. Idempotent per shard (.done_shard{i} guard), so a
# resubmit skips finished shards. Inference only (no grad, no checkpointing),
# fp16 output vectors.

set -euo pipefail
PERSIST=/groups/cress/projects/jaymin/IS1111/eptrans
REPO=${REPO:-/groups/cress/projects/jaymin/eptrans_scratch/repo}
export HF_HOME=${HF_HOME:-$PERSIST/hf_cache}

source /shared/software/miniconda3/latest/etc/profile.d/conda.sh
conda activate eptrans_ml

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python "$REPO/scripts/09_embed_secretome.py" \
  --labeled "$PERSIST/labeled_dataset_r232_clustered.parquet" \
  --fasta   "$PERSIST/secreted_proteins_r232.faa" \
  --mlm-adapter "$PERSIST/models/mlm_adapt/mlm_adapter_best" \
  --backbone-size 3B --lora-rank 32 --lora-alpha 64 \
  --shard "$SLURM_ARRAY_TASK_ID" --nshards 8 \
  --batch-size 8 --max-len 1022 --device cuda \
  --out-dir "$PERSIST/embeddings/secretome_r232"
echo "[09] shard $SLURM_ARRAY_TASK_ID EXIT $?"
