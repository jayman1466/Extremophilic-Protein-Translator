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


# ---- run tagging + clobber guard -------------------------------------------------
# Stages 08/09/10 previously wrote to FIXED paths with no run identifier, so a retrain
# silently overwrote the previous run's artifacts in place. That put ~10.7 h of H200 MLM
# training (and the ~3 h embedding cache behind it) one accidental resubmission away from
# destruction; run 1 had to be rescued by hand into models_old/ and embeddings_old/.
#
# RUN_TAG defaults to empty, preserving the historical paths so existing tooling and
# docs keep working. Set RUN_TAG=run2 (or any label) to write to a parallel tree.
# OVERWRITE=1 is required to write into a directory that already holds artifacts --
# otherwise the job refuses to start, before spending any GPU time.
RUN_TAG=${RUN_TAG:-}
OVERWRITE=${OVERWRITE:-0}
if [ -n "$RUN_TAG" ]; then
  MODEL_ROOT="$PERSIST/models_$RUN_TAG"
  EMB_ROOT="$PERSIST/embeddings_$RUN_TAG"
else
  MODEL_ROOT="$PERSIST/models"
  EMB_ROOT="$PERSIST/embeddings"
fi

# refuse_clobber <dir> <what> -- non-empty target plus OVERWRITE!=1 is a hard stop
refuse_clobber() {
  local d="$1" what="$2"
  if [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ]; then
    if [ "$OVERWRITE" != "1" ]; then
      echo "REFUSING TO START: $what already exists at" >&2
      echo "  $d" >&2
      echo "Overwriting it would destroy prior-run artifacts (the MLM adapter alone is" >&2
      echo "~10.7 h of H200 time). Either:" >&2
      echo "  * set RUN_TAG=<label> to write to a parallel tree, or" >&2
      echo "  * archive the existing tree: mv '$d' '${d}_old'  -- then rerun, or" >&2
      echo "  * set OVERWRITE=1 if you genuinely intend to replace it." >&2
      exit 1
    fi
    echo "WARNING: OVERWRITE=1 -- replacing existing $what at $d" >&2
  fi
}
# ---------------------------------------------------------------------------------

source /shared/software/miniconda3/latest/etc/profile.d/conda.sh
conda activate eptrans_ml

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

refuse_clobber "$EMB_ROOT/secretome_r232" "an embedding cache"
python "$REPO/scripts/09_embed_secretome.py" \
  --labeled "$PERSIST/labeled_dataset_r232_clustered.parquet" \
  --fasta   "$PERSIST/secreted_proteins_r232.faa" \
  --mlm-adapter "$MODEL_ROOT/mlm_adapt/mlm_adapter_best" \
  --backbone-size 3B --lora-rank 32 --lora-alpha 64 \
  --shard "$SLURM_ARRAY_TASK_ID" --nshards 8 \
  --batch-size 8 --max-len 1022 --device cuda \
  --out-dir "$EMB_ROOT/secretome_r232"
echo "[09] shard $SLURM_ARRAY_TASK_ID EXIT $?"
