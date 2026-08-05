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

refuse_clobber "$MODEL_ROOT/cached_probes" "cached probe heads"
python "$REPO/scripts/10_train_cached_probe.py" \
  --cache-dir "$EMB_ROOT/secretome_r232" \
  --labeled "$PERSIST/labeled_dataset_r232_clustered.parquet" \
  --pairs   "$PERSIST/labeled_dataset_r232_clustered_protein_pairs.tsv" \
  --fasta   "$PERSIST/secreted_proteins_r232.faa" \
  --epochs 30 --lr 1e-3 --batch-size 4096 --device cuda \
  --out-root "$MODEL_ROOT/cached_probes"
echo "[10] EXIT $?"
