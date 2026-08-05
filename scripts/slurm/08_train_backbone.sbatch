#!/bin/bash
#SBATCH --job-name=eptrans_train
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/groups/cress/projects/jaymin/eptrans_scratch/logs/train_%j.log
#
# Fine-tune the shared ESM-2 3B backbone (design doc Section 11) on the
# leakage-controlled secreted-protein dataset. Stage 1 = domain-adaptive MLM;
# Stage 2 (separate submit) = per-phenotype classifier heads.
#
# Requires a modeling conda env on biotite with torch(+CUDA)/transformers/peft/
# scikit-learn. Create once:
#   conda create -n eptrans_ml python=3.11 -y
#   conda activate eptrans_ml
#   pip install "torch" transformers peft scikit-learn pandas pyarrow pyyaml
# ESM-2 3B weights (~11 GB) download from HF on first run to $HF_HOME.

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

MODE=${1:-mlm}                 # mlm | classifier
PHENO=${2:-thermophile}        # used when MODE=classifier

# MLM uses a cluster-stratified ~400k subsample (scripts/09_subsample_mlm.py);
# classifier uses the full labeled table.
MLM_LABELED=$PERSIST/labeled_mlm_subsample.parquet
CLF_LABELED=$PERSIST/labeled_dataset_r232_clustered.parquet
FASTA=$PERSIST/secreted_proteins_r232.faa
PAIRS=$PERSIST/labeled_dataset_r232_clustered_protein_pairs.tsv

# guard BEFORE any GPU time is spent
if [ "$MODE" = "mlm" ]; then
  refuse_clobber "$MODEL_ROOT/mlm_adapt" "an MLM adapter"
else
  refuse_clobber "$MODEL_ROOT/clf_${PHENO}" "a ${PHENO} classifier"
fi

if [ "$MODE" = "mlm" ]; then
  python "$REPO/scripts/08_train_backbone.py" mlm \
    --labeled "$MLM_LABELED" --fasta "$FASTA" \
    --backbone-size 3B --lora-rank 32 --lora-alpha 64 \
    --epochs 3 --lr 1e-4 --mask-rate 0.15 --gamma 1.0 \
    --batch-size 16 --max-len 1022 --device cuda \
    --out-dir "$MODEL_ROOT/mlm_adapt"
else
  python "$REPO/scripts/08_train_backbone.py" classifier \
    --phenotype "$PHENO" \
    --labeled "$CLF_LABELED" --fasta "$FASTA" --pairs "$PAIRS" \
    --mlm-adapter "$MODEL_ROOT/mlm_adapt/mlm_adapter_best" \
    --backbone-size 3B --lora-rank 32 --lora-alpha 64 \
    --epochs 5 --lr-head 1e-3 --lr-adapter 1e-5 --lam 1.0 --margin 1.0 \
    --batch-size 16 --max-len 1022 --device cuda \
    --out-dir "$MODEL_ROOT/clf_${PHENO}"
fi
echo "TRAIN EXIT $?"
