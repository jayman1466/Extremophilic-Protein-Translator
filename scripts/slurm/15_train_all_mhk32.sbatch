#!/bin/bash
#SBATCH --job-name=mhk32_build
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=900G
#SBATCH --output=/groups/cress/projects/jaymin/eptrans_scratch/logs/train/mhk32_build_%j.log
#
# mhk32 rebuild -- ONE gpu_h200 allocation, three stages, M+H-only MLM adapter.
#   00  precompute contacts on the M+H subsample (seeded from scoped, --resume)
#   08  domain-adaptive MLM adapter  -- M+H extremophile-only, coupling-aware masking
#   09  embedding cache (DUAL-EMIT)  -- WHOLE corpus through the mhk32 adapter;
#                                       one pass writes mean_shard + topk_shard (K=32, L2-norm)
# Downstream (scope_test / lambda sweep / attention) are SEPARATE controlled
# submits per the mhk32 plan -- NOT in this driver.
#
# Held byte-identical to 15_train_all_scoped stages 00/08/09 EXCEPT the MLM
# subsample is M+H-only (built by patched 09_subsample_mlm --tiers medium,high).
# NO wall cap (biotite standing rule). Idempotent .done markers -> ressubmit resumes.
set -uo pipefail

REPO=${REPO:-/groups/cress/projects/jaymin/eptrans_scratch/repo}
SCR=/groups/cress/projects/jaymin/eptrans_scratch
ASM=$SCR/assemble
PY=/home/jayminp/miniconda3/envs/eptrans_ml/bin/python
export PYTHONPATH=$REPO/src
export HF_HOME=${HF_HOME:-/groups/cress/projects/jaymin/IS1111/eptrans/hf_cache}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RUN=/groups/cress/projects/jaymin/IS1111/eptrans/runs/mhk32
SCOPED=/groups/cress/projects/jaymin/IS1111/eptrans/models_scoped_k32
MLM_EPOCHS=${MLM_EPOCHS:-3}
TOPK=${TOPK:-32}
EMB_NSHARDS=${EMB_NSHARDS:-16}

LABELED=$ASM/labeled_dataset.parquet          # full 22.01M scoped corpus
CORPUS_FASTA=$ASM/corpus_all.faa              # secretome + wholeproteome (prebuilt)
MLM_SUB=$RUN/labeled_mlm_subsample.parquet    # M+H-only, prebuilt + verified
CONTACTS=$RUN/contact_pairs.parquet
MODEL=$RUN/mlm_adapt
EMB=$RUN/embeddings/secretome_mhk32

mkdir -p "$RUN" "$MODEL" "$EMB" "$SCR/logs/train"

source /shared/software/miniconda3/latest/etc/profile.d/conda.sh
conda activate eptrans_ml
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "[mhk32] host=$(hostname) start=$(date -Is)"
test -s "$MLM_SUB"      || { echo "MLM subsample missing: $MLM_SUB"; exit 1; }
test -s "$CORPUS_FASTA" || { echo "corpus FASTA missing: $CORPUS_FASTA"; exit 1; }

# ---- stage 00: contacts on the M+H subsample (seed from scoped, resume-fill) ----
if [ -f "$RUN/.contacts_done" ]; then
  echo "[mhk32] 00 contacts already done -- skipping"
else
  if [ ! -f "$CONTACTS" ] && [ -f "$SCOPED/contact_pairs.parquet" ]; then
    echo "[mhk32] seeding contacts from scoped (23k overlap free)"
    cp "$SCOPED/contact_pairs.parquet" "$CONTACTS"
  fi
  $PY "$REPO/scripts/10_precompute_contacts.py" \
    --labeled "$MLM_SUB" --fasta "$CORPUS_FASTA" --out "$CONTACTS" \
    --backbone-size 3B --threshold 0.5 --min-sep 6 --top-k 128 \
    --max-len 1022 --device cuda --resume
  rc=$?; [ $rc -ne 0 ] && { echo "[mhk32] 00 FAILED rc=$rc"; exit $rc; }
  touch "$RUN/.contacts_done"
fi
echo "[mhk32] 00 done t=$(date -Is)"

# ---- stage 08: MLM adapter (M+H-only, coupling-aware) ----
if [ -f "$MODEL/.done" ]; then
  echo "[mhk32] 08 adapter already done -- skipping"
else
  $PY "$REPO/scripts/08_train_backbone.py" mlm \
    --labeled "$MLM_SUB" --fasta "$CORPUS_FASTA" \
    --backbone-size 3B --lora-rank 32 --lora-alpha 64 \
    --epochs "$MLM_EPOCHS" --lr 1e-4 --mask-rate 0.15 --gamma 1.0 \
    --coupling-mode contact --contact-pairs "$CONTACTS" \
    --contact-threshold 0.5 --contact-min-sep 6 --span-len 3 \
    --batch-size 16 --max-len 1022 --device cuda \
    --out-dir "$MODEL"
  rc=$?; [ $rc -ne 0 ] && { echo "[mhk32] 08 FAILED rc=$rc"; exit $rc; }
  touch "$MODEL/.done"
fi
echo "[mhk32] 08 done t=$(date -Is)"
test -d "$MODEL/mlm_adapter_best" || { echo "adapter_best missing"; exit 1; }

# ---- stage 09: embedding cache (whole corpus, mhk32 adapter, dual-emit top-32) ----
if [ -f "$EMB/.done" ]; then
  echo "[mhk32] 09 cache already done -- skipping"
else
  for sh in $(seq 0 $((EMB_NSHARDS-1))); do
    if [ -f "$EMB/.done_shard${sh}" ]; then echo "[mhk32] 09 shard $sh done"; continue; fi
    $PY "$REPO/scripts/09_embed_secretome.py" \
      --labeled "$LABELED" --fasta "$CORPUS_FASTA" \
      --mlm-adapter "$MODEL/mlm_adapter_best" \
      --backbone-size 3B --lora-rank 32 --lora-alpha 64 \
      --shard "$sh" --nshards "$EMB_NSHARDS" \
      --emit-topk --topk "$TOPK" --select norm \
      --batch-size 8 --max-len 1022 --device cuda \
      --out-dir "$EMB"
    rc=$?; [ $rc -ne 0 ] && { echo "[mhk32] 09 shard $sh FAILED rc=$rc"; exit $rc; }
    touch "$EMB/.done_shard${sh}"
  done
  touch "$EMB/.done"
fi
echo "[mhk32] 09 done t=$(date -Is)"
echo "[mhk32] BUILD COMPLETE end=$(date -Is)"
