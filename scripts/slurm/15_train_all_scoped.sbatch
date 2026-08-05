#!/bin/bash
#SBATCH --job-name=eptrans_train_all
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --output=/groups/cress/projects/jaymin/eptrans_scratch/logs/train/train_all_%j.log
#
# ONE allocation, four stages, scope-corrected corpus (stage F job 1165065):
#   00  precompute contacts           -- ESM-2 contact maps for the MLM subsample (cached)
#   08  domain-adaptive MLM adapter   -- extremophile-only corpus subsample, coupled masking
#   09  embedding cache               -- whole corpus through the MLM-adapted backbone
#   10  classifier heads, LAMBDA SWEEP -- all phenotypes x lam in {0.5,1,2,4}
#
# Runs the whole chain inside a single gpu_h200 job (no inter-job SLURM
# dependency that can wedge in queue). NO wall-time cap (biotite standing rule).
# Each stage has an idempotent .done marker, so a resubmit resumes at the first
# unfinished stage rather than repeating ~10 h of MLM training.
#
# Submit:  RUN_TAG=scoped sbatch scripts/slurm/15_train_all_scoped.sbatch
#   RUN_TAG   -> writes to models_<tag>/ + embeddings_<tag>/ (default 'scoped')
#   OVERWRITE=1 to replace an existing non-empty tree (guarded; default refuses)
#   LAMS="0.5 1 2 4"      override the sweep grid
#   MLM_EPOCHS / CLF_EPOCHS override epoch counts
set -uo pipefail

REPO=${REPO:-/groups/cress/projects/jaymin/eptrans_scratch/repo}
SCR=/groups/cress/projects/jaymin/eptrans_scratch
ASM=$SCR/assemble
PY=/home/jayminp/miniconda3/envs/eptrans_ml/bin/python
export PYTHONPATH=$REPO/src
export HF_HOME=${HF_HOME:-/groups/cress/projects/jaymin/IS1111/eptrans/hf_cache}

RUN_TAG=${RUN_TAG:-scoped}
OVERWRITE=${OVERWRITE:-0}
LAMS=${LAMS:-"0.5 1 2 4"}
MLM_EPOCHS=${MLM_EPOCHS:-3}
CLF_EPOCHS=${CLF_EPOCHS:-30}

MODEL_ROOT=/groups/cress/projects/jaymin/IS1111/eptrans/models_$RUN_TAG
EMB_ROOT=/groups/cress/projects/jaymin/IS1111/eptrans/embeddings_$RUN_TAG
mkdir -p "$MODEL_ROOT" "$EMB_ROOT" "$SCR/logs/train"

# ---- scope-corrected inputs (stage F 1165065) ----
LABELED=$ASM/labeled_dataset.parquet                       # 18.06M scoped proteins
PAIRS=$ASM/labeled_dataset_protein_pairs.tsv               # 66,759 matched pairs
MLM_SUB=$MODEL_ROOT/labeled_mlm_subsample.parquet          # built below (extremophile-only)
# corpus spans secretome + whole proteome -> embedding needs sequences for both
CORPUS_FASTA=$ASM/corpus_all.faa

refuse_clobber() {  # <dir-or-file> <what>
  local d="$1" what="$2"
  if [ -e "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null || echo x)" ]; then
    if [ "$OVERWRITE" != "1" ]; then
      echo "REFUSING TO START: $what already exists at $d" >&2
      echo "  set RUN_TAG=<label> for a parallel tree, or OVERWRITE=1 to replace." >&2
      exit 1
    fi
    echo "WARNING: OVERWRITE=1 -- replacing $what at $d" >&2
  fi
}

source /shared/software/miniconda3/latest/etc/profile.d/conda.sh
conda activate eptrans_ml
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "[15] RUN_TAG=$RUN_TAG  LAMS=$LAMS  host=$(hostname)  start=$(date -Is)"

# ---- combined corpus FASTA (secretome + whole proteome, deduped by header) ----
# The scoped corpus spans both scopes; embedding + adapter need sequences for every
# tagged_id. A whole-scope genome's secreted proteins appear in BOTH source FASTAs,
# so dedup on the header id (keep first). awk streams it in one pass, no full load.
if [ ! -s "$CORPUS_FASTA" ]; then
  echo "[15] building $CORPUS_FASTA from secretome.faa + wholeproteome.faa"
  awk '/^>/{id=substr($1,2); if(seen[id]++){skip=1}else{skip=0}} {if(!skip)print}' \
    "$ASM/secretome.faa" "$ASM/wholeproteome.faa" > "$CORPUS_FASTA.tmp"
  mv "$CORPUS_FASTA.tmp" "$CORPUS_FASTA"
  echo "[15] corpus FASTA seqs: $(grep -c '^>' "$CORPUS_FASTA")"
fi

CONTACTS=$MODEL_ROOT/contact_pairs.parquet

# ============================================================================
# stage 00 -- extremophile-only MLM subsample + precomputed contact maps
#   The subsample (~400k train / 20k val, cluster-stratified, mesophiles dropped)
#   is the only set that needs contacts, so precompute runs on it, not the 18M
#   corpus. --resume makes it idempotent per id.
# ============================================================================
if [ ! -f "$MLM_SUB" ]; then
  $PY "$REPO/scripts/09_subsample_mlm.py" \
    --labeled "$LABELED" --out "$MLM_SUB" --n-train 400000 --n-val 20000 --seed 1466
  rc=$?; [ $rc -ne 0 ] && { echo "[15] subsample FAILED rc=$rc"; exit $rc; }
fi
echo "[15] MLM subsample ready: $MLM_SUB"

if [ -f "$MODEL_ROOT/.contacts_done" ]; then
  echo "[15] 00 contacts already done -- skipping"
else
  $PY "$REPO/scripts/10_precompute_contacts.py" \
    --labeled "$MLM_SUB" --fasta "$CORPUS_FASTA" --out "$CONTACTS" \
    --backbone-size 3B --threshold 0.5 --min-sep 6 --top-k 128 \
    --max-len 1022 --device cuda --resume
  rc=$?; [ $rc -ne 0 ] && { echo "[15] 00 contacts FAILED rc=$rc"; exit $rc; }
  touch "$MODEL_ROOT/.contacts_done"
fi
echo "[15] 00 done  t=$(date -Is)"

# ============================================================================
# stage 08 -- MLM adapter (extremophile-only, coupling-aware masking from cache)
# ============================================================================
if [ -f "$MODEL_ROOT/mlm_adapt/.done" ]; then
  echo "[15] 08 MLM adapter already done -- skipping"
else
  refuse_clobber "$MODEL_ROOT/mlm_adapt" "an MLM adapter"
  $PY "$REPO/scripts/08_train_backbone.py" mlm \
    --labeled "$MLM_SUB" --fasta "$CORPUS_FASTA" \
    --backbone-size 3B --lora-rank 32 --lora-alpha 64 \
    --epochs "$MLM_EPOCHS" --lr 1e-4 --mask-rate 0.15 --gamma 1.0 \
    --coupling-mode contact --contact-pairs "$CONTACTS" \
    --contact-threshold 0.5 --contact-min-sep 6 --span-len 3 \
    --batch-size 16 --max-len 1022 --device cuda \
    --out-dir "$MODEL_ROOT/mlm_adapt"
  rc=$?; [ $rc -ne 0 ] && { echo "[15] 08 FAILED rc=$rc"; exit $rc; }
  touch "$MODEL_ROOT/mlm_adapt/.done"
fi
echo "[15] 08 done  t=$(date -Is)"

# ============================================================================
# stage 09 -- embedding cache (whole corpus, MLM-adapted backbone, frozen)
#   single-GPU: loop the 8 shards in-process so it all fits one allocation.
#   load_mlm_adapter_into_classifier RAISES if 0 LoRA tensors transfer, so a
#   silent vanilla-ESM cache is impossible.
# ============================================================================
if [ -f "$EMB_ROOT/secretome_scoped/.done" ]; then
  echo "[15] 09 embedding cache already done -- skipping"
else
  refuse_clobber "$EMB_ROOT/secretome_scoped" "an embedding cache"
  for sh in 0 1 2 3 4 5 6 7; do
    if [ -f "$EMB_ROOT/secretome_scoped/.done_shard${sh}" ]; then
      echo "[15] 09 shard $sh already done"; continue
    fi
    $PY "$REPO/scripts/09_embed_secretome.py" \
      --labeled "$LABELED" --fasta "$CORPUS_FASTA" \
      --mlm-adapter "$MODEL_ROOT/mlm_adapt/mlm_adapter_best" \
      --backbone-size 3B --lora-rank 32 --lora-alpha 64 \
      --shard "$sh" --nshards 8 \
      --batch-size 8 --max-len 1022 --device cuda \
      --out-dir "$EMB_ROOT/secretome_scoped"
    rc=$?; [ $rc -ne 0 ] && { echo "[15] 09 shard $sh FAILED rc=$rc"; exit $rc; }
  done
  touch "$EMB_ROOT/secretome_scoped/.done"
fi
echo "[15] 09 done  t=$(date -Is)"

# ============================================================================
# stage 10 -- classifier heads, LAMBDA SWEEP
#   For each lam, train ALL phenotypes on the cached embeddings into a per-lam
#   out-root. An epoch is seconds (head-only on frozen vectors), so the whole
#   grid is minutes. Records val_auprc (ranking-free precision) AND val_pair_auc
#   (anti-taxonomy ranking) per (pheno, lam) in metrics.json.
# ============================================================================
for lam in $LAMS; do
  OUT="$MODEL_ROOT/cached_probes_lam${lam}"
  if [ -f "$OUT/.done" ]; then echo "[15] 10 lam=$lam already done"; continue; fi
  echo "[15] 10 sweep lam=$lam -> $OUT"
  $PY "$REPO/scripts/10_train_cached_probe.py" \
    --cache-dir "$EMB_ROOT/secretome_scoped" \
    --labeled "$LABELED" --pairs "$PAIRS" --fasta "$CORPUS_FASTA" \
    --phenotypes thermophile hyperthermophile psychrophile halophile acidophile alkaliphile \
    --epochs "$CLF_EPOCHS" --lr 1e-3 --batch-size 4096 \
    --lam "$lam" --margin 1.0 \
    --device cuda --out-root "$OUT"
  rc=$?; [ $rc -ne 0 ] && { echo "[15] 10 lam=$lam FAILED rc=$rc"; exit $rc; }
  touch "$OUT/.done"
done
echo "[15] 10 sweep done  t=$(date -Is)"
echo "[15] ALL STAGES COMPLETE  end=$(date -Is)"
