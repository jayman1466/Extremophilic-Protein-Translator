#!/bin/bash
#SBATCH --job-name=eptrans_genB
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/groups/cress/projects/jaymin/eptrans_scratch/logs/genB_%j.log
#
# Pipeline B (MPNN-in-the-loop, PLM filter) — generic backend.
# Mirrors 11_generate_pipeline.sbatch (Pipeline A) but calls 11_generate_pipelineB.py,
# which does its OWN MPNN proposal + in-loop refold gate. So this wrapper is thinner:
# it still folds WT once and runs annotation transfer, launches the ESMFold worker for
# the in-loop RefoldClient, then invokes 11_generate_pipelineB.py per phenotype.
#
# H200 partition + expandable_segments to avoid the shared-A5000 fragmentation OOMs
# hit on A. Same job-level env-var contract as Pipeline A where possible; B-specific
# knobs are the GEN_B_* / GEN_MPNN_* / GEN_BIAS_AA_JSON.
#
# Args (env vars, set by caller or manual submit):
#   GEN_JOBDIR              per-job dir; must contain input.fasta (>header\nSEQ)
#   GEN_PHENOS              space-separated phenotypes
#   GEN_NDESIGNS            designs per phenotype AFTER ranking (default 3)
#   GEN_B_ROUNDS            MPNN proposal rounds (default 4)
#   GEN_B_BATCH             candidates per round per phenotype (default 6)
#   GEN_B_TEMPERATURES      comma-separated MPNN sampling temps (default 0.05,0.10,0.20)
#   GEN_B_MPNN_REPO         path to LigandMPNN clone (default $SW/LigandMPNN)
#   GEN_B_MPNN_ENV          conda env with LigandMPNN deps (default ligandmpnn)
#   GEN_B_MPNN_MODEL        ligand_mpnn or protein_mpnn (default ligand_mpnn)
#   GEN_B_BIAS_AA_JSON      per-phenotype bias_AA table (default $REPO/data/bias_aa_by_phenotype.json)
#   GEN_B_CONS_FREEZE       conservation frozen cutoff (default 0.90)
#   GEN_CORE_RMSD_CAP       catalytic-core CA-RMSD cap (default 1.0 A)
#   GEN_INTERFACE_RMSD_CAP  per-face CA-RMSD cap (default 1.5 A)
#   GEN_ADDITIONAL_CONSTRAINTS  NL phrases (e.g. "tetramer interfaces, protein-bRNA interface")
#   GEN_COMPLEX_CIF         complex CIF for interface resolution
#   GEN_DESIGN_CHAIN        chain being redesigned (default A)
#   GEN_INTERFACE_CONTACT_CUTOFF   heavy-atom cutoff (default 4.5 A)
#   GEN_INTERFACES_JSON_OUT audit sidecar (default $JOBDIR/interfaces_<PH>.json)
#   GEN_MLM_ADAPTER         PLM LoRA adapter dir (default mhk32)
#   GEN_HEADDIR / GEN_HEAD_LAYOUT  same convention as Pipeline A
#   GEN_COUPLING            mask-unit mode for scoring (default both)
set -euo pipefail

# H200 late-run allocator smoothing (matched to Pipeline A wrapper)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PERSIST=/groups/cress/projects/jaymin/IS1111/eptrans
REPO=${REPO:-/groups/cress/projects/jaymin/eptrans_scratch/repo}
SW=/groups/cress/projects/jaymin/eptrans_scratch/software
export HF_HOME=$PERSIST/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source /shared/software/miniconda3/latest/etc/profile.d/conda.sh

JOBDIR=${GEN_JOBDIR:?set GEN_JOBDIR}
PHENOS=${GEN_PHENOS:-thermophile}
NDES=${GEN_NDESIGNS:-3}
ROUNDS=${GEN_B_ROUNDS:-4}
BATCH=${GEN_B_BATCH:-6}
TEMPS=${GEN_B_TEMPERATURES:-0.05,0.10,0.20}
# Survivor-conditioned iteration (v2). Default 0 keeps legacy iid behavior.
SURVIVOR_TOPK=${GEN_B_SURVIVOR_TOPK:-0}
SURVIVOR_MIN_AGREE=${GEN_B_SURVIVOR_MIN_AGREE:-2}
SURVIVOR_BIAS_STRENGTH=${GEN_B_SURVIVOR_BIAS_STRENGTH:-5.0}
SURVIVOR_WARMUP=${GEN_B_SURVIVOR_WARMUP:-2}
MPNN_REPO=${GEN_B_MPNN_REPO:-$SW/LigandMPNN}
MPNN_ENV=${GEN_B_MPNN_ENV:-ligandmpnn}
MPNN_MODEL=${GEN_B_MPNN_MODEL:-ligand_mpnn}
BIAS_AA_JSON=${GEN_B_BIAS_AA_JSON:-$REPO/data/bias_aa_by_phenotype.json}
CONS_FREEZE=${GEN_B_CONS_FREEZE:-0.90}
CORE_CAP=${GEN_CORE_RMSD_CAP:-1.0}
IFACE_CAP=${GEN_INTERFACE_RMSD_CAP:-1.5}
IFACE_CUTOFF=${GEN_INTERFACE_CONTACT_CUTOFF:-4.5}
DESIGN_CHAIN=${GEN_DESIGN_CHAIN:-A}
COUPLING=${GEN_COUPLING:-both}
ADD_CONS=${GEN_ADDITIONAL_CONSTRAINTS:-}
COMPLEX_CIF=${GEN_COMPLEX_CIF:-}
IFACES_JSON=${GEN_INTERFACES_JSON_OUT:-}

MLM_ADAPTER=${GEN_MLM_ADAPTER:-$PERSIST/runs/mhk32/mlm_adapt/mlm_adapter_best}
UNIREF=${GEN_MSA_DB:-/shared/db/uniref/uniref50/latest/uniref50.fasta}
HEADDIR=${GEN_HEADDIR:-$PERSIST/runs/mhk32/heads_meanpool}
HEAD_LAYOUT=${GEN_HEAD_LAYOUT:-mhk32}

STRUCT=$JOBDIR/structures
mkdir -p "$STRUCT"
cd "$JOBDIR"

if [ ! -s "$BIAS_AA_JSON" ]; then
  echo "[genB] FATAL: bias_AA table not found: $BIAS_AA_JSON" >&2
  echo "[genB]   run scripts/prep_bias_aa.py first (or set GEN_B_BIAS_AA_JSON)." >&2
  exit 2
fi
if [ ! -d "$MPNN_REPO" ]; then
  echo "[genB] FATAL: MPNN repo not found: $MPNN_REPO" >&2
  exit 2
fi

SEQ=$(awk '!/^>/{printf "%s",$0}' input.fasta)
echo "[genB] seq len=${#SEQ}  phenos=$PHENOS  n_designs=$NDES"
echo "[genB] rounds=$ROUNDS  batch=$BATCH  temps=$TEMPS  mpnn=$MPNN_MODEL"
echo "[genB] core_cap=$CORE_CAP iface_cap=$IFACE_CAP  conservation_freeze=$CONS_FREEZE"

# ---------- Stage 0: fold WT ONCE (esmfold) — needed for annotation + interfaces + gate ----------
conda activate esmfold
echo "[genB] === fold WT ==="
python "$REPO/scripts/11b_fold_rmsd.py" --wt-only "$SEQ" --structures "$STRUCT" --max-len 1000
conda deactivate

# ---------- Stage 3: annotation transfer (identical to Pipeline A) ----------
ANN_DIR=${GEN_ANN_DIR:-$SW/annotations}
FOLDSEEK_AF=${GEN_FOLDSEEK_AF:-/shared/db/foldseek/latest/db/alphafold_uniprot}
SPROT_DB=${GEN_SPROT_DB:-$ANN_DIR/swissprot_mmseqs/sprot}
USE_FOLDSEEK=${GEN_USE_FOLDSEEK:-0}

conda activate eptrans_ml
if [ -f "$ANN_DIR/swissprot_sites.tsv.gz" ] || [ -f "$ANN_DIR/mcsa_catalytic_residues.json" ]; then
  echo "[genB] === Stage 3: annotation transfer ==="
  FS_FLAG="--no-foldseek"; [ "$USE_FOLDSEEK" = "1" ] && FS_FLAG=""
  python "$REPO/scripts/11a_annotate.py" \
    --wt-pdb "$STRUCT/wt.pdb" --wt-seq "$SEQ" \
    --out "$JOBDIR/active_site_transfer.json" \
    --ann-dir "$ANN_DIR" --swissprot-db "$SPROT_DB" \
    --foldseek-af-db "$FOLDSEEK_AF" $FS_FLAG \
    --workdir "$JOBDIR/_annot" --seq-len "${#SEQ}" || \
    echo '{"transferred":[]}' > "$JOBDIR/active_site_transfer.json"
else
  echo "[genB] annotation tables not staged; skipping"
  echo '{"transferred":[]}' > "$JOBDIR/active_site_transfer.json"
fi

# ---------- ESMFold refold worker (in-loop gate needs it) ----------
REFOLD_WD="$JOBDIR/_refold"; mkdir -p "$REFOLD_WD"
rm -f "$REFOLD_WD/READY" "$REFOLD_WD/STOP"
echo "[genB] launching ESMFold refold worker -> $REFOLD_WD"
conda run -n esmfold python "$REPO/scripts/11e_esmfold_worker.py" \
  --workdir "$REFOLD_WD" --max-len 1000 > "$JOBDIR/_refold_worker.log" 2>&1 &
WORKER_PID=$!

# ---------- Stage: Pipeline B generation per phenotype ----------
CAND_LIST=()
for PH in $PHENOS; do
  echo "[genB] === $PH: MPNN-propose + PLM-filter + fold-gate ==="
  case "$HEAD_LAYOUT" in
    mhk32)         HEAD_PATH="$HEADDIR/clf_${PH}/head_best.pt" ;;
    cached_probes) HEAD_PATH="$HEADDIR/clf_${PH}_cached/head_best.pt" ;;
    *)             echo "[genB] unknown GEN_HEAD_LAYOUT=$HEAD_LAYOUT" >&2; exit 2 ;;
  esac
  echo "[genB] head: $HEAD_PATH"

  IF_ARGS=()
  if [ -n "$ADD_CONS" ]; then
    IF_ARGS+=("--additional-constraints" "$ADD_CONS")
    [ -n "$COMPLEX_CIF" ] && IF_ARGS+=("--complex-cif" "$COMPLEX_CIF")
    IF_ARGS+=("--design-chain" "$DESIGN_CHAIN")
    IF_ARGS+=("--interface-contact-cutoff" "$IFACE_CUTOFF")
    IF_ARGS+=("--interface-rmsd-cap" "$IFACE_CAP")
    IF_ARGS+=("--interfaces-json-out" "${IFACES_JSON:-$JOBDIR/interfaces_${PH}.json}")
  fi

  python "$REPO/scripts/11_generate_pipelineB.py" \
    --seq "$SEQ" --phenotype "$PH" \
    --out "$JOBDIR/candidates_${PH}.json" \
    --mlm-adapter "$MLM_ADAPTER" \
    --head "$HEAD_PATH" \
    --uniref-db "$UNIREF" \
    --transfer-json "$JOBDIR/active_site_transfer.json" \
    --bias-aa-json "$BIAS_AA_JSON" \
    --backbone-size 3B \
    --mpnn-repo "$MPNN_REPO" \
    --mpnn-env "$MPNN_ENV" \
    --mpnn-model "$MPNN_MODEL" \
    --n-designs "$NDES" \
    --n-rounds "$ROUNDS" \
    --batch-per-round "$BATCH" \
    --temperatures "$TEMPS" \
    --survivor-topk "$SURVIVOR_TOPK" \
    --survivor-min-agreement "$SURVIVOR_MIN_AGREE" \
    --survivor-bias-strength "$SURVIVOR_BIAS_STRENGTH" \
    --survivor-warmup-rounds "$SURVIVOR_WARMUP" \
    --wt-pdb "$STRUCT/wt.pdb" \
    --conservation-freeze "$CONS_FREEZE" \
    --core-rmsd-cap "$CORE_CAP" \
    --refold-workdir "$REFOLD_WD" \
    --coupling-mode "$COUPLING" \
    "${IF_ARGS[@]}" \
    --device cuda --workdir "$JOBDIR/_work_${PH}" --seed 1466
  CAND_LIST+=("$JOBDIR/candidates_${PH}.json")
done

# stop refold worker
touch "$REFOLD_WD/STOP"; wait "$WORKER_PID" 2>/dev/null || true
echo "[genB] refold worker stopped"
conda deactivate

# ---------- Stage 6a: MPNN WT-backbone confidence (informational, same as A) ----------
conda activate ligandmpnn
echo "[genB] === MPNN structural gate (WT backbone confidence, informational) ==="
python "$REPO/scripts/11c_mpnn_gate.py" \
  --wt-pdb "$STRUCT/wt.pdb" --repo "$MPNN_REPO" \
  --out "$JOBDIR/mpnn.json" --workdir "$JOBDIR" --seed 1 || \
  echo '{"model_type":"none","wt_mpnn_confidence":null}' > "$JOBDIR/mpnn.json"
conda deactivate

# ---------- Assemble results.json ----------
# Pipeline B candidates carry per-design refold RMSDs from the in-loop fold gate,
# so no separate 11b_fold_rmsd pass is run. 11d_assemble falls back to the
# per-candidate RMSDs when the folded-list is omitted.
conda activate eptrans_ml
echo "[genB] === assemble results.json ==="
python "$REPO/scripts/11d_assemble.py" \
  --candidates "${CAND_LIST[@]}" \
  --mpnn "$JOBDIR/mpnn.json" --out "$JOBDIR/results.json" || \
  echo "[genB] WARN: 11d_assemble had trouble; candidates_*.json are canonical"
conda deactivate

echo "[genB] DONE -> $JOBDIR/results.json"
ls -lh "$JOBDIR"/candidates_*.json "$STRUCT"/*.pdb 2>/dev/null | head
