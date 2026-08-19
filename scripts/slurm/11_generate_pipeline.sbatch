#!/bin/bash
#SBATCH --job-name=eptrans_gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/groups/cress/projects/jaymin/eptrans_scratch/logs/gen_%j.log
#
# End-to-end generation pipeline (design doc: docs/generation_pipeline.md).
# ONE job chains all stages across three conda envs, passing data through files in
# the per-job workdir. GPU nodes have NO internet, so MSA/weights are LOCAL only.
#
# Args (env vars, set by SlurmBackend.submit or a manual run):
#   GEN_JOBDIR    per-job dir; must contain input.fasta (one seq, >header\nSEQ)
#   GEN_PHENOS    space-separated phenotypes (e.g. "thermophile halophile")
#   GEN_NDESIGNS  designs per phenotype (default 3)
#   GEN_GIBBS     gibbs iters per design (default 24)
#   GEN_MAX_MUT_FRAC  hard mutation budget, fraction of length (default 0.25)
#   GEN_RMSD_CAP  hard full active-site CA-RMSD cap in A (default 2.0)
#   GEN_MH_T0     initial Metropolis temperature (default 0.05; 0 = strict
#                 hill-climbing, i.e. the pre-annealing behaviour)
#   GEN_MH_T1     final Metropolis temperature (default 0.005)
#   GEN_COUPLING  mask unit mode none|contact|span|both (default both, which is
#                 what the production MLM adapter was TRAINED with -- changing it
#                 reintroduces a train/inference masking mismatch)
#   GEN_REFOLD_RECOVER        mask_rate recovery factor after sustained refold
#                 success (default 1.25; 1.0 disables)
#   GEN_REFOLD_RECOVER_AFTER  consecutive passing checkpoints before recovery
#                 (default 2)
#   GEN_CORE_RMSD_CAP  hard catalytic-core CA-RMSD cap in A (default 1.5)
set -euo pipefail

PERSIST=/groups/cress/projects/jaymin/IS1111/eptrans
REPO=${REPO:-/groups/cress/projects/jaymin/eptrans_scratch/repo}
SW=/groups/cress/projects/jaymin/eptrans_scratch/software
export HF_HOME=$PERSIST/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source /shared/software/miniconda3/latest/etc/profile.d/conda.sh

JOBDIR=${GEN_JOBDIR:?set GEN_JOBDIR}
PHENOS=${GEN_PHENOS:-thermophile}
NDES=${GEN_NDESIGNS:-3}
GIBBS=${GEN_GIBBS:-24}
MAX_MUT_FRAC=${GEN_MAX_MUT_FRAC:-0.25}   # hard mutation budget (fraction of length)
RMSD_CAP=${GEN_RMSD_CAP:-2.0}            # hard full active-site CA-RMSD cap (A)
CORE_RMSD_CAP=${GEN_CORE_RMSD_CAP:-1.5}  # hard catalytic-core CA-RMSD cap (A)
REFOLD=${GEN_REFOLD:-1}                  # in-loop periodic ESMFold refold + rollback (1=on)
REFOLD_EVERY=${GEN_REFOLD_EVERY:-4}     # refold every N Gibbs passes
MH_T0=${GEN_MH_T0:-0.05}                 # Metropolis start temp (0 = hill-climbing)
MH_T1=${GEN_MH_T1:-0.005}                # Metropolis end temp (geometric anneal)
COUPLING=${GEN_COUPLING:-both}           # mask units; 'both' matches the trained adapter
REFOLD_RECOVER=${GEN_REFOLD_RECOVER:-1.25}          # mask_rate recovery factor
REFOLD_RECOVER_AFTER=${GEN_REFOLD_RECOVER_AFTER:-2} # passes before recovery
STRUCT=$JOBDIR/structures
mkdir -p "$STRUCT"
cd "$JOBDIR"

SEQ=$(awk '!/^>/{printf "%s",$0}' input.fasta)
echo "[gen] seq len=${#SEQ}  phenos=$PHENOS  n_designs=$NDES  gibbs=$GIBBS"

MLM_ADAPTER=${GEN_MLM_ADAPTER:-$PERSIST/runs/mhk32/mlm_adapt/mlm_adapter_best}
# MSA target DB for `mmseqs easy-search` conservation. Use the CLUSTERED UniRef50
# FASTA (non-redundant, ~24GB) — NOT the full uniprot_kb (90GB), which OOM'd a
# single-query search. 11_generate passes --split-memory-limit as a guardrail either
# way. (/shared/db/uniclust/30_2020_06 is an HH-suite DB for hhblits, not mmseqs.)
UNIREF=${GEN_MSA_DB:-/shared/db/uniref/uniref50/latest/uniref50.fasta}
# Production mhk32 heads at $PERSIST/runs/mhk32/heads_meanpool/clf_<pheno>/head_best.pt.
# Legacy cached_probes at $PERSIST/models/cached_probes/clf_<pheno>_cached/head_best.pt
# is available by setting GEN_HEAD_LAYOUT=cached_probes.
HEADDIR=${GEN_HEADDIR:-$PERSIST/runs/mhk32/heads_meanpool}
HEAD_LAYOUT=${GEN_HEAD_LAYOUT:-mhk32}   # mhk32 -> clf_<PH>/head_best.pt; cached_probes -> clf_<PH>_cached/head_best.pt

# ---- §16b interface-constraint knobs (optional; only wired if set) ----
# GEN_ADDITIONAL_CONSTRAINTS   comma-separated explicit tokens and/or NL phrases.
#                              Explicit tokens (contain a digit) go straight to frozen[].
#                              NL phrases (e.g. "tetramer interfaces, protein-bRNA interface")
#                              require GEN_COMPLEX_CIF and are resolved by biopython from
#                              partner-chain contact geometry.
# GEN_COMPLEX_CIF              path to the tetramer/complex CIF (required if any NL phrase)
# GEN_DESIGN_CHAIN             chain being redesigned in the complex (default "A")
# GEN_INTERFACE_CONTACT_CUTOFF heavy-atom cutoff for contact-residue selection (default 4.5)
# GEN_INTERFACE_RMSD_CAP       per-face CA-RMSD cap in A after periodic refold (default 1.5;
#                              0 disables per-face RMSD tracking)
# GEN_INTERFACES_JSON_OUT      optional audit sidecar with resolved residues per face
ADD_CONS=${GEN_ADDITIONAL_CONSTRAINTS:-}
COMPLEX_CIF=${GEN_COMPLEX_CIF:-}
DESIGN_CHAIN=${GEN_DESIGN_CHAIN:-A}
IFACE_CUTOFF=${GEN_INTERFACE_CONTACT_CUTOFF:-4.5}
IFACE_CAP=${GEN_INTERFACE_RMSD_CAP:-1.5}
IFACES_JSON=${GEN_INTERFACES_JSON_OUT:-}
IFACE_ARGS=()
if [ -n "$ADD_CONS" ]; then
  IFACE_ARGS+=("--additional-constraints" "$ADD_CONS")
  [ -n "$COMPLEX_CIF" ]  && IFACE_ARGS+=("--complex-cif" "$COMPLEX_CIF")
  IFACE_ARGS+=("--design-chain" "$DESIGN_CHAIN")
  IFACE_ARGS+=("--interface-contact-cutoff" "$IFACE_CUTOFF")
  IFACE_ARGS+=("--interface-rmsd-cap" "$IFACE_CAP")
fi

CAND_LIST=(); FOLD_LIST=()
ANN_DIR=${GEN_ANN_DIR:-$SW/annotations}
FOLDSEEK_AF=${GEN_FOLDSEEK_AF:-/shared/db/foldseek/latest/db/alphafold_uniprot}
SPROT_DB=${GEN_SPROT_DB:-$ANN_DIR/swissprot_mmseqs/sprot}
# Structural foldseek-vs-AF channel is OFF by default: it returns mostly UNREVIEWED
# TrEMBL neighbours (no site annotations) and is slow vs the 645GB DB. Set GEN_USE_FOLDSEEK=1
# to enable it as a recall booster.
USE_FOLDSEEK=${GEN_USE_FOLDSEEK:-0}

# ---------- Stage 0: fold WT ONCE (esmfold) — annotation needs the structure ----------
conda activate esmfold
echo "[gen] === fold WT (for annotation transfer) ==="
python "$REPO/scripts/11b_fold_rmsd.py" --wt-only "$SEQ" --structures "$STRUCT" --max-len 1000
conda deactivate

# ---------- Stage 3: annotation transfer (eptrans_ml: mmseqs+foldseek on PATH) ----------
# PRIMARY: mmseqs WT-seq vs reviewed Swiss-Prot DB -> transfer ACT_SITE/BINDING/SITE
# (+ M-CSA catalytic) in UniProt coord space. Foldseek-vs-AF optional recall booster.
# Skips gracefully (empty transfer) if annotation tables aren't staged.
conda activate eptrans_ml
if [ -f "$ANN_DIR/swissprot_sites.tsv.gz" ] || [ -f "$ANN_DIR/mcsa_catalytic_residues.json" ]; then
  echo "[gen] === Stage 3: annotation transfer (Swiss-Prot seq + M-CSA) ==="
  FS_FLAG="--no-foldseek"; [ "$USE_FOLDSEEK" = "1" ] && FS_FLAG=""
  python "$REPO/scripts/11a_annotate.py" \
    --wt-pdb "$STRUCT/wt.pdb" --wt-seq "$SEQ" \
    --out "$JOBDIR/active_site_transfer.json" \
    --ann-dir "$ANN_DIR" --swissprot-db "$SPROT_DB" \
    --foldseek-af-db "$FOLDSEEK_AF" $FS_FLAG \
    --workdir "$JOBDIR/_annot" --seq-len "${#SEQ}" || \
    echo '{"transferred":[]}' > "$JOBDIR/active_site_transfer.json"
else
  echo "[gen] annotation tables not staged in $ANN_DIR; skipping Stage 3 transfer"
  echo '{"transferred":[]}' > "$JOBDIR/active_site_transfer.json"
fi

# ---------- optional: persistent ESMFold refold worker (esmfold env, backgrounded) ----------
# Loads ESMFold ONCE and folds sequences the generation loop hands it via a queue dir,
# enabling in-loop structural rollback. Shares the GPU with ESM-2 3B (fits on H200).
REFOLD_WD=""; REFOLD_ARGS=""; WORKER_PID=""
if [ "$REFOLD" = "1" ] && [ -f "$STRUCT/wt.pdb" ]; then
  REFOLD_WD="$JOBDIR/_refold"; mkdir -p "$REFOLD_WD"
  rm -f "$REFOLD_WD/READY" "$REFOLD_WD/STOP"
  echo "[gen] launching ESMFold refold worker (esmfold env) -> $REFOLD_WD"
  conda run -n esmfold python "$REPO/scripts/11e_esmfold_worker.py" \
    --workdir "$REFOLD_WD" --max-len 1000 > "$JOBDIR/_refold_worker.log" 2>&1 &
  WORKER_PID=$!
  REFOLD_ARGS="--refold-workdir $REFOLD_WD --wt-pdb $STRUCT/wt.pdb --refold-every $REFOLD_EVERY --refold-rmsd-cap $RMSD_CAP"
fi

# ---------- Stages 1,2,4,5 per phenotype (eptrans_ml) ----------
for PH in $PHENOS; do
  echo "[gen] === $PH: masked-gen + scoring ==="
  case "$HEAD_LAYOUT" in
    mhk32)         HEAD_PATH="$HEADDIR/clf_${PH}/head_best.pt" ;;
    cached_probes) HEAD_PATH="$HEADDIR/clf_${PH}_cached/head_best.pt" ;;
    *)             echo "[gen] unknown GEN_HEAD_LAYOUT=$HEAD_LAYOUT" >&2; exit 2 ;;
  esac
  echo "[gen] head: $HEAD_PATH"
  # per-phenotype interfaces audit sidecar (only in NL-phrase mode)
  IF_ARGS_PH=("${IFACE_ARGS[@]}")
  if [ -n "$ADD_CONS" ]; then
    IF_ARGS_PH+=("--interfaces-json-out" "${IFACES_JSON:-$JOBDIR/interfaces_${PH}.json}")
  fi
  python "$REPO/scripts/11_generate.py" \
    --seq "$SEQ" --phenotype "$PH" \
    --out "$JOBDIR/candidates_${PH}.json" \
    --mlm-adapter "$MLM_ADAPTER" \
    --head "$HEAD_PATH" \
    --uniref-db "$UNIREF" \
    --transfer-json "$JOBDIR/active_site_transfer.json" \
    --backbone-size 3B --n-designs "$NDES" --gibbs-iters "$GIBBS" \
    --max-mut-frac "$MAX_MUT_FRAC" $REFOLD_ARGS \
    --mh-t0 "$MH_T0" --mh-t1 "$MH_T1" \
    --coupling-mode "$COUPLING" \
    --refold-recover "$REFOLD_RECOVER" \
    --refold-recover-after "$REFOLD_RECOVER_AFTER" \
    "${IF_ARGS_PH[@]}" \
    --device cuda --workdir "$JOBDIR/_work_${PH}" --seed 1466
  CAND_LIST+=("$JOBDIR/candidates_${PH}.json")
done
# stop the refold worker (fold any pending, then exit)
if [ -n "$WORKER_PID" ]; then
  touch "$REFOLD_WD/STOP"; wait "$WORKER_PID" 2>/dev/null || true
  echo "[gen] refold worker stopped"
fi
conda deactivate

# ---------- Stage 6b/6c fold designs + RMSD per phenotype (esmfold; WT reused) ----------
conda activate esmfold
for PH in $PHENOS; do
  echo "[gen] === $PH: ESMFold refold designs + active-site RMSD ==="
  python "$REPO/scripts/11b_fold_rmsd.py" \
    --candidates "$JOBDIR/candidates_${PH}.json" \
    --structures "$STRUCT" \
    --out "$JOBDIR/folded_${PH}.json" --max-len 1000 \
    --rmsd-cap "$RMSD_CAP" --core-rmsd-cap "$CORE_RMSD_CAP"
  FOLD_LIST+=("$JOBDIR/folded_${PH}.json")
done
conda deactivate

# ---------- Stage 6a MPNN gate on WT backbone (ligandmpnn) ----------
conda activate ligandmpnn
echo "[gen] === MPNN structural gate (auto protein/ligand) ==="
python "$REPO/scripts/11c_mpnn_gate.py" \
  --wt-pdb "$STRUCT/wt.pdb" --repo "$SW/LigandMPNN" \
  --out "$JOBDIR/mpnn.json" --workdir "$JOBDIR" --seed 1 || \
  echo '{"model_type":"none","wt_mpnn_confidence":null}' > "$JOBDIR/mpnn.json"
conda deactivate

# ---------- Assemble results.json (stdlib) ----------
conda activate eptrans_ml
echo "[gen] === assemble results.json ==="
python "$REPO/scripts/11d_assemble.py" \
  --candidates "${CAND_LIST[@]}" --folded "${FOLD_LIST[@]}" \
  --mpnn "$JOBDIR/mpnn.json" --out "$JOBDIR/results.json"
conda deactivate

echo "[gen] DONE -> $JOBDIR/results.json"
ls -lh "$JOBDIR/results.json" "$STRUCT"/*.pdb | head
