#!/bin/bash
#SBATCH --job-name=eptrans_gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
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
STRUCT=$JOBDIR/structures
mkdir -p "$STRUCT"
cd "$JOBDIR"

SEQ=$(awk '!/^>/{printf "%s",$0}' input.fasta)
echo "[gen] seq len=${#SEQ}  phenos=$PHENOS  n_designs=$NDES  gibbs=$GIBBS"

MLM_ADAPTER=$PERSIST/models/mlm_adapt/mlm_adapter_best
# MSA target DB: the mmseqs2-format UniProtKB sequence DB (for `mmseqs easy-search`).
# NOTE: /shared/db/uniclust/30_2020_06 is an HH-suite DB (_a3m/_cs219/_hhm .ffdata),
# usable by hhblits NOT mmseqs — despite config.yaml's "uniref30_mmseqs" label. The
# real mmseqs DBs are uniprot_kb (plain seq DB, used here) and the ColabFold envdb
# (profile DB, for the deeper ColabFold MSA protocol).
UNIREF=${GEN_MSA_DB:-/shared/db/uniprot/latest/mmseqs/uniprot_kb}
HEADDIR=$PERSIST/models/cached_probes

CAND_LIST=(); FOLD_LIST=()
ANN_DIR=${GEN_ANN_DIR:-$SW/annotations}
FOLDSEEK_AF=${GEN_FOLDSEEK_AF:-/shared/db/foldseek/latest/db/alphafold_uniprot}

# ---------- Stage 0: fold WT ONCE (esmfold) — annotation needs the structure ----------
conda activate esmfold
echo "[gen] === fold WT (for annotation transfer) ==="
python "$REPO/scripts/11b_fold_rmsd.py" --wt-only "$SEQ" --structures "$STRUCT" --max-len 1000
conda deactivate

# ---------- Stage 3: annotation transfer (eptrans_ml: foldseek on PATH) ----------
# foldseek WT vs local AlphaFold DB -> transfer M-CSA/Swiss-Prot catalytic residues.
# Skips gracefully (empty transfer) if annotation tables aren't staged.
conda activate eptrans_ml
if [ -f "$ANN_DIR/swissprot_sites.tsv.gz" ] || [ -f "$ANN_DIR/mcsa_catalytic_residues.json" ]; then
  echo "[gen] === Stage 3: annotation transfer (foldseek + M-CSA/Swiss-Prot) ==="
  python "$REPO/scripts/11a_annotate.py" \
    --wt-pdb "$STRUCT/wt.pdb" --out "$JOBDIR/active_site_transfer.json" \
    --ann-dir "$ANN_DIR" --foldseek-af-db "$FOLDSEEK_AF" \
    --workdir "$JOBDIR/_annot" --seq-len "${#SEQ}" || \
    echo '{"transferred":[]}' > "$JOBDIR/active_site_transfer.json"
else
  echo "[gen] annotation tables not staged in $ANN_DIR; skipping Stage 3 transfer"
  echo '{"transferred":[]}' > "$JOBDIR/active_site_transfer.json"
fi

# ---------- Stages 1,2,4,5 per phenotype (eptrans_ml) ----------
for PH in $PHENOS; do
  echo "[gen] === $PH: masked-gen + scoring ==="
  python "$REPO/scripts/11_generate.py" \
    --seq "$SEQ" --phenotype "$PH" \
    --out "$JOBDIR/candidates_${PH}.json" \
    --mlm-adapter "$MLM_ADAPTER" \
    --head "$HEADDIR/clf_${PH}_cached/head_best.pt" \
    --uniref-db "$UNIREF" \
    --transfer-json "$JOBDIR/active_site_transfer.json" \
    --backbone-size 3B --n-designs "$NDES" --gibbs-iters "$GIBBS" \
    --device cuda --workdir "$JOBDIR/_work_${PH}" --seed 1466
  CAND_LIST+=("$JOBDIR/candidates_${PH}.json")
done
conda deactivate

# ---------- Stage 6b/6c fold designs + RMSD per phenotype (esmfold; WT reused) ----------
conda activate esmfold
for PH in $PHENOS; do
  echo "[gen] === $PH: ESMFold refold designs + active-site RMSD ==="
  python "$REPO/scripts/11b_fold_rmsd.py" \
    --candidates "$JOBDIR/candidates_${PH}.json" \
    --structures "$STRUCT" \
    --out "$JOBDIR/folded_${PH}.json" --max-len 1000
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
