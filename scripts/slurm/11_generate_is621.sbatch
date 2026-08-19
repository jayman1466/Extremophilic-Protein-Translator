#!/bin/bash
#SBATCH --job-name=is621_gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/groups/cress/projects/jaymin/eptrans_scratch/logs/is621_gen_%j.log
#
# IS621 (PDB 8WT6, chain A) redesign wrapper.
#
# Two campaigns (submit as separate jobs — one --gres GPU each):
#   sbatch --export=ALL,GEN_PHENOS=thermophile scripts/slurm/11_generate_is621.sbatch
#   sbatch --export=ALL,GEN_PHENOS=halophile   scripts/slurm/11_generate_is621.sbatch
#
# What this wrapper hard-codes for IS621 (all overrideable via GEN_*):
#   - chain A sequence of 8WT6 (306 residues, extracted 2026-08-19)
#   - 8WT6 complex CIF for interface resolution
#   - §16b interface phrases: tetramer + bRNA + target DNA (A:donor is empty
#     in the pre-strand-exchange 8WT6 complex; A:C tetramer diagonal is empty)
#   - active-site cap 1.0 A, per-face cap 1.5 A (§16b spec)
#   - mhk32 production adapter and heads
#
# Non-obvious knobs, all overrideable:
#   GEN_NDESIGNS  designs per phenotype (default 3)
#   GEN_GIBBS     gibbs iters per design (default 24)
#   GEN_TAG       jobdir tag (default is621_<timestamp>_<phenos>)
set -euo pipefail

REPO=/groups/cress/projects/jaymin/eptrans_scratch/repo
PERSIST=/groups/cress/projects/jaymin/IS1111/eptrans
SCRATCH=/groups/cress/projects/jaymin/eptrans_scratch

: "${GEN_PHENOS:?set GEN_PHENOS=thermophile or halophile (or both, space-separated)}"

STAMP=$(date +%Y%m%d_%H%M%S)
TAG=${GEN_TAG:-is621_${STAMP}_${GEN_PHENOS// /_}}
export GEN_JOBDIR=$SCRATCH/gen/$TAG
mkdir -p "$GEN_JOBDIR"

# --- IS621 chain A sequence (8WT6, resolved 4-321, 306 aa) ---
IS621_SEQ="ELHYIGIDTAKEKLDVDVLRPDGRHRTKKFANTTKGHDELVSWLKGHKIDHAHICIEATGTYMEPVAECLYDAGYIVSVINPALGKAFAQSEGLRNKTDTVDARMLAEFCRQKRPAAWEAPHPLERALRALVVRHQALTDMHTQELNRTETAREVQRPSIDAHLLWLEAELKRLEKQIKDLTDDDPDMKHRRKLLESIPGIGEKTSAVLLAYIGLKDRFAHARQFAAFAGLTPRRMSKAGHVSLRRALYMPAMVATSKTEWGRAFRDRLAANGKKGKVILGAMMRKLAQVAYGVLKSGVPFDASRH"
printf ">IS621_8WT6_chainA\n%s\n" "$IS621_SEQ" > "$GEN_JOBDIR/input.fasta"

# --- 8WT6 complex CIF (staged from RCSB earlier) ---
export GEN_COMPLEX_CIF=$SCRATCH/8wt6.cif
if [ ! -s "$GEN_COMPLEX_CIF" ]; then
  curl -sSL -o "$GEN_COMPLEX_CIF" https://files.rcsb.org/download/8WT6.cif
fi

# --- §16b interface constraints ---
export GEN_ADDITIONAL_CONSTRAINTS="tetramer interfaces, protein-bRNA interface, protein-target interface"
export GEN_DESIGN_CHAIN=A
export GEN_INTERFACE_CONTACT_CUTOFF=4.5
export GEN_INTERFACE_RMSD_CAP=1.5
# (per-phenotype interfaces_<PH>.json audit auto-created inside $GEN_JOBDIR)

# --- §16b active-site cap ---
export GEN_CORE_RMSD_CAP=1.0
export GEN_RMSD_CAP=${GEN_RMSD_CAP:-2.0}

# --- mhk32 production adapter + heads (default in the pipeline sbatch) ---
export GEN_MLM_ADAPTER=$PERSIST/runs/mhk32/mlm_adapt/mlm_adapter_best
export GEN_HEADDIR=$PERSIST/runs/mhk32/heads_meanpool
export GEN_HEAD_LAYOUT=mhk32

export GEN_NDESIGNS=${GEN_NDESIGNS:-3}
export GEN_GIBBS=${GEN_GIBBS:-24}
export GEN_MH_T0=${GEN_MH_T0:-0.05}
export GEN_MH_T1=${GEN_MH_T1:-0.005}
export GEN_COUPLING=${GEN_COUPLING:-both}

echo "[is621] jobdir=$GEN_JOBDIR"
echo "[is621] phenos=$GEN_PHENOS n_designs=$GEN_NDESIGNS gibbs=$GEN_GIBBS"
echo "[is621] core_cap=$GEN_CORE_RMSD_CAP full_cap=$GEN_RMSD_CAP iface_cap=$GEN_INTERFACE_RMSD_CAP"
echo "[is621] additional_constraints='$GEN_ADDITIONAL_CONSTRAINTS'"
echo "[is621] complex_cif=$GEN_COMPLEX_CIF"

# --- delegate to the generic pipeline sbatch (in the SAME allocation) ---
exec bash "$REPO/scripts/slurm/11_generate_pipeline.sbatch"
