#!/bin/bash
#SBATCH --job-name=is621_genBv2
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/groups/cress/projects/jaymin/eptrans_scratch/logs/is621_genBv2_%j.log
# H200 large-mem + expandable_segments (mirrors 11_generate_is621.sbatch).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#
# IS621 (PDB 8WT6, chain A) redesign wrapper — Pipeline B v2.
# Differences from v1 (11_generate_is621_pipelineB.sbatch):
#   (a) Budget: 8 rounds × 12/round = 96 proposals (v1: 4×6=24, ~4× more sampling)
#   (b) Survivor-conditioned MPNN-in-the-loop: after 2 iid warmup rounds, each
#       later round biases MPNN's per-residue AA distribution toward consensus
#       AAs seen in the top-8 survivors so far (log-odds +5.0, min 3/8 agree).
#       Turns v1's iid rejection-sampling loop into an actual hill-climb.
#
# Runs alongside v1 (different output dir tag is621Bv2_*, different jobname).
#
# Submit one job per phenotype:
#   sbatch --export=ALL,GEN_PHENOS=thermophile      scripts/slurm/11_generate_is621_pipelineB_v2.sbatch
#   sbatch --export=ALL,GEN_PHENOS=halophile        scripts/slurm/11_generate_is621_pipelineB_v2.sbatch
#   sbatch --export=ALL,GEN_PHENOS=hyperthermophile scripts/slurm/11_generate_is621_pipelineB_v2.sbatch
#   sbatch --export=ALL,GEN_PHENOS=alkaliphile      scripts/slurm/11_generate_is621_pipelineB_v2.sbatch
#
# What this wrapper hard-codes (identical to Pipeline A's IS621 wrapper):
#   - chain A sequence of 8WT6 (306 residues)
#   - 8WT6 complex CIF for §16b interface resolution
#   - active-site cap 1.0 A, per-face cap 1.5 A
#   - mhk32 production adapter and heads
#   - GEN_ADDITIONAL_CONSTRAINTS = "tetramer interfaces, protein-bRNA interface, protein-target interface"
#
# Pipeline-B specific (overrideable via GEN_B_*):
#   GEN_B_ROUNDS=4               MPNN proposal rounds
#   GEN_B_BATCH=6                candidates per round
#   GEN_B_TEMPERATURES=0.05,0.10,0.20   annealed MPNN temps (cycled by round)
#   GEN_B_MPNN_MODEL=ligand_mpnn
#   GEN_NDESIGNS=3               final designs kept per phenotype (post-ranking)
set -euo pipefail

REPO=/groups/cress/projects/jaymin/eptrans_scratch/repo
PERSIST=/groups/cress/projects/jaymin/IS1111/eptrans
SCRATCH=/groups/cress/projects/jaymin/eptrans_scratch
SW=$SCRATCH/software

: "${GEN_PHENOS:?set GEN_PHENOS=thermophile|halophile|hyperthermophile|alkaliphile}"

STAMP=$(date +%Y%m%d_%H%M%S)
TAG=${GEN_TAG:-is621Bv2_${STAMP}_${GEN_PHENOS// /_}}
export GEN_JOBDIR=$SCRATCH/gen/$TAG
mkdir -p "$GEN_JOBDIR"

# --- IS621 chain A sequence (8WT6, resolved 4-321, 306 aa) ---
IS621_SEQ="ELHYIGIDTAKEKLDVDVLRPDGRHRTKKFANTTKGHDELVSWLKGHKIDHAHICIEATGTYMEPVAECLYDAGYIVSVINPALGKAFAQSEGLRNKTDTVDARMLAEFCRQKRPAAWEAPHPLERALRALVVRHQALTDMHTQELNRTETAREVQRPSIDAHLLWLEAELKRLEKQIKDLTDDDPDMKHRRKLLESIPGIGEKTSAVLLAYIGLKDRFAHARQFAAFAGLTPRRMSKAGHVSLRRALYMPAMVATSKTEWGRAFRDRLAANGKKGKVILGAMMRKLAQVAYGVLKSGVPFDASRH"
printf ">IS621_8WT6_chainA\n%s\n" "$IS621_SEQ" > "$GEN_JOBDIR/input.fasta"

# --- 8WT6 complex CIF (staged from RCSB earlier; re-download if missing) ---
export GEN_COMPLEX_CIF=$SCRATCH/8wt6.cif
if [ ! -s "$GEN_COMPLEX_CIF" ]; then
  curl -sSL -o "$GEN_COMPLEX_CIF" https://files.rcsb.org/download/8WT6.cif
fi

# --- §16b interface constraints ---
export GEN_ADDITIONAL_CONSTRAINTS="tetramer interfaces, protein-bRNA interface, protein-target interface"
export GEN_DESIGN_CHAIN=A
export GEN_INTERFACE_CONTACT_CUTOFF=4.5
export GEN_INTERFACE_RMSD_CAP=1.5

# --- §16b active-site cap ---
export GEN_CORE_RMSD_CAP=1.0

# --- mhk32 production adapter + heads ---
export GEN_MLM_ADAPTER=$PERSIST/runs/mhk32/mlm_adapt/mlm_adapter_best
export GEN_HEADDIR=$PERSIST/runs/mhk32/heads_meanpool
export GEN_HEAD_LAYOUT=mhk32

# --- Pipeline B knobs (v2: bigger budget + survivor-conditioned iteration) ---
# (a) Budget bump: 8 rounds × 12/round = 96 proposals (v1 was 4×6=24).
# (b) Survivor-conditioned hill-climb: after --survivor-warmup-rounds warmup
#     rounds of iid draws, each subsequent round biases MPNN's per-residue
#     distribution toward consensus AAs seen in the top-K survivors so far
#     (log-odds = --survivor-bias-strength, min --survivor-min-agreement/K).
export GEN_NDESIGNS=${GEN_NDESIGNS:-3}
export GEN_B_ROUNDS=${GEN_B_ROUNDS:-8}
export GEN_B_BATCH=${GEN_B_BATCH:-12}
export GEN_B_TEMPERATURES=${GEN_B_TEMPERATURES:-0.05,0.10,0.20}
# v2 survivor-iteration knobs (in v1 wrapper these default 0 = off)
export GEN_B_SURVIVOR_TOPK=${GEN_B_SURVIVOR_TOPK:-8}
export GEN_B_SURVIVOR_MIN_AGREE=${GEN_B_SURVIVOR_MIN_AGREE:-3}
export GEN_B_SURVIVOR_BIAS_STRENGTH=${GEN_B_SURVIVOR_BIAS_STRENGTH:-5.0}
export GEN_B_SURVIVOR_WARMUP=${GEN_B_SURVIVOR_WARMUP:-2}
export GEN_B_MPNN_MODEL=${GEN_B_MPNN_MODEL:-ligand_mpnn}
export GEN_B_MPNN_REPO=${GEN_B_MPNN_REPO:-$SW/LigandMPNN}
export GEN_B_MPNN_ENV=${GEN_B_MPNN_ENV:-ligandmpnn}
export GEN_B_BIAS_AA_JSON=${GEN_B_BIAS_AA_JSON:-$REPO/data/bias_aa_by_phenotype.json}
export GEN_B_CONS_FREEZE=${GEN_B_CONS_FREEZE:-0.90}
export GEN_COUPLING=${GEN_COUPLING:-both}

echo "[is621Bv2] jobdir=$GEN_JOBDIR"
echo "[is621Bv2] phenos=$GEN_PHENOS n_designs=$GEN_NDESIGNS rounds=$GEN_B_ROUNDS batch=$GEN_B_BATCH temps=$GEN_B_TEMPERATURES"
echo "[is621Bv2] survivor topk=$GEN_B_SURVIVOR_TOPK min_agree=$GEN_B_SURVIVOR_MIN_AGREE strength=$GEN_B_SURVIVOR_BIAS_STRENGTH warmup=$GEN_B_SURVIVOR_WARMUP"
echo "[is621B] core_cap=$GEN_CORE_RMSD_CAP iface_cap=$GEN_INTERFACE_RMSD_CAP conservation_freeze=$GEN_B_CONS_FREEZE"
echo "[is621B] additional_constraints='$GEN_ADDITIONAL_CONSTRAINTS'"
echo "[is621B] complex_cif=$GEN_COMPLEX_CIF"
echo "[is621B] mpnn: repo=$GEN_B_MPNN_REPO env=$GEN_B_MPNN_ENV model=$GEN_B_MPNN_MODEL"
echo "[is621B] bias_aa=$GEN_B_BIAS_AA_JSON"

# --- delegate to the generic pipeline B sbatch (in the SAME allocation) ---
exec bash "$REPO/scripts/slurm/11_generate_pipelineB_pipeline.sbatch"
