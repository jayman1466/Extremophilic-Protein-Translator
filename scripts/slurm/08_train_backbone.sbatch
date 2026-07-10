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

if [ "$MODE" = "mlm" ]; then
  python "$REPO/scripts/08_train_backbone.py" mlm \
    --labeled "$MLM_LABELED" --fasta "$FASTA" \
    --backbone-size 3B --lora-rank 32 --lora-alpha 64 \
    --epochs 3 --lr 1e-4 --mask-rate 0.15 --gamma 1.0 \
    --batch-size 16 --max-len 1022 --device cuda \
    --out-dir "$PERSIST/models/mlm_adapt"
else
  python "$REPO/scripts/08_train_backbone.py" classifier \
    --phenotype "$PHENO" \
    --labeled "$CLF_LABELED" --fasta "$FASTA" --pairs "$PAIRS" \
    --mlm-adapter "$PERSIST/models/mlm_adapt/mlm_adapter_best" \
    --backbone-size 3B --lora-rank 32 --lora-alpha 64 \
    --epochs 5 --lr-head 1e-3 --lr-adapter 1e-5 --lam 1.0 --margin 1.0 \
    --batch-size 16 --max-len 1022 --device cuda \
    --out-dir "$PERSIST/models/clf_${PHENO}"
fi
echo "TRAIN EXIT $?"
