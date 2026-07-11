#!/bin/bash
# Stage-1 MLM on a Jarvislabs H200 spot instance.
#
# Assumes a persistent Jarvis filesystem mounted at $FS (default /home/jl_fs)
# holding: the repo, the 400k subsample parquet, the mature-chain FASTA, and the
# ESM-2 3B weights (HF cache). Writes the adapter + step-checkpoints back to $FS,
# so a spot preemption is recovered by re-launching this same script (it resumes
# from $FS/models/mlm_adapt/mlm_ckpt.pt).
#
# Usage on the instance:  bash run_mlm.sh
set -euo pipefail
FS=${FS:-/home/jl_fs}
REPO=$FS/repo
export HF_HOME=$FS/hf_cache            # 3B weights live here (survives preemption)

cd "$REPO"
# venv built once on the filesystem; reused across preemption restarts
if [ ! -d "$FS/venv" ]; then
  python -m venv --system-site-packages "$FS/venv"
  "$FS/venv/bin/pip" install -q -r "$REPO/scripts/jarvis/requirements.txt"
fi
source "$FS/venv/bin/activate"

python "$REPO/scripts/08_train_backbone.py" mlm \
  --labeled "$FS/data/labeled_mlm_subsample.parquet" \
  --fasta   "$FS/data/mlm_subsample_mature.faa.gz" \
  --backbone-size 3B --lora-rank 32 --lora-alpha 64 \
  --epochs 3 --lr 1e-4 --mask-rate 0.15 --gamma 1.0 \
  --batch-size 16 --max-len 1022 --device cuda \
  --ckpt-every 200 \
  --out-dir "$FS/models/mlm_adapt"
echo "MLM EXIT $?"
