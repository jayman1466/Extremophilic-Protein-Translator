#!/bin/bash
#SBATCH --job-name=eptrans_cluster
#SBATCH --partition=standard
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/groups/cress/projects/jaymin/eptrans_scratch/logs/cluster_%j.log

set -euo pipefail
PERSIST=/groups/cress/projects/jaymin/IS1111/eptrans
SCRATCH=/groups/cress/projects/jaymin/eptrans_scratch
IN_FAA="$PERSIST/secreted_proteins_r232.faa"
OUT_PREFIX="$PERSIST/secreted_clusters_r232"
TMPDIR_MM="$SCRATCH/mmseqs_tmp"

bash "$SCRATCH/07_cluster_secreted.sh" \
    "$IN_FAA" "$OUT_PREFIX" "$TMPDIR_MM" 0.5 0.8 48

echo "CLUSTER EXIT $?"
ls -la "${OUT_PREFIX}"_cluster.tsv "${OUT_PREFIX}"_rep_seq.fasta
