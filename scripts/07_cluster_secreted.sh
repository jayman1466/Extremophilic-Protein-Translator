#!/bin/bash
# Stage 07 - cluster the secreted mature-chain proteins with mmseqs2 for
# leakage-aware, sequence-cluster-level train/val/test splits (Stage 06).
#
# Produces a cluster map TSV (cluster_rep<TAB>member) consumable by
# 06_assemble_dataset.py --cluster-map. Members are the FASTA header ids
# (GENOME~PROTID), which match the dataset's tagged_id.
#
# Usage (SLURM wrapper 07_cluster_secreted.sbatch calls this):
#   07_cluster_secreted.sh <in.faa> <out_prefix> <tmpdir> <min_seq_id> <cov> <threads>
set -euo pipefail

IN_FAA="${1:?input FASTA}"
OUT_PREFIX="${2:?output prefix}"
TMPDIR_MM="${3:?tmp dir}"
MIN_SEQ_ID="${4:-0.5}"
COV="${5:-0.8}"
THREADS="${6:-48}"
MMSEQS="${MMSEQS_BIN:-/shared/software/bin/mmseqs}"

mkdir -p "$(dirname "$OUT_PREFIX")" "$TMPDIR_MM"

# easy-cluster: cascaded clustering. --min-seq-id sequence-identity threshold,
# -c coverage fraction, --cov-mode 0 = bidirectional (both seqs covered) so a
# short fragment is not absorbed into a long protein's cluster.
"$MMSEQS" easy-cluster \
    "$IN_FAA" "$OUT_PREFIX" "$TMPDIR_MM" \
    --min-seq-id "$MIN_SEQ_ID" \
    -c "$COV" --cov-mode 0 \
    --threads "$THREADS" \
    --split-memory-limit 0

# easy-cluster writes:
#   ${OUT_PREFIX}_cluster.tsv   (cluster_rep<TAB>member)  <- the map we want
#   ${OUT_PREFIX}_rep_seq.fasta (representative sequences)
#   ${OUT_PREFIX}_all_seqs.fasta
N_MEMBERS=$(wc -l < "${OUT_PREFIX}_cluster.tsv")
N_CLUSTERS=$(cut -f1 "${OUT_PREFIX}_cluster.tsv" | sort -u | wc -l)
echo "[07] members=${N_MEMBERS} clusters=${N_CLUSTERS}"
echo "[07] cluster map: ${OUT_PREFIX}_cluster.tsv"
