#!/bin/bash
#SBATCH --job-name=eptrans_genomespot
#SBATCH --partition=standard
#SBATCH --cpus-per-task=48
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/groups/cress/projects/jaymin/eptrans_scratch/logs/genomespot/genomespot_%A_%a.out
#SBATCH --array=0-19%10

set -euo pipefail

ACC_LIST="/groups/cress/projects/jaymin/eptrans_scratch/all_accessions_biotite.txt"
GENOME_INDEX="/groups/cress/projects/jaymin/IS1111/work/genome_index.tsv"
PROT_DIR="/groups/cress/projects/jaymin/IS1111/gtdb/protein_faa_reps"
MODELS="/groups/cress/projects/jaymin/eptrans_scratch/GenomeSPOT/models"
OUT_ROOT="/groups/cress/projects/jaymin/eptrans_scratch/genomespot"
GS_PY="/groups/cress/projects/jaymin/eptrans_scratch/gsenv/bin/python"
CHUNK_SIZE=10000
CPUS=48

START=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE + 1))
END=$((START + CHUNK_SIZE - 1))
OUT="$OUT_ROOT/chunk_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$OUT"

run_one() {
  local ACC="$1"
  # bare accession for genome-index lookup (strip GB_/RS_)
  local BARE=$(echo "$ACC" | sed -E 's/^(GB_|RS_)//')
  local FNA=$(awk -F'\t' -v a="$BARE" '$1==a{print $2; exit}' "$GENOME_INDEX")
  if [ -z "$FNA" ]; then echo "no fna: $ACC"; return 0; fi
  local FAA=""
  for DOM in bacteria archaea; do
    local CAND="$PROT_DIR/$DOM/${ACC}_protein.faa.gz"
    if [ -f "$CAND" ]; then FAA="$CAND"; break; fi
  done
  if [ -z "$FAA" ]; then echo "no faa: $ACC"; return 0; fi
  "$GS_PY" -m genome_spot.genome_spot --models "$MODELS" \
      -c "$FNA" -p "$FAA" -o "$OUT/$BARE" >/dev/null 2>&1 || echo "gs failed: $ACC"
}
export -f run_one
export GENOME_INDEX PROT_DIR MODELS OUT GS_PY

sed -n "${START},${END}p" "$ACC_LIST" | xargs -P "$CPUS" -I{} bash -c 'run_one "$@"' _ {}
echo "done chunk $SLURM_ARRAY_TASK_ID ($(ls "$OUT" | wc -l) outputs)"
