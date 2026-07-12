#!/bin/bash
#SBATCH --job-name=eptrans_signalp
#SBATCH --partition=standard
#SBATCH --cpus-per-task=48
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --output=/groups/cress/projects/jaymin/eptrans_scratch/logs/signalp/signalp_%A_%a.out
#SBATCH --array=0-189%5

set -euo pipefail

# SignalP 6.0 is installed as a pyenv shim on biotite - activate pyenv.
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"

ACC_LIST="/groups/cress/projects/jaymin/eptrans_scratch/signalp_gtdb_fill_accessions.txt"
PROT_DIR="/groups/cress/projects/jaymin/IS1111/gtdb/protein_faa_reps"      # per-genome proteomes: <domain>/<GENOME>_protein.faa.gz
OUT_ROOT="/groups/cress/projects/jaymin/eptrans_scratch/signalp_gtdb_fill"
ORGANISM="other"
MODE="fast"
CHUNK_SIZE=1014
TORCH_THREADS=48
WRITE_PROCS=48
BSIZE=32

# This task processes accessions [START, END) of the list.
START=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE + 1))
END=$((START + CHUNK_SIZE - 1))
mkdir -p "$OUT_ROOT"

# Resume guard: skip a chunk whose SignalP output already looks complete. Lets a
# long background fill be killed/resubmitted freely (idempotent per chunk). A
# chunk is "done" when its prediction_results.txt exists AND ends with a data
# line for the last expected genome (SignalP writes it last); we approximate
# completeness by a non-empty prediction_results.txt plus a DONE sentinel.
OUT="$OUT_ROOT/chunk_${SLURM_ARRAY_TASK_ID}"
if [ -f "$OUT/.done" ]; then
  echo "chunk $SLURM_ARRAY_TASK_ID already complete (.done present) - skipping"
  exit 0
fi

# SignalP batches internally: concatenate this chunk's proteomes into ONE FASTA
# (protein ids namespaced by genome as {GENOME}~{PROTID}) and run once.
# Write the concatenated chunk FASTA to the big VAST scratch, NOT node-local /tmp:
# each chunk is ~2.5M proteins (~1-1.5 GB) and several concurrent tasks on one
# node exhausted /tmp -> awk died with ENOSPC and the task failed in ~1s (exit 2),
# so the %5 throttle burned through array indices without doing work. /groups has
# ~247 TB free. trap-clean so temp files never accumulate.
mkdir -p "$OUT_ROOT/tmp"
CHUNK_FAA=$(mktemp -p "$OUT_ROOT/tmp" --suffix=.faa)
trap 'rm -f "$CHUNK_FAA"' EXIT
N=0
for i in $(seq "$START" "$END"); do
  ACC=$(sed -n "${i}p" "$ACC_LIST")
  [ -z "$ACC" ] && continue
  FAA=""
  for DOM in bacteria archaea; do
    CAND="$PROT_DIR/$DOM/${ACC}_protein.faa.gz"
    if [ -f "$CAND" ]; then FAA="$CAND"; break; fi
  done
  if [ -z "$FAA" ]; then echo "proteome not found for $ACC"; continue; fi
  # prefix each header with the genome id so proteins stay attributable
  gunzip -c "$FAA" | awk -v g="$ACC" '/^>/{sub(/^>/,">"g"~")}1' >> "$CHUNK_FAA"
  N=$((N+1))
done
echo "chunk $SLURM_ARRAY_TASK_ID: $N genomes -> $(grep -c "^>" "$CHUNK_FAA") proteins"

mkdir -p "$OUT"
signalp6 --fastafile "$CHUNK_FAA" --output_dir "$OUT" \
         --format none --organism "$ORGANISM" --mode "$MODE" \
         --torch_num_threads "$TORCH_THREADS" --write_procs "$WRITE_PROCS" --bsize "$BSIZE"

rm -f "$CHUNK_FAA"
# sentinel marks the chunk complete for the resume guard above
touch "$OUT/.done"
echo "done chunk $SLURM_ARRAY_TASK_ID -> $OUT"
