#!/bin/bash
#SBATCH --job-name=eptrans_sp_tgt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/groups/cress/projects/jaymin/eptrans_scratch/logs/signalp/sptgt_%A_%a.out
#SBATCH --array=0-3
#
# Stage 05c - TARGETED SignalP over just the genomes the current selection needs.
#
# Distinct from 05b (scripts/slurm/05b_signalp_gtdb_fill.sbatch, running as array
# 1152569), which walks all 192,652 r232 genomes in accession order and is months
# from finishing. 05c runs only the 2,582 genomes still missing for the LOCKED
# tier rules (thermophile high-only, all other classes high+medium), so the
# retrain is not gated on that fill. Both may run concurrently; 05c deliberately
# targets the gpu/gpu_h200 partitions so it does not compete with 05b for the
# `standard` slots under MaxJobsPerUser=10.
#
# No --gres is requested: SignalP 6.0 in `fast` mode was measured at 12.75 seq/s
# on an H200 vs 8.86 seq/s on 16 CPU cores, i.e. only ~1.4x, so a GPU
# reservation would idle an accelerator for a marginal gain. Throughput source:
# labnotebook.md, section "SignalP throughput benchmark (measured)" -- SignalP 6.0
# fast mode over 10,000 proteins, H200 12.75 seq/s at bsize 32, A5000/gpu 9.60 at
# bsize 64, CPU standard 16-thread 8.86 (figure: results/signalp_gpu_benchmark.png).
# Fast mode is CPU-decode-bound, not GPU-bound: the transformer sees only short
# N-terminal windows, so the GPU idles while CRF decoding sets the pace.
#
# These partitions were chosen for free CPU capacity, from a POINT-IN-TIME
# snapshot taken 2026-08-04 via
#   sinfo -p gpu,gpu_h200 -o '%P %D %C %m %G %t'   # %C = alloc/idle/other/total
# which returned  gpu 40/312/0/352 | gpu_h200 12/212/0/224 | memory 3008/0/0/3008
# i.e. 312 and 212 idle CPUs on the two gpu partitions and ZERO on memory, whose
# short pending queue (20 jobs vs standard's 261) reflects nothing finishing
# rather than room to run.
#
# CLUSTER LOAD IS VOLATILE -- these counts are certainly stale by the time you
# read this. Re-run the sinfo line above before resubmitting; if memory has freed
# up it is the better target (more cores per node, no GPU jobs to crowd), and if
# gpu_h200 is busy leave it clear, since the MLM training stage needs it.
#
# Submit BOTH halves (partition on the command line overrides the directive):
#   sbatch --partition=gpu      --array=0-3 scripts/slurm/05c_signalp_targeted.sbatch
#   sbatch --partition=gpu_h200 --array=4-7 scripts/slurm/05c_signalp_targeted.sbatch
#
# Proteomes are resolved by ONE code path for both GTDB and custom genomes.
# Stage 03d (03d_ingest_mags_custom_tree.sbatch) lands the selected MAGs in
# custom_genomes/ with the same layout as the GTDB mirror --
# protein_faa_reps/{archaea,bacteria}/<ACCESSION>_protein.faa.gz -- so the only
# difference here is which root directory to look under, selected by accession
# prefix. No Prodigal, no MAG special-casing: an earlier draft of this script
# re-ran Prodigal inline from the raw contigs, which would have produced correct
# predictions while leaving the MAGs permanently outside the standard layout.
#
# REQUIRES 03d to have completed. A CU_ accession with no proteome under
# $CUSTOM_DIR means 03d has not run (or failed for that genome); the loop counts
# it as missing rather than silently reconstructing it.
#
# `set -e` is deliberately OMITTED: a single unreadable proteome must not abort
# a 323-genome chunk.
set -uo pipefail

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"

ACC_LIST="/groups/cress/projects/jaymin/eptrans_scratch/sp_diff/new3.txt"
PROT_DIR="/groups/cress/projects/jaymin/IS1111/gtdb/protein_faa_reps"
CUSTOM_DIR="/groups/cress/projects/jaymin/IS1111/custom_genomes/protein_faa_reps"
OUT_ROOT="/groups/cress/projects/jaymin/eptrans_scratch/signalp_targeted"
ORGANISM="other"
MODE="fast"
CHUNK_SIZE=323
TORCH_THREADS=32
WRITE_PROCS=32
BSIZE=32

START=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE + 1))
END=$((START + CHUNK_SIZE - 1))
mkdir -p "$OUT_ROOT" "$OUT_ROOT/tmp"

OUT="$OUT_ROOT/chunk_${SLURM_ARRAY_TASK_ID}"
if [ -f "$OUT/.done" ]; then
  echo "chunk $SLURM_ARRAY_TASK_ID already complete (.done present) - skipping"
  exit 0
fi

# Chunk FASTA goes on the big VAST scratch, NOT node-local /tmp: concurrent tasks
# on one node exhausted /tmp in the 05b run and awk died with ENOSPC, which burned
# array indices without doing work.
CHUNK_FAA=$(mktemp -p "$OUT_ROOT/tmp" --suffix=.faa)
trap 'rm -f "$CHUNK_FAA"' EXIT

N_GTDB=0; N_CUSTOM=0; N_MISS=0
for i in $(seq "$START" "$END"); do
  ACC=$(sed -n "${i}p" "$ACC_LIST")
  [ -z "$ACC" ] && continue

  # Root depends only on the accession prefix; the filename convention is
  # identical in both trees. Custom accessions are CU_CUST_NNNNNNNNN.1 after the
  # 03d rename, so no path component needs defensive quoting -- but the quoting
  # is kept anyway, since $ACC comes from a file.
  case "$ACC" in
    CU_*) ROOT="$CUSTOM_DIR" ;;
    *)    ROOT="$PROT_DIR" ;;
  esac

  FAA=""
  for DOM in bacteria archaea; do
    CAND="$ROOT/$DOM/${ACC}_protein.faa.gz"
    if [ -f "$CAND" ]; then FAA="$CAND"; break; fi
  done
  if [ -z "$FAA" ]; then
    echo "proteome not found for $ACC (root=$ROOT)"; N_MISS=$((N_MISS+1)); continue
  fi

  gunzip -c "$FAA" | awk -v g="$ACC" '/^>/{sub(/^>/,">"g"~")}1' >> "$CHUNK_FAA"
  case "$ACC" in
    CU_*) N_CUSTOM=$((N_CUSTOM+1)) ;;
    *)    N_GTDB=$((N_GTDB+1)) ;;
  esac
done

NPROT=$(grep -c "^>" "$CHUNK_FAA" 2>/dev/null || echo 0)
echo "chunk $SLURM_ARRAY_TASK_ID: gtdb=$N_GTDB custom=$N_CUSTOM missing=$N_MISS proteins=$NPROT"
if [ "$NPROT" -eq 0 ]; then
  echo "no proteins collected for chunk $SLURM_ARRAY_TASK_ID - not invoking signalp6"
  exit 1
fi

mkdir -p "$OUT"
signalp6 --fastafile "$CHUNK_FAA" --output_dir "$OUT" \
         --format none --organism "$ORGANISM" --mode "$MODE" \
         --torch_num_threads "$TORCH_THREADS" --write_procs "$WRITE_PROCS" --bsize "$BSIZE"
SP_RC=$?
echo "SIGNALP_RC $SP_RC"

if [ "$SP_RC" -eq 0 ] && [ -s "$OUT/prediction_results.txt" ]; then
  touch "$OUT/.done"
  echo "done chunk $SLURM_ARRAY_TASK_ID -> $OUT"
else
  echo "chunk $SLURM_ARRAY_TASK_ID FAILED (rc=$SP_RC) - no .done sentinel written"
  exit "$SP_RC"
fi
