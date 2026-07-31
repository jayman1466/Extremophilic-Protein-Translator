#!/bin/bash
# GenomeSPOT on the deep-sea MAG set (Guo et al. 2026, DSGC).
#
# Differs from 02_genomespot_full.sbatch in one important way: GTDB representatives
# ship precomputed proteomes (protein_faa_reps/), but these MAGs are NUCLEOTIDE-ONLY
# (<SAMPLE_ID>_bin.<N>.fa). GenomeSPOT needs both contigs (-c) and proteins (-p),
# so Prodigal must run first, in metagenome mode (-p meta) because a MAG is a
# population bin rather than a single clonal genome and may carry mixed GC/coding
# statistics that the single-genome trainer handles poorly.
#
# Filenames contain parentheses -- e.g. FDZ071-WW16-18(OES00301993)_bin.24.fa --
# so every path expansion here is quoted, and the file list is passed via a
# NUL-delimited stream rather than word-splitting.
#
#SBATCH --job-name=eptrans_gs_mags
#SBATCH --partition=standard
#SBATCH --cpus-per-task=48
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/groups/cress/projects/jaymin/eptrans_scratch/logs/genomespot/gs_mags_%A_%a.out
#SBATCH --array=0-8%5

set -uo pipefail   # NOTE: no -e; a single bad MAG must not kill a whole chunk

MAG_DIR="/groups/cress/projects/jaymin/IS1111/deepsea_extract/HQ_MAGs_drep"
MAG_LIST="/groups/cress/projects/jaymin/eptrans_scratch/mag_list.txt"
MODELS="/groups/cress/projects/jaymin/eptrans_scratch/GenomeSPOT/models"
OUT_ROOT="/groups/cress/projects/jaymin/eptrans_scratch/genomespot_mags"
GS_PY="/groups/cress/projects/jaymin/eptrans_scratch/gsenv/bin/python"
PRODIGAL="/shared/software/bin/prodigal"
CHUNK_SIZE=500
CPUS=48

mkdir -p "$OUT_ROOT" "$(dirname "$MAG_LIST")" \
         /groups/cress/projects/jaymin/eptrans_scratch/logs/genomespot

# Build the list once (task 0 only) so all tasks index the same ordering.
if [ "${SLURM_ARRAY_TASK_ID}" = "0" ] && [ ! -s "$MAG_LIST" ]; then
  find "$MAG_DIR" -maxdepth 1 -name '*.fa' | sort > "$MAG_LIST"
fi
# Other tasks wait for it rather than racing to build their own.
for _ in $(seq 1 60); do [ -s "$MAG_LIST" ] && break; sleep 5; done
if [ ! -s "$MAG_LIST" ]; then echo "FATAL: $MAG_LIST missing/empty"; exit 1; fi

START=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE + 1))
END=$((START + CHUNK_SIZE - 1))
WORK="$OUT_ROOT/chunk_${SLURM_ARRAY_TASK_ID}"
SCRATCH="$WORK/faa"
mkdir -p "$WORK" "$SCRATCH"

run_one() {
  local F="$1"
  local BASE
  BASE=$(basename "$F" .fa)
  local OUTPREFIX="$WORK/$BASE"
  # idempotent: skip MAGs already predicted, so a requeued task resumes
  if [ -s "${OUTPREFIX}.predictions.tsv" ]; then return 0; fi
  local FAA="$SCRATCH/$BASE.faa"
  if [ ! -s "$FAA" ]; then
    if ! "$PRODIGAL" -p meta -q -i "$F" -a "$FAA" -o /dev/null 2>/dev/null; then
      echo "prodigal failed: $BASE"; rm -f "$FAA"; return 0
    fi
  fi
  # GenomeSPOT needs >=1 protein; empty proteomes crash the featurizer
  if ! grep -q '^>' "$FAA" 2>/dev/null; then echo "no proteins: $BASE"; return 0; fi
  if ! "$GS_PY" -m genome_spot.genome_spot --models "$MODELS" \
        -c "$F" -p "$FAA" -o "$OUTPREFIX" >/dev/null 2>&1; then
    echo "genomespot failed: $BASE"; return 0
  fi
  # proteomes are large and re-derivable; drop once predictions exist
  rm -f "$FAA"
}
export -f run_one
export MODELS GS_PY PRODIGAL WORK SCRATCH

# NUL-delimited so parenthesised filenames survive
sed -n "${START},${END}p" "$MAG_LIST" | tr '\n' '\0' \
  | xargs -0 -P "$CPUS" -I{} bash -c 'run_one "$@"' _ {}

N_OK=$(find "$WORK" -maxdepth 1 -name '*.predictions.tsv' | wc -l)
echo "done chunk ${SLURM_ARRAY_TASK_ID}: ${N_OK} predictions in $WORK"
