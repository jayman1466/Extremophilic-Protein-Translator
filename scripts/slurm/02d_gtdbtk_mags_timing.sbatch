#!/usr/bin/env bash
#SBATCH --job-name=eptrans_gtdbtk_t50
#SBATCH --partition=memory
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=600G
#SBATCH --time=12:00:00
#SBATCH --output=/groups/cress/projects/jaymin/eptrans_scratch/logs/gtdbtk/gtdbtk_t50_%j.out
#SBATCH --error=/groups/cress/projects/jaymin/eptrans_scratch/logs/gtdbtk/gtdbtk_t50_%j.out

# Timing/sizing run: GTDB-Tk classify_wf on a 50-MAG sample of the deep-sea set.
#
# PURPOSE. This is NOT the production taxonomy run. It exists to measure three
# numbers that decide how the full 4,084-MAG run is shaped, none of which can be
# guessed from the GenomeSPOT run (that was embarrassingly parallel and
# CPU-bound; classify_wf is neither):
#   1. peak RSS of the pplacer step  -> the memory request, and whether the
#      bacterial tree fits on `memory` (677 G) or needs `high-memory` (3 T)
#   2. wall time per MAG, split by stage (identify / align / classify)
#   3. whether --mash_db must be built (no sketch db was found under
#      /shared/db/gtdbtk/latest/mash), or whether --skip_ani_screen is needed
#
# WHY NOT `standard`. pplacer holds the reference tree plus per-genome
# placements in memory and is the documented memory ceiling of the whole
# workflow. `standard` nodes top out at ~257 G. Requesting 600 G on `memory`
# buys headroom for the measurement; the production run gets sized from what
# this reports, not from this request.
#
# WHY 50 MAGs. classify_wf batches: align and pplacer cost is dominated by the
# reference data, not the query count, so a 50-genome run measures the fixed
# cost and gives a per-genome marginal. Extrapolating from 1 genome would
# attribute all the fixed cost to that genome and overestimate the full run by
# more than an order of magnitude.
#
# SAMPLING. Deterministic (sorted list, every Nth) rather than random, so a
# rerun measures the same MAGs; and spread across the whole list so the sample
# spans the assembly-quality range rather than one sample's bins.

set -uo pipefail   # NOT -e: a single MAG failure must not abort the measurement

MAG_DIR=/groups/cress/projects/jaymin/IS1111/deepsea_extract/HQ_MAGs_drep
WORK=/groups/cress/projects/jaymin/eptrans_scratch/gtdbtk_timing
GTDBTK=/shared/software/bin/gtdbtk
export GTDBTK_DATA_PATH=/shared/db/gtdbtk/r232
N_SAMPLE=50
CPUS=${SLURM_CPUS_PER_TASK:-64}

mkdir -p "$WORK"/{genomes,out,logs}
cd "$WORK"

echo "HOST $(hostname)"
echo "START $(date)"
echo "GTDBTK_DATA_PATH $GTDBTK_DATA_PATH"
"$GTDBTK" --version 2>&1 | head -1
echo "VERSION_DATA $(grep -h VERSION_DATA "$GTDBTK_DATA_PATH"/metadata/* 2>/dev/null | head -1)"
echo "CPUS $CPUS"
echo "MEM_REQUESTED ${SLURM_MEM_PER_NODE:-unknown} MB"

# ---- tarball inventory (folded in: `tar -tzf` on the 3.8 GB archive exceeds
# ---- the 60 s interactive SSH ceiling, so it runs here where there is time.
# ---- Settles whether the archive shipped any non-FASTA payload -- a taxonomy
# ---- or checkm table would make this whole run unnecessary.
TARBALL=/groups/cress/projects/jaymin/IS1111/Deepsea_HQ_MAGs_drep.tar.gz
if [[ -f "$TARBALL" ]]; then
  tar -tzf "$TARBALL" > "$WORK/tar_entries.txt" 2>"$WORK/tar_entries.err"
  echo "TAR_TOTAL_ENTRIES $(wc -l < "$WORK/tar_entries.txt")"
  echo "TAR_FA_ENTRIES $(grep -c '\.fa$' "$WORK/tar_entries.txt")"
  echo "TAR_NON_FA_ENTRIES $(grep -v '\.fa$' "$WORK/tar_entries.txt" | grep -c . || true)"
  echo "TAR_NON_FA_SAMPLE_BEGIN"
  grep -v '\.fa$' "$WORK/tar_entries.txt" | grep . | head -20 || true
  echo "TAR_NON_FA_SAMPLE_END"
else
  echo "TAR_MISSING $TARBALL"
fi

# ---- ANI screen: nothing to configure on 2.7.2 ----
# MEASURED, not assumed: GTDB-Tk 2.7.2 replaced the mash-based ANI screen with
# skani and REMOVED --skip_ani_screen and --mash_db altogether. `classify_wf
# --help` on this install lists an "optional skani arguments" group whose only
# member is --place_species; the screen runs unconditionally and needs no
# user-supplied sketch database (the reference skani db ships in the r232 data
# dir, which is why /shared/db/gtdbtk/r232/skani exists and .../mash does not).
#
# A first version of this script passed --skip_ani_screen (correct for 2.2-2.3,
# which is the version the source paper used) and classify_wf aborted in 1.5 s
# with "unrecognized arguments". Kept as a note because the flag appears in most
# online GTDB-Tk recipes and in the paper's own methods.
echo "SKANI_REF_DIR $(ls -d "$GTDBTK_DATA_PATH"/skani 2>/dev/null || echo absent)"
# NOTE: "${ANI_ARGS[@]}" on an EMPTY array is an unbound-variable error under
# `set -u` on bash < 4.4 (verified: `set -u; A=(); f "${A[@]}"` -> "A[@]:
# unbound variable"). Seed with one harmless real flag so the array is never
# empty; --min_af 0.5 is classify_wf's own default, so this changes nothing.
ANI_ARGS=(--min_af 0.5)

# ---- deterministic spread sample ----
# Quoting is load-bearing: MAG filenames contain parentheses, e.g.
# FDZ071-WW16-18(OES00301993)_bin.24.fa. NUL-delimited throughout.
find "$MAG_DIR" -maxdepth 1 -name '*.fa' -print0 \
  | sort -z \
  | tr '\0' '\n' > all_mags.txt
TOTAL=$(wc -l < all_mags.txt)
echo "MAGS_AVAILABLE $TOTAL"
awk -v n="$N_SAMPLE" -v t="$TOTAL" 'NR % int(t/n ? t/n : 1) == 1' all_mags.txt \
  | head -"$N_SAMPLE" > sample_mags.txt
echo "MAGS_SAMPLED $(wc -l < sample_mags.txt)"

# stage copies so the reference dir and the source tree stay untouched.
#
# GENOME IDs MUST BE SANITIZED. Shell quoting is not sufficient here: GTDB-Tk
# derives a genome ID from the filename and runs its OWN validator on it, which
# rejects the characters  [ ) = ; ( , ]  and aborts the whole run --
#   ERROR: Invalid genome ID: 10A(CNS0876618)_bin.16
# 1,445 of the 4,084 deep-sea MAG names (35.4%) embed an accession in
# parentheses -- OES 1,264, SAMN 95, CNS 86 -- so this is not an edge case.
# Counted over the deep-sea labels TSV, and 1,445 is also the count of names
# holding ANY of GTDB-Tk's invalid characters, i.e. parentheses are the only
# offending characters present anywhere in the corpus.
#
# Staged filenames therefore replace each invalid character with '_', and
# mag_id_map.tsv records sanitized -> original so downstream joins can recover
# the true MAG id (which is the key used by the GenomeSPOT predictions table and
# the deep-sea labels TSV). Verified injective over all 4,084 real MAG names:
# 4,084 distinct originals -> 4,084 distinct sanitized, zero collisions.
rm -f genomes/*.fa 2>/dev/null
: > mag_id_map.tsv
while IFS= read -r F; do
  [[ -z "$F" ]] && continue
  ORIG=$(basename "$F" .fa)
  SAFE=$(printf '%s' "$ORIG" | tr ' )=;(,' '______')
  cp "$F" "genomes/${SAFE}.fa"
  printf '%s\t%s\n' "$SAFE" "$ORIG" >> mag_id_map.tsv
done < sample_mags.txt
echo "MAGS_STAGED $(find genomes -name '*.fa' | wc -l)"
echo "ID_MAP_ROWS $(wc -l < mag_id_map.tsv)"
echo "ID_MAP_DISTINCT_SAFE $(cut -f1 mag_id_map.tsv | sort -u | wc -l)"
echo "ID_MAP_RENAMED $(awk -F'\t' '$1!=$2' mag_id_map.tsv | wc -l)"
# fail loudly rather than silently classifying fewer genomes than intended
if [[ "$(cut -f1 mag_id_map.tsv | sort -u | wc -l)" != "$(wc -l < mag_id_map.tsv)" ]]; then
  echo "FATAL sanitized genome IDs collided -- aborting"; exit 1
fi
echo "STAGED_CONTIGS_TOTAL $(grep -ch '^>' genomes/*.fa | paste -sd+ | bc 2>/dev/null || echo NA)"

# ---- classify_wf, timed ----
# --extension fa      : MAGs are .fa, not the .fna default
# --skip_ani_screen /
#   --mash_db         : resolved above
# /usr/bin/time -v    : gives Maximum resident set size, the number that
#                       actually sizes the production run
echo "CLASSIFY_START $(date +%s)"
/usr/bin/time -v "$GTDBTK" classify_wf \
    --genome_dir genomes \
    --extension fa \
    --out_dir out \
    --cpus "$CPUS" \
    "${ANI_ARGS[@]}" \
    > logs/classify.stdout 2> logs/classify.time
RC=$?
echo "CLASSIFY_END $(date +%s)"
echo "CLASSIFY_RC $RC"

echo "=== peak memory and wall (from /usr/bin/time -v) ==="
grep -E 'Maximum resident set size|Elapsed \(wall clock\)|User time|System time' logs/classify.time || true

echo "=== per-stage timing from gtdbtk log ==="
grep -E '^\[|done in|Finished' out/gtdbtk.log 2>/dev/null | tail -25 || true

echo "=== summary files produced ==="
find out -maxdepth 1 -name '*summary.tsv' -exec sh -c 'echo "$1 rows=$(($(wc -l < "$1")-1))"' _ {} \;

echo "=== classification sample (genome IDs shown as staged, i.e. sanitized) ==="
for S in out/gtdbtk.bac120.summary.tsv out/gtdbtk.ar53.summary.tsv; do
  [[ -f "$S" ]] && { echo "--- $S"; cut -f1,2 "$S" | head -6; }
done

# ---- restore original MAG ids in the summaries ----
# The summary's first column is the sanitized id; every downstream table (the
# GenomeSPOT predictions TSV, the deep-sea labels TSV) is keyed by the ORIGINAL
# MAG name, so emit a joined copy rather than leaving the reconciliation to
# whoever reads this next.
for S in out/gtdbtk.bac120.summary.tsv out/gtdbtk.ar53.summary.tsv; do
  [[ -f "$S" ]] || continue
  awk -F'\t' -v OFS='\t' '
    NR==FNR { orig[$1]=$2; next }
    FNR==1  { print "mag_id", $0; next }
    { print (($1 in orig) ? orig[$1] : $1), $0 }
  ' mag_id_map.tsv "$S" > "${S%.tsv}.orig_ids.tsv"
  echo "RESTORED ${S%.tsv}.orig_ids.tsv rows=$(($(wc -l < "${S%.tsv}.orig_ids.tsv")-1))"
  # Count ids that are NOT in the map, i.e. genuinely unmapped. The earlier
  # pattern matched '_bin.<digits>$', which every id ends with, so it reported
  # the total row count and would have read as "everything failed" when in fact
  # the restoration succeeded. Compare against the map instead of guessing from
  # the id shape.
  echo "UNMAPPED_IDS $(awk -F'\t' 'NR==FNR{m[$2];next} FNR>1 && !($1 in m){c++} END{print c+0}' \
      mag_id_map.tsv "${S%.tsv}.orig_ids.tsv")"
  echo "RESTORED_WITH_PARENS $(awk -F'\t' 'FNR>1{print $1}' "${S%.tsv}.orig_ids.tsv" | grep -c '(' || true)"
done

echo "=== unclassified / failed ==="
grep -c 'Unclassified' out/*summary.tsv 2>/dev/null || echo "none"

echo "END $(date)"
echo "TIMING_DONE"
