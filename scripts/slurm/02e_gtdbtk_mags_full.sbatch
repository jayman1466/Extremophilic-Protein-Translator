#!/bin/bash
#SBATCH --job-name=eptrans_gtdbtk_full
#SBATCH --partition=memory
#SBATCH --cpus-per-task=64
#SBATCH --mem=600G
#SBATCH --time=48:00:00
#SBATCH --output=/groups/cress/projects/jaymin/eptrans_scratch/logs/gtdbtk/gtdbtk_full_%j.out
#SBATCH --error=/groups/cress/projects/jaymin/eptrans_scratch/logs/gtdbtk/gtdbtk_full_%j.out
#
# GTDB-Tk taxonomy for all 4,084 deep-sea MAGs.
#
# WHY THIS IS ONE JOB AND NOT AN ARRAY
# ------------------------------------
# Measured from the 50-MAG timing run (job 1162998, 02:13:09, exit 0):
#
#   stage                             wall     n   s/genome
#   ANI pre-screen                    2.0 min  50      2.4
#   Prodigal + marker identify        0.7 min  21      2.0
#   bacterial align + mask            2.5 min  19      7.9
#   pplacer archaeal ref tree         7.9 min   2    237.5
#   pplacer bacterial BACKBONE        5.0 min  19     15.7
#   pplacer class-level tree 8       15.5 min   6    154.8
#   pplacer class-level tree 2       28.5 min   5    341.8
#   pplacer class-level tree 7       17.1 min   3    342.7
#   pplacer class-level tree 3       23.8 min   3    475.7
#   pplacer class-level tree 1       25.2 min   2    755.0
#
# The class-level placements are 83% of total wall time, and their cost is
# essentially INDEPENDENT of how many genomes are placed:
#   correlation(n_genomes, seconds) = -0.27  over the five trees
#   per-tree cost 15.5-28.5 min, mean 22.0, while n ranged 2-6
# pplacer's expense is loading and traversing a huge reference tree, not placing
# a query on it. The r232 split package holds 8 class-level trees plus 1
# backbone (5.3 GB), so the FIXED ceiling is roughly
#   8 x 22 min + backbone 5 min + archaeal 8 min = ~3.2 h
# regardless of whether we submit 50 MAGs or 4,084.
#
# A chunked array would therefore pay that ~3.2 h in EVERY task: nine chunks of
# 500 would burn ~29 h of redundant tree loading, and each task needs its own
# 133 GiB allocation. One job pays it once. This is the opposite of the sizing
# logic used for GenomeSPOT in 02c, where per-MAG Prodigal cost dominated and
# chunking was the right call -- the shape of the cost curve, not habit, picks
# the architecture.
#
# MEMORY. Peak measured on the completed timing run 1162998 (02:13:09,
# CLASSIFY_RC 0), via two accounting mechanisms that agree to 1.2%:
#
#   139,125,764 kB = 132.68 GiB   getrusage, GNU time
#       $ grep 'Maximum resident set size' \
#           /groups/cress/projects/jaymin/eptrans_scratch/gtdbtk_timing/logs/classify.time
#   140,725,016 kB = 134.21 GiB   SLURM cgroup polling
#       $ sacct -j 1162998 --format=JobID,State,Elapsed,MaxRSS   # .batch row
#
# Both re-derivable from the paths above rather than from any transcript. They
# are not fully independent -- same run, same process tree -- but they are
# different measurement mechanisms: GNU time reports the peak of the largest
# single process it wrapped, SLURM polls the whole batch step's cgroup. Hence the
# small excess in the sacct figure. Wall clock corroborates too: time -v 2:10:53
# inside sacct's 02:13:09 step.
#
# 600 G requested = 4.5x headroom over the getrusage figure.
#
# Note on the partition choice: `standard` nodes (257,821 MB = 251.8 GiB) would
# TECHNICALLY fit 132.7 GiB, but at only ~1.9x headroom on a shared node, and
# 4,084 genomes may place into class-level trees that the 50-MAG sample never
# touched (it reached 5 of the 8). `memory` (677,146 MB = 661.3 GiB) is the
# defensible choice, not a necessary one -- stated that way because an earlier
# version of this comment claimed standard "cannot host this", which the numbers
# do not support.
#
# Do NOT size from the FAILED run 1162978: its peak was 48,440 kB (0.046 GiB),
# 2,872x smaller, because it aborted in 1.5 s on an unrecognized flag before
# pplacer ever loaded a tree. That figure is a measurement of nothing.
#
# WALLTIME: 48 h requested against a fixed component of ~3.2 h. The per-genome
# marginal cost could NOT be estimated from the timing run -- with n = 2..6 per
# tree the correlation was negative, so the per-genome term is buried in
# between-tree variance. 48 h is deliberately generous rather than fitted, and
# all partitions here have no time limit so there is nothing to trade against.
#
# RESUMABILITY: GTDB-Tk has no --resume. The workflow is therefore run as its
# three constituent steps (identify -> align -> classify) with a completion
# marker after each, so a requeue after a crash or walltime kill skips finished
# steps instead of re-doing ~3 h of tree work. classify_wf is exactly these
# three in sequence.
#
# EXPECTED OUTPUT QUALITY, from the timing run's 50 MAGs:
#   family named  49/50 (98%)   <- selection.py matches on family
#   genus named   41/50 (82%)   <- and outgroups on genus
#   species named 29/50 (58%)
#   47 distinct families among 50 MAGs
# So these MAGs are usable for phylogeny-controlled pairing, which is the whole
# reason taxonomy is on the critical path.

set -uo pipefail        # NOT -e: one unclassifiable genome must not void the run

GTDBTK=/shared/software/bin/gtdbtk
export GTDBTK_DATA_PATH=/shared/db/gtdbtk/r232
MAG_DIR=/groups/cress/projects/jaymin/IS1111/deepsea_extract/HQ_MAGs_drep
WORK=/groups/cress/projects/jaymin/eptrans_scratch/gtdbtk_full
CPUS="${SLURM_CPUS_PER_TASK:-64}"

mkdir -p "$WORK"/{genomes,out,logs,markers}
cd "$WORK" || exit 1

echo "HOST $(hostname)"
echo "START $(date)"
"$GTDBTK" --version 2>&1 | head -1
echo "VERSION_DATA $(grep -h VERSION_DATA "$GTDBTK_DATA_PATH"/metadata/* 2>/dev/null | head -1)"
echo "CPUS $CPUS"
echo "MEM_REQUESTED ${SLURM_MEM_PER_NODE:-unknown} MB"

# ---- stage genomes with sanitized ids -------------------------------------
# GTDB-Tk validates the genome id it derives from the filename and rejects
# [ )=;(, ] outright (job 1162992 died on "Invalid genome ID:
# 10A(CNS0876618)_bin.16"). Shell quoting does not help: the file is already in
# place when the validator runs. 1,445 of 4,084 names (35.4%) embed an accession
# in parentheses -- OES 1,264, SAMN 95, CNS 86 -- and parentheses are the only
# offending characters anywhere in the corpus.
#
# Verified injective over all 4,084 real names: 4,084 distinct originals ->
# 4,084 distinct sanitized, zero collisions. Re-checked at runtime below.
if [[ ! -f markers/staged.done ]]; then
  rm -f genomes/*.fa 2>/dev/null
  : > mag_id_map.tsv
  # NUL-delimited: parenthesised filenames break naive word-splitting.
  while IFS= read -r -d '' F; do
    ORIG=$(basename "$F" .fa)
    SAFE=$(printf '%s' "$ORIG" | tr ' )=;(,' '______')
    cp "$F" "genomes/${SAFE}.fa"
    printf '%s\t%s\n' "$SAFE" "$ORIG" >> mag_id_map.tsv
  done < <(find "$MAG_DIR" -maxdepth 1 -name '*.fa' -print0 | sort -z)

  N_STAGED=$(find genomes -name '*.fa' | wc -l)
  N_MAP=$(wc -l < mag_id_map.tsv)
  N_UNIQ=$(cut -f1 mag_id_map.tsv | sort -u | wc -l)
  echo "MAGS_STAGED $N_STAGED"
  echo "ID_MAP_ROWS $N_MAP"
  echo "ID_MAP_DISTINCT_SAFE $N_UNIQ"
  echo "ID_MAP_RENAMED $(awk -F'\t' '$1!=$2' mag_id_map.tsv | wc -l)"
  if [[ "$N_UNIQ" != "$N_MAP" ]]; then
    echo "FATAL sanitized genome ids collided -- aborting"; exit 1
  fi
  if [[ "$N_STAGED" != "4084" ]]; then
    echo "WARNING staged $N_STAGED, expected 4084 -- continuing but check the source dir"
  fi
  touch markers/staged.done
else
  echo "MAGS_STAGED skipped (marker present, $(find genomes -name '*.fa' | wc -l) files)"
fi

# ---- step 1/3: identify ---------------------------------------------------
# Per-genome cost (~2 s/genome measured), so this is the part that actually
# scales with 4,084. Its output is reused by align, hence the marker.
if [[ ! -f markers/identify.done ]]; then
  echo "IDENTIFY_START $(date +%s)"
  /usr/bin/time -v "$GTDBTK" identify \
      --genome_dir genomes \
      --extension fa \
      --out_dir identify \
      --cpus "$CPUS" \
      > logs/identify.stdout 2> logs/identify.time
  RC=$?
  echo "IDENTIFY_RC $RC"
  grep -E 'Maximum resident set size|Elapsed \(wall clock\)' logs/identify.time || true
  [[ $RC -eq 0 ]] && touch markers/identify.done
else
  echo "IDENTIFY skipped (marker present)"
fi

# ---- step 2/3: align ------------------------------------------------------
if [[ -f markers/identify.done && ! -f markers/align.done ]]; then
  echo "ALIGN_START $(date +%s)"
  /usr/bin/time -v "$GTDBTK" align \
      --identify_dir identify \
      --out_dir align \
      --cpus "$CPUS" \
      > logs/align.stdout 2> logs/align.time
  RC=$?
  echo "ALIGN_RC $RC"
  grep -E 'Maximum resident set size|Elapsed \(wall clock\)' logs/align.time || true
  [[ $RC -eq 0 ]] && touch markers/align.done
else
  [[ -f markers/align.done ]] && echo "ALIGN skipped (marker present)"
fi

# ---- step 3/3: classify (the ~3.2 h fixed-cost step) ----------------------
if [[ -f markers/align.done && ! -f markers/classify.done ]]; then
  echo "CLASSIFY_START $(date +%s)"
  /usr/bin/time -v "$GTDBTK" classify \
      --genome_dir genomes \
      --extension fa \
      --align_dir align \
      --out_dir out \
      --cpus "$CPUS" \
      --min_af 0.5 \
      > logs/classify.stdout 2> logs/classify.time
  RC=$?
  echo "CLASSIFY_END $(date +%s)"
  echo "CLASSIFY_RC $RC"
  grep -E 'Maximum resident set size|Elapsed \(wall clock\)|User time' logs/classify.time || true
  [[ $RC -eq 0 ]] && touch markers/classify.done
else
  [[ -f markers/classify.done ]] && echo "CLASSIFY skipped (marker present)"
fi

# ---- restore original MAG ids --------------------------------------------
# Every downstream table (GenomeSPOT predictions, deep-sea labels) is keyed by
# the ORIGINAL name, so emit a joined copy rather than leaving a silent join
# failure for the next consumer.
echo "=== summaries ==="
for S in out/gtdbtk.bac120.summary.tsv out/gtdbtk.ar53.summary.tsv; do
  [[ -f "$S" ]] || { echo "MISSING $S"; continue; }
  echo "$S rows=$(($(wc -l < "$S")-1))"
  awk -F'\t' -v OFS='\t' '
    NR==FNR { orig[$1]=$2; next }
    FNR==1  { print "mag_id", $0; next }
    { print (($1 in orig) ? orig[$1] : $1), $0 }
  ' mag_id_map.tsv "$S" > "${S%.tsv}.orig_ids.tsv"
  echo "RESTORED ${S%.tsv}.orig_ids.tsv rows=$(($(wc -l < "${S%.tsv}.orig_ids.tsv")-1))"
  # Compare against the map -- an id-shape regex matches every row and would
  # read as "all failed" (the bug in 02d's first version).
  echo "UNMAPPED_IDS $(awk -F'\t' 'NR==FNR{m[$2];next} FNR>1 && !($1 in m){c++} END{print c+0}' \
      mag_id_map.tsv "${S%.tsv}.orig_ids.tsv")"
  echo "RESTORED_WITH_PARENS $(awk -F'\t' 'FNR>1{print $1}' "${S%.tsv}.orig_ids.tsv" | grep -c '(' || true)"
done

# ---- rank completeness: does selection.py have what it needs? ------------
echo "=== rank completeness (selection.py needs family and genus) ==="
cat out/gtdbtk.*.summary.orig_ids.tsv 2>/dev/null \
  | awk -F'\t' 'FNR>1{print $3}' \
  | awk -F';' '{
      f=""; g=""; s="";
      for(i=1;i<=NF;i++){ if($i ~ /^f__/) f=substr($i,4);
                          if($i ~ /^g__/) g=substr($i,4);
                          if($i ~ /^s__/) s=substr($i,4) }
      tf += (f!=""); tg += (g!=""); ts += (s!=""); n++ }
    END{ printf "rows=%d family_named=%d genus_named=%d species_named=%d\n", n, tf, tg, ts }'
echo -n "distinct families: "
cat out/gtdbtk.*.summary.orig_ids.tsv 2>/dev/null \
  | awk -F'\t' 'FNR>1{print $3}' | tr ';' '\n' | grep '^f__' | grep -v 'f__$' | sort -u | wc -l

echo "END $(date)"
echo "FULL_RUN_DONE"
