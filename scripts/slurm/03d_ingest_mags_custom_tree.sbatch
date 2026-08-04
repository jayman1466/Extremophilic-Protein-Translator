#!/bin/bash
#SBATCH --job-name=eptrans_mag_ingest
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/groups/cress/projects/jaymin/eptrans_scratch/logs/mag_ingest/ingest_%j.out
#
# Stage 03d - land the 330 selected deep-sea MAGs in the custom_genomes/ tree,
# in the SAME on-disk layout as the GTDB r232 mirror, so every downstream stage
# (SignalP, secretome collection, clustering, embedding) resolves them through
# the identical code path with no MAG special-casing.
#
# WHY THIS EXISTS: the MAGs were labelled and taxonomically classified in place
# from deepsea_extract/HQ_MAGs_drep/*.fa, and the GenomeSPOT run (02c) deleted
# each Prodigal proteome after predicting to bound scratch usage. So as of now
# there is no proteome on disk for any MAG, and custom_genomes/ was never
# created. Stage 05c cannot run until this does.
#
# SCOPE: 330 MAGs, not 4,084 -- the ones the locked selection actually uses
# (320 needed for SignalP under secreted-scope classes + 15 in whole-proteome
# classes, overlapping by 5). map.tsv is authoritative; it is generated locally
# with injectivity and collision checks and staged to $WORK.
#
# ACCESSION RENAME: MAG ids embed an accession in parentheses for 35.4% of the
# set (e.g. "10C(CNS0876620)_bin.42"), which GTDB-Tk's own genome-ID validator
# rejects and which requires defensive quoting in every shell path. Each is
# remapped to the skill's GTDB-shaped scheme CU_CUST_NNNNNNNNN.1. The old id is
# preserved as mag_id in the provenance table, which is the join key every
# upstream artifact (GenomeSPOT predictions, GTDB-Tk summaries) already uses --
# so those artifacts need no remapping.
#
# `set -e` is omitted deliberately: one unreadable contig file or one Prodigal
# failure must not abort the remaining genomes. Failures are counted and the job
# exits non-zero at the end if any occurred.
set -uo pipefail

BASE="/groups/cress/projects/jaymin/IS1111"
MAG_DIR="$BASE/deepsea_extract/HQ_MAGs_drep"
CUSTOM="$BASE/custom_genomes"
WORK="/groups/cress/projects/jaymin/eptrans_scratch/mag_ingest"
MAP="$WORK/map.tsv"                       # <mag_id>\t<CU_CUST_NNNNNNNNN.1>
INDEX="$BASE/work/genome_index.tsv"
PRODIGAL="/shared/software/bin/prodigal"
TAXON="$WORK/mag_domain.tsv"              # <mag_id>\t<archaea|bacteria>, staged with map

mkdir -p "$CUSTOM/database/CUST" "$CUSTOM/protein_faa_reps/bacteria" \
         "$CUSTOM/protein_faa_reps/archaea" "$WORK/logs"

echo "MAP_ROWS $(wc -l < "$MAP")"

N_OK=0; N_FAIL=0; N_SKIP=0
: > "$WORK/appended_index_rows.tsv"
: > "$WORK/ingest_report.tsv"
printf 'accession\tbare\tdomain\tn_proteins\tfna_path\tfaa_path\n' > "$WORK/ingest_report.tsv"

while IFS=$'\t' read -r MAG_ID ACC; do
  [ -z "${MAG_ID:-}" ] && continue
  SRC="$MAG_DIR/${MAG_ID}.fa"
  if [ ! -f "$SRC" ]; then
    echo "MISSING_SOURCE $MAG_ID"; N_FAIL=$((N_FAIL+1)); continue
  fi

  BARE="${ACC#CU_}"                        # CUST_000000001.1
  # Nested path mirrors GTDB: database/CUST/NNN/NNN/NNN/<bare>_genomic.fna.gz
  # built from the 9 digits of the numeric body.
  DIGITS="${BARE#CUST_}"; DIGITS="${DIGITS%.*}"
  D1="${DIGITS:0:3}"; D2="${DIGITS:3:3}"; D3="${DIGITS:6:3}"
  FNA_DIR="$CUSTOM/database/CUST/$D1/$D2/$D3"
  FNA="$FNA_DIR/${BARE}_genomic.fna.gz"

  # Domain comes from the GTDB-Tk summaries (bac120 vs ar53) via mag_domain.tsv.
  # Measured on this set: 205 bacteria / 125 archaea -- so a "default to
  # bacteria" fallback would misfile 38% of these genomes into the wrong
  # protein_faa_reps subdirectory, where the GTDB-shaped lookup would not find
  # them. Missing domain is therefore a hard failure, not a defaulted value.
  DOM=$(awk -F'\t' -v m="$MAG_ID" '$1==m{print $2; exit}' "$TAXON" 2>/dev/null)
  if [ -z "${DOM:-}" ]; then
    echo "NO_DOMAIN $MAG_ID"; N_FAIL=$((N_FAIL+1)); continue
  fi
  FAA="$CUSTOM/protein_faa_reps/$DOM/${ACC}_protein.faa.gz"

  # Idempotent: a completed genome has BOTH files non-empty.
  if [ -s "$FNA" ] && [ -s "$FAA" ]; then
    N_SKIP=$((N_SKIP+1)); continue
  fi

  mkdir -p "$FNA_DIR"
  # Contigs: gzip into the nested path. Source is plain .fa.
  if ! gzip -c "$SRC" > "$FNA".tmp 2>/dev/null; then
    echo "GZIP_FAILED $MAG_ID"; rm -f "$FNA".tmp; N_FAIL=$((N_FAIL+1)); continue
  fi
  mv "$FNA".tmp "$FNA"

  # Proteins: Prodigal -p meta (population bin, not a clonal genome -- same mode
  # as the 02c GenomeSPOT run, so protein calls are identical to what produced
  # the predictions these labels rest on).
  PTMP="$WORK/tmp_${ACC}.faa"
  if ! "$PRODIGAL" -i "$SRC" -a "$PTMP" -p meta -q -o /dev/null 2>>"$WORK/logs/prodigal.err"; then
    echo "PRODIGAL_FAILED $MAG_ID"; rm -f "$PTMP"; N_FAIL=$((N_FAIL+1)); continue
  fi
  if [ ! -s "$PTMP" ]; then
    echo "PRODIGAL_EMPTY $MAG_ID"; rm -f "$PTMP"; N_FAIL=$((N_FAIL+1)); continue
  fi

  # Header format must match the GTDB proteomes exactly: >{PROTID} with Prodigal's
  # " # start # end # strand # attrs" suffix stripped at the FIRST space. The
  # {GENOME}~{PROTID} form is applied later when proteomes are concatenated.
  awk '/^>/{sub(/^>/,""); split($0,p," "); print ">" p[1]; next} 1' "$PTMP" \
    | gzip -c > "$FAA".tmp
  mv "$FAA".tmp "$FAA"
  NPROT=$(zgrep -c "^>" "$FAA" 2>/dev/null || echo 0)
  rm -f "$PTMP"

  printf '%s\t%s\n' "$BARE" "$FNA" >> "$WORK/appended_index_rows.tsv"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$ACC" "$BARE" "$DOM" "$NPROT" "$FNA" "$FAA" \
    >> "$WORK/ingest_report.tsv"
  N_OK=$((N_OK+1))
done < "$MAP"

echo "INGESTED $N_OK"
echo "SKIPPED_ALREADY_PRESENT $N_SKIP"
echo "FAILED $N_FAIL"

# Append to the shared index LAST, once, and only for genomes that fully
# succeeded. Guard the trailing newline: without it the first new row fuses to
# the last existing one and silently corrupts that entry.
if [ -s "$WORK/appended_index_rows.tsv" ]; then
  if [ -f "$INDEX" ] && [ -n "$(tail -c 1 "$INDEX")" ]; then
    echo "" >> "$INDEX"
    echo "INDEX_TRAILING_NEWLINE_ADDED"
  fi
  BEFORE=$(wc -l < "$INDEX")
  # Idempotent append: skip bare accessions already present.
  awk -F'\t' 'NR==FNR{seen[$1]=1; next} !($1 in seen)' "$INDEX" \
      "$WORK/appended_index_rows.tsv" >> "$INDEX"
  AFTER=$(wc -l < "$INDEX")
  echo "INDEX_BEFORE $BEFORE"
  echo "INDEX_AFTER $AFTER"
  echo "INDEX_ADDED $((AFTER-BEFORE))"
  echo "INDEX_CU_ROWS $(grep -c '^CUST_' "$INDEX" || true)"
fi

echo "TOTAL_PROTEINS $(awk -F'\t' 'NR>1{s+=$4} END{print s+0}' "$WORK/ingest_report.tsv")"
echo "FNA_ON_DISK $(find "$CUSTOM/database" -name '*_genomic.fna.gz' | wc -l)"
echo "FAA_ON_DISK $(find "$CUSTOM/protein_faa_reps" -name '*_protein.faa.gz' | wc -l)"
echo "INGEST_DONE"
[ "$N_FAIL" -eq 0 ] || exit 1
