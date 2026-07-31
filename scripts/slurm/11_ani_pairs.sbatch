#!/usr/bin/env bash
#SBATCH --job-name=eptrans_ani
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/groups/cress/projects/jaymin/eptrans_scratch/logs/ani/ani_%j.out
#SBATCH --error=/groups/cress/projects/jaymin/eptrans_scratch/logs/ani/ani_%j.out

# Whole-genome ANI for the selection's matched genome pairs, with controls.
#
# QUESTION. Stage 07 clusters proteins at 50% amino-acid identity / 80%
# coverage. That is a per-protein homology threshold; ANI is a genome-wide
# nucleotide average. They are not interchangeable, and nobody has checked how
# far apart they sit for OUR pairs. Two things could be wrong:
#   1. the clustering threshold could be admitting pairs so diverged that
#      "ortholog pair" overstates what is being compared
#   2. extremophile-mesophile pairs matched at a given rank could be
#      systematically MORE diverged than same-phenotype pairs matched at the
#      same rank -- which would mean the "phylogenetic control" is weaker than
#      the rank label implies, and part of what the classifier learns is
#      divergence rather than adaptation
#
# DESIGN. Treatment = extremophile/mesophile pairs actually used by selection,
# stratified across class x matched_rank. Controls, matched at the same ranks:
#   - meso_vs_meso        : two mesophiles (no phenotype contrast at all)
#   - same_pheno_halophile: two halophiles
#   - same_pheno_thermophile
# Control pairs are constructed to share the target rank but DIFFER at the
# next-finer rank, mirroring how selection.py picks an outgroup -- otherwise a
# "genus" control could be two strains of one species and would sit at 99% ANI
# for reasons that have nothing to do with phenotype.
#
# TOOL. skani, not fastANI: it is built for this scale, and unlike fastANI it
# reports aligned fraction alongside identity. Aligned fraction is the number
# that matters here -- two genomes can show high ANI over a small shared
# fraction, which is precisely the regime where a 50% AA cluster is doing
# something different from what "ortholog" suggests.
#
# skani dist is run ONCE over all pairs as a query/reference list rather than
# per pair: it sketches each genome once, so N sketches instead of 2*N_pairs.

set -uo pipefail   # no -e: a single unreadable genome must not abort the sweep

WORK=/groups/cress/projects/jaymin/eptrans_scratch/ani
GENOME_INDEX=/groups/cress/projects/jaymin/IS1111/work/genome_index.tsv
SKANI=/shared/software/bin/skani
CPUS=${SLURM_CPUS_PER_TASK:-32}

mkdir -p "$WORK"/{fna,out}
cd "$WORK"

echo "HOST $(hostname)"
echo "START $(date)"
"$SKANI" --version 2>&1 | head -1
echo "CPUS $CPUS"

# ---- resolve accessions to fna paths via the index ----
# genome_index.tsv is headerless: <bare_accession>\t<absolute .fna.gz path>.
# Accessions in ani_genomes.txt carry the GB_/RS_ prefix; strip it to join.
: > genome_paths.txt
MISSING=0
while IFS= read -r ACC; do
  [[ -z "$ACC" ]] && continue
  BARE="${ACC#GB_}"; BARE="${BARE#RS_}"; BARE="${BARE#CU_}"
  P=$(awk -F'\t' -v b="$BARE" '$1==b{print $2; exit}' "$GENOME_INDEX")
  if [[ -n "$P" && -f "$P" ]]; then
    printf '%s\t%s\n' "$ACC" "$P" >> genome_paths.txt
  else
    echo "MISSING_GENOME $ACC" ; MISSING=$((MISSING+1))
  fi
done < ani_genomes.txt
echo "GENOMES_RESOLVED $(wc -l < genome_paths.txt)"
echo "GENOMES_MISSING $MISSING"

# skani reads plain or gzipped fasta but needs a file list; build one and keep
# an accession<->path map so results can be joined back without parsing names.
cut -f2 genome_paths.txt > fna_list.txt

# ---- one triangle over all genomes ----
# Only ~500 of the 947*946/2 = 447,731 possible comparisons are wanted, but skani
# has no "just these pairs" mode. The alternative is ~500 separate `dist` calls,
# each re-reading and re-sketching two genomes. `triangle` sketches each genome
# exactly once and screens with FracMinHash before any alignment, so distant
# pairs cost almost nothing -- cheaper than 500 cold starts, and it yields the
# control comparisons for free. Filtering to our pairs happens downstream.
#
# -s 80 sets the screening floor: pairs below ~80% ANI are not reported. That is
# deliberate and must be remembered when reading the output -- an ABSENT pair
# means "below 80% ANI", not "failed". Distinguishing the two downstream matters,
# because pairs matched at order/class/phylum are expected to fall below it.
echo "DIST_START $(date +%s)"
/usr/bin/time -v "$SKANI" triangle \
    -l fna_list.txt \
    -t "$CPUS" \
    --sparse \
    --min-af 0 -s 80 \
    -o out/skani_all.tsv \
    > out/skani.stdout 2> out/skani.time
RC=$?
echo "DIST_END $(date +%s)"
echo "SKANI_RC $RC"
echo "SKANI_ROWS $(( $(wc -l < out/skani_all.tsv) - 1 ))"

echo "=== peak memory / wall ==="
grep -E 'Maximum resident set size|Elapsed \(wall clock\)' out/skani.time || true

echo "=== skani header ==="
head -1 out/skani_all.tsv

echo "=== ANI distribution over all reported comparisons ==="
awk -F'\t' 'NR>1{print $3}' out/skani_all.tsv \
  | sort -n \
  | awk '{a[NR]=$1} END{if(NR){printf "n=%d min=%.2f p05=%.2f p25=%.2f median=%.2f p75=%.2f p95=%.2f max=%.2f\n",
      NR,a[1],a[int(NR*0.05)+1],a[int(NR*0.25)+1],a[int(NR*0.5)+1],a[int(NR*0.75)+1],a[int(NR*0.95)+1],a[NR]}}'

echo "=== aligned-fraction distribution (query AF) ==="
awk -F'\t' 'NR>1{print $4}' out/skani_all.tsv \
  | sort -n \
  | awk '{a[NR]=$1} END{if(NR){printf "n=%d min=%.2f p25=%.2f median=%.2f p75=%.2f max=%.2f\n",
      NR,a[1],a[int(NR*0.25)+1],a[int(NR*0.5)+1],a[int(NR*0.75)+1],a[NR]}}'

gzip -f out/skani_all.tsv
echo "OUT out/skani_all.tsv.gz $(stat -c %s out/skani_all.tsv.gz 2>/dev/null || stat -f %z out/skani_all.tsv.gz) bytes"
echo "END $(date)"
echo "ANI_DONE"
