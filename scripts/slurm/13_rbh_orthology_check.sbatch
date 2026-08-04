#!/bin/bash
#SBATCH --job-name=eptrans_rbh
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/groups/cress/projects/jaymin/eptrans_scratch/logs/rbh/rbh_%j.out
#
# Stage 13 - is the identity-threshold yield gain REAL ORTHOLOGY or spurious merging?
#
# The 12_identity_sweep measured that hyperthermophile pairs/gp rises 43.2 -> 122.8
# -> 210.5 at 50/40/30% identity (4.87x, vs 1.30-1.70x for the four
# genus/family-matched classes). That measured YIELD ONLY. Lowering the identity
# threshold merges clusters (224,022 -> 130,712, max prevalence 0.684 -> 0.867), and
# a merged cluster can contain PARALOGS or remote homologs, in which case
# _derive_protein_pairs emits a pair that is not an ortholog contrast at all.
#
# Ground truth here is mmseqs easy-rbh: reciprocal best hit between the two
# proteomes of each matched genome pair. A derived pair is called CORRECT when the
# two proteins are each other's reciprocal best hit.
#
# TWO DESIGN POINTS THAT DECIDE WHETHER THE ANSWER TRANSFERS:
#
#  1. --cov-mode 0, NOT 1. The probe and the sweep both used --cov-mode 1
#     (target coverage), but PRODUCTION stage 07 clusters with --cov-mode 0
#     (bidirectional). Sweep pair counts therefore do not transfer to production.
#     This job re-clusters at cov-mode 0 so precision AND yield are both measured
#     in the regime the real assembly will use.
#
#  2. Precision is reported on the pairs UNIQUE to each threshold, not on all
#     pairs. Pairs already present at 50% are not in question; the decision rests
#     entirely on whether the ADDED pairs are orthologs. Reporting pooled precision
#     would dilute the added pairs with the known-good baseline and inflate it.
#
# Also emits the RBH verdict for the 50% baseline itself, so "is cs_prob
# tie-breaking picking the right representative?" gets a number rather than an
# assumption.
set -uo pipefail   # deliberately NO -e: a failed threshold must not kill the rest

MM=/shared/software/bin/mmseqs
P=/groups/cress/projects/jaymin/eptrans_scratch/prevalence_probe
W=/groups/cress/projects/jaymin/eptrans_scratch/rbh
THREADS=${SLURM_CPUS_PER_TASK:-32}
mkdir -p "$W" "$(dirname /groups/cress/projects/jaymin/eptrans_scratch/logs/rbh/x)"
cd "$W" || exit 1

echo "HOST $(hostname) THREADS $THREADS"
echo "PROBE_FAA $(grep -c '>' $P/all.faa)"
cp "$P/probe_pairs.tsv" "$W/pairs.tsv"

# ---------------------------------------------------------------- 1. per-genome FASTAs
# RBH is pairwise between proteomes, so split the pooled probe FASTA by genome.
# Headers are {GENOME}~{PROTID}; the genome is everything before the first '~'.
if [ ! -s "$W/.split_done" ]; then
  mkdir -p "$W/genomes"
  awk -v D="$W/genomes" '
    /^>/ { id=substr($0,2); split(id,a,"~"); g=a[1];
           gsub(/[^A-Za-z0-9._-]/,"_",g); f=D"/"g".faa"; print > f; next }
    { print > f }' "$P/all.faa"
  ls "$W/genomes" | wc -l > "$W/.split_done"
fi
echo "GENOME_FASTAS $(cat $W/.split_done)"

# ---------------------------------------------------------------- 2. cluster at each id, cov-mode 0
for SID in 0.5 0.4 0.3; do
  TAG="id$(echo "$SID" | tr -d '0.')"
  if [ -s "$W/clu_${TAG}_cluster.tsv" ]; then echo "SKIP_CLUSTER $TAG"; continue; fi
  echo "=== CLUSTER $TAG (--min-seq-id $SID -c 0.8 --cov-mode 0, production regime) ==="
  /usr/bin/time -v "$MM" easy-cluster "$P/all.faa" "$W/clu_${TAG}" "$W/tmp_${TAG}" \
      --min-seq-id "$SID" -c 0.8 --cov-mode 0 --threads "$THREADS" \
      > "$W/mmseqs_${TAG}.log" 2> "$W/mmseqs_${TAG}.time"
  echo "CLUSTER_RC_${TAG} $?"
  rm -rf "$W/tmp_${TAG}"
  echo "  clusters $(cut -f1 $W/clu_${TAG}_cluster.tsv | sort -u | wc -l)"
done

# ---------------------------------------------------------------- 3. RBH per matched genome pair
# One easy-rbh per (extremophile, outgroup) proteome pair, reused across all three
# thresholds -- RBH is a property of the two proteomes, independent of clustering.
mkdir -p "$W/rbh"
n=0
while IFS=$'\t' read -r CLS EXT OUT REST; do
  [ "$CLS" = "class" ] && continue
  [ -z "${EXT:-}" ] && continue
  # strip GB_/RS_/CU_ prefix to match the FASTA-derived filenames
  EB=$(echo "$EXT" | sed -E 's/^(GB_|RS_|CU_)//')
  OB=$(echo "$OUT" | sed -E 's/^(GB_|RS_|CU_)//')
  FE="$W/genomes/${EB}.faa"; FO="$W/genomes/${OB}.faa"
  OUTF="$W/rbh/${EB}__${OB}.tsv"
  if [ -s "$OUTF" ]; then n=$((n+1)); continue; fi
  if [ ! -s "$FE" ] || [ ! -s "$FO" ]; then
    echo "MISSING_PROTEOME $CLS $EB $OB"; continue
  fi
  "$MM" easy-rbh "$FE" "$FO" "$OUTF" "$W/tmprbh" \
      --threads "$THREADS" -e 1e-5 > /dev/null 2>&1
  rc=$?; rm -rf "$W/tmprbh"
  [ $rc -ne 0 ] && echo "RBH_FAIL $CLS $EB $OB rc=$rc"
  n=$((n+1))
done < "$W/pairs.tsv"
echo "RBH_PAIRS_DONE $n"
echo "RBH_FILES $(ls $W/rbh 2>/dev/null | wc -l)"

# ---------------------------------------------------------------- 4. score
python3 - <<'PY'
import os, csv, json, collections
W = "/groups/cress/projects/jaymin/eptrans_scratch/rbh"

def strip(a):
    for p in ("GB_", "RS_", "CU_"):
        if a.startswith(p):
            return a[len(p):]
    return a

pairs = []
with open(f"{W}/pairs.tsv") as fh:
    for row in csv.reader(fh, delimiter="\t"):
        if not row or row[0] == "class":
            continue
        pairs.append((row[0], strip(row[1]), strip(row[2])))
n_by_class = collections.Counter(c for c, _, _ in pairs)

# RBH sets: (query_id, target_id) both directions, keyed by genome pair
rbh = {}
for cls, e, m in pairs:
    f = f"{W}/rbh/{e}__{m}.tsv"
    s = set()
    if os.path.exists(f):
        with open(f) as fh:
            for line in fh:
                p = line.split("\t")
                if len(p) >= 2:
                    s.add((p[0], p[1])); s.add((p[1], p[0]))
    rbh[(e, m)] = s

def load_clusters(tag):
    g2c, c2m = {}, collections.defaultdict(set)
    with open(f"{W}/clu_{tag}_cluster.tsv") as fh:
        for line in fh:
            rep, mem = line.rstrip("\n").split("\t")[:2]
            g2c[mem] = rep
            c2m[rep].add(mem)
    return g2c, c2m

def derive(tag):
    """Reproduce _derive_protein_pairs: one protein per (genome, cluster),
    deterministic lexicographic representative (the whole-proteome rule)."""
    g2c, _ = load_clusters(tag)
    bygen = collections.defaultdict(lambda: collections.defaultdict(list))
    for pid, rep in g2c.items():
        gen = pid.split("~")[0]
        bygen[gen][rep].append(pid)
    best = {g: {c: sorted(v)[0] for c, v in d.items()} for g, d in bygen.items()}
    out = collections.defaultdict(list)
    for cls, e, m in pairs:
        be, bm = best.get(e, {}), best.get(m, {})
        for cl in set(be) & set(bm):
            out[cls].append((cl, be[cl], bm[cl], (e, m)))
    return out

res = {}
for tag in ("id5", "id4", "id3"):
    if not os.path.exists(f"{W}/clu_{tag}_cluster.tsv"):
        continue
    res[tag] = derive(tag)

def precision(rows):
    ok = sum(1 for _, a, b, gp in rows if (a, b) in rbh.get(gp, ()))
    return ok, len(rows), (100.0 * ok / len(rows) if rows else float("nan"))

report = {"n_by_class": dict(n_by_class), "thresholds": {}, "unique_vs_id5": {}}
print("=== PRECISION ON ALL DERIVED PAIRS (cov-mode 0) ===")
print(f"{'tag':5s} {'class':18s} {'pairs':>7s} {'per_gp':>7s} {'rbh_ok':>7s} {'prec%':>7s}")
for tag in ("id5", "id4", "id3"):
    if tag not in res: continue
    report["thresholds"][tag] = {}
    for cls in sorted(n_by_class):
        rows = res[tag].get(cls, [])
        ok, tot, pc = precision(rows)
        pg = tot / n_by_class[cls]
        report["thresholds"][tag][cls] = dict(pairs=tot, per_gp=round(pg, 1),
                                              rbh_ok=ok, precision=round(pc, 2))
        print(f"{tag:5s} {cls:18s} {tot:7d} {pg:7.1f} {ok:7d} {pc:7.2f}")

print("\n=== PRECISION ON PAIRS *UNIQUE* TO THE LOWER THRESHOLD (the decision) ===")
print(f"{'tag':5s} {'class':18s} {'added':>7s} {'rbh_ok':>7s} {'prec%':>7s}")
if "id5" in res:
    base = {cls: set((a, b) for _, a, b, _ in res["id5"].get(cls, []))
            for cls in n_by_class}
    for tag in ("id4", "id3"):
        if tag not in res: continue
        report["unique_vs_id5"][tag] = {}
        for cls in sorted(n_by_class):
            rows = [r for r in res[tag].get(cls, [])
                    if (r[1], r[2]) not in base.get(cls, set())]
            ok, tot, pc = precision(rows)
            report["unique_vs_id5"][tag][cls] = dict(added=tot, rbh_ok=ok,
                                                     precision=round(pc, 2))
            print(f"{tag:5s} {cls:18s} {tot:7d} {ok:7d} {pc:7.2f}")

with open(f"{W}/rbh_report.json", "w") as fh:
    json.dump(report, fh, indent=1, sort_keys=True)
print("\nWROTE rbh_report.json")
PY
echo "RBH_CHECK_DONE"
