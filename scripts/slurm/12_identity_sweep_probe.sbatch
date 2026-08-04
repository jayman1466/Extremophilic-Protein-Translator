#!/bin/bash
#SBATCH --job-name=eptrans_idsweep
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/groups/cress/projects/jaymin/eptrans_scratch/logs/idsweep/idsweep_%j.out
#
# Does lowering the mmseqs identity threshold recover cross-phylum core pairs?
#
# MOTIVATION (measured, see labnotebook "Why protein pairs per genome pair don't
# track the ~300-core-gene expectation"): whole-proteome pairs/gp tracks outgroup
# match rank monotonically (Spearman rho = 1.000, Pearson r = 0.973, n = 5).
# Hyperthermophile yields 43.2 pairs/gp against 334-540 for the other four, and it
# is 65.0% phylum-matched vs ~30% genus+family for the others. The hypothesis is
# that its shared core genes ARE present in both genomes but fail to co-cluster at
# 50% identity, which sits near the twilight zone for cross-phylum orthologs.
#
# DESIGN. Re-cluster the SAME 158-genome probe proteome (323,178 proteins,
# all.faa, already on disk) at 30 / 40 / 50% identity, changing ONE parameter.
# Every other flag is byte-identical to the original probe run
# (-c 0.8 --cov-mode 1), so the comparison is controlled and 50% reproduces the
# published 43.2 as a positive control.
#
# All five classes are recomputed at every threshold, not just hyperthermophile.
# This is the control that decides the question: if a lower threshold lifts every
# class proportionally it is a global sensitivity knob and buys nothing for the
# confounding problem. It is only a targeted fix if hyperthermophile -- the
# phylum-matched class -- gains disproportionately.
#
# SENSITIVITY CONFOUND. mmseqs prefilters before aligning, and at 30% identity the
# default sensitivity can miss remote homologs for reasons unrelated to the
# identity cutoff -- which would bias the result AGAINST the hypothesis. So 0.3 is
# additionally run at -s 7.5 to separate "the threshold is the binding constraint"
# from "the prefilter is".
#
# `set -e` omitted: one failing threshold must not discard the others.
set -uo pipefail

P="/groups/cress/projects/jaymin/eptrans_scratch/prevalence_probe"
W="/groups/cress/projects/jaymin/eptrans_scratch/idsweep"
MM="/shared/software/bin/mmseqs"
FAA="$P/all.faa"
PAIRS="$P/probe_pairs.tsv"

mkdir -p "$W"
cd "$W" || exit 1

echo "INPUT_PROTEINS $(grep -c '^>' "$FAA")"
echo "INPUT_PAIRS $(( $(wc -l < "$PAIRS") - 1 ))"

# runs: <tag> <min-seq-id> <extra flags>
run_one () {
  TAG="$1"; SID="$2"; shift 2
  EXTRA=("$@")
  OUT="$W/clu_${TAG}"
  if [ -s "${OUT}_cluster.tsv" ]; then
    echo "SKIP $TAG (already present)"; return 0
  fi
  rm -rf "$W/tmp_$TAG"; mkdir -p "$W/tmp_$TAG"
  echo "RUN $TAG min-seq-id=$SID extra='${EXTRA[*]:-none}'"
  T0=$(date +%s)
  if [ "${#EXTRA[@]}" -gt 0 ]; then
    "$MM" easy-cluster "$FAA" "$OUT" "$W/tmp_$TAG" \
      --min-seq-id "$SID" -c 0.8 --cov-mode 1 "${EXTRA[@]}" \
      --threads 32 -v 1 > "$W/mmseqs_${TAG}.log" 2>&1
  else
    "$MM" easy-cluster "$FAA" "$OUT" "$W/tmp_$TAG" \
      --min-seq-id "$SID" -c 0.8 --cov-mode 1 \
      --threads 32 -v 1 > "$W/mmseqs_${TAG}.log" 2>&1
  fi
  RC=$?
  echo "RC_${TAG} $RC  SECONDS $(( $(date +%s) - T0 ))"
  [ "$RC" -eq 0 ] && echo "CLUSTERS_${TAG} $(cut -f1 "${OUT}_cluster.tsv" | sort -u | wc -l)"
  rm -rf "$W/tmp_$TAG"
  return 0
}

run_one id50 0.5
run_one id40 0.4
run_one id30 0.3
run_one id30s75 0.3 -s 7.5

# ---- pairs/gp per class at each threshold, same logic as the original calc.sh:
# a pair contributes one protein pair per cluster the two genomes SHARE.
python3 - <<'PY'
import csv, collections, json, os
W = "/groups/cress/projects/jaymin/eptrans_scratch/idsweep"
P = "/groups/cress/projects/jaymin/eptrans_scratch/prevalence_probe"

pairs = []
with open(f"{P}/probe_pairs.tsv") as fh:
    for row in csv.DictReader(fh, delimiter="\t"):
        pairs.append((row["class"], row["extremophile_acc"], row["outgroup_acc"]))
n_by_class = collections.Counter(c for c, _, _ in pairs)

out = {}
for tag in ("id50", "id40", "id30", "id30s75"):
    f = f"{W}/clu_{tag}_cluster.tsv"
    if not os.path.exists(f):
        print(f"MISSING {tag}"); continue
    g2c = collections.defaultdict(set)
    with open(f) as fh:
        for rep, mem in csv.reader(fh, delimiter="\t"):
            g2c[mem.split("~", 1)[0]].add(rep)
    prev = collections.Counter()
    for g, cs in g2c.items():
        for c_ in cs:
            prev[c_] += 1
    ng = len(g2c)
    per_class = collections.Counter()
    missing = 0
    for cls, e, m in pairs:
        if e not in g2c or m not in g2c:
            missing += 1; continue
        per_class[cls] += len(g2c[e] & g2c[m])
    maxprev = max(prev.values()) / ng if prev else 0.0
    out[tag] = {
        "genomes": ng,
        "clusters": len(prev),
        "singletons": sum(1 for v in prev.values() if v == 1),
        "max_prevalence": round(maxprev, 4),
        "missing_genomes_in_pairs": missing,
        "pairs_total": sum(per_class.values()),
        "per_class": {c: per_class[c] for c in sorted(n_by_class)},
        "per_gp": {c: round(per_class[c] / n_by_class[c], 1) for c in sorted(n_by_class)},
    }

print("=== RESULTS ===")
print(json.dumps(out, indent=1, sort_keys=True))
with open(f"{W}/idsweep_results.json", "w") as fh:
    json.dump({"n_by_class": dict(n_by_class), "runs": out}, fh, indent=1, sort_keys=True)

if "id50" in out and "id30" in out:
    print("=== FOLD CHANGE 30% vs 50% (the control comparison) ===")
    for c in sorted(n_by_class):
        a = out["id50"]["per_gp"][c]; b = out["id30"]["per_gp"][c]
        print(f"  {c:17s} {a:8.1f} -> {b:8.1f}   x{(b/a if a else float('nan')):.2f}")
PY

echo "IDSWEEP_DONE"
