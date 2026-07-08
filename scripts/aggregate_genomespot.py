#!/usr/bin/env python3
"""Aggregate GenomeSPOT per-genome predictions.tsv into one wide TSV.
Pivots each genome's long-format (target/value/error/units/is_novel/warning, 10 rows)
into one row per genome. Joins genome absolute paths from genome_index.tsv (the
02b path-attachment folded into the recompute output). Writes headers.
"""
import os, glob, sys, csv

GS = "/groups/cress/projects/jaymin/eptrans_scratch/genomespot"
GENOME_INDEX = "/groups/cress/projects/jaymin/IS1111/work/genome_index.tsv"
OUT = "/groups/cress/projects/jaymin/eptrans_scratch/genomespot_predictions_r232.tsv"

TARGETS = ["temperature_optimum","temperature_min","temperature_max",
           "ph_optimum","ph_min","ph_max",
           "salinity_optimum","salinity_min","salinity_max","oxygen"]

# genome_index: bare_acc -> abs fna path
print("loading genome_index...", flush=True)
paths = {}
with open(GENOME_INDEX) as fh:
    for line in fh:
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2:
            paths[p[0]] = p[1]
print(f"  {len(paths)} paths", flush=True)

files = glob.glob(f"{GS}/chunk_*/*.predictions.tsv")
print(f"found {len(files)} prediction files", flush=True)

# output columns: bare accession + per-target value+warning + genome path
cols = ["accession"]
for t in TARGETS:
    cols += [f"{t}", f"{t}_error", f"{t}_warning"]
cols += ["genome_fna_path"]

n_ok = 0; n_err = 0
with open(OUT, "w", newline="") as out:
    w = csv.writer(out, delimiter="\t")
    w.writerow(cols)
    for i, f in enumerate(files):
        bare = os.path.basename(f).replace(".predictions.tsv","")
        rec = {}
        try:
            with open(f) as fh:
                r = csv.DictReader(fh, delimiter="\t")
                for row in r:
                    rec[row["target"]] = row
        except Exception:
            n_err += 1; continue
        out_row = [bare]
        for t in TARGETS:
            rr = rec.get(t, {})
            out_row += [rr.get("value",""), rr.get("error",""), rr.get("warning","")]
        out_row += [paths.get(bare, "")]
        w.writerow(out_row)
        n_ok += 1
        if (i+1) % 20000 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)

print(f"DONE: wrote {n_ok} genomes ({n_err} errors) to {OUT}", flush=True)
# quick path-attach stat
with open(OUT) as fh:
    next(fh)
    withpath = sum(1 for line in fh if line.rstrip("\n").split("\t")[-1])
print(f"genome paths attached: {withpath}/{n_ok}", flush=True)
