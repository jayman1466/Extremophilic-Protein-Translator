#!/usr/bin/env python3
"""Aggregate GenomeSPOT predictions for the deep-sea MAG set into one wide TSV.

Same pivot as aggregate_genomespot.py (long 10-row target/value/error form -> one
row per genome), with three differences forced by the MAG input:

  1. Identity is the MAG basename, not a GTDB accession. MAG names contain
     parentheses and hyphens -- e.g. FDZ071-WW16-18(OES00301993)_bin.24 -- so the
     name is emitted verbatim and never parsed by regex.
  2. There is no genome_index.tsv entry, so the contig path is recorded directly.
  3. The sample id is split out (the part before the final _bin.<N>) so the result
     joins to deepsea_sample_labels.tsv, which is keyed by sample rather than MAG.

Writes accession-style column names so downstream stages (03_combine_bins) can
consume this file with the same reader as the r232 predictions table.
"""
import csv
import glob
import os
import sys

GS = "/groups/cress/projects/jaymin/eptrans_scratch/genomespot_mags"
MAG_DIR = "/groups/cress/projects/jaymin/IS1111/deepsea_extract/HQ_MAGs_drep"
OUT = "/groups/cress/projects/jaymin/eptrans_scratch/genomespot_predictions_deepsea_mags.tsv"

TARGETS = ["temperature_optimum", "temperature_min", "temperature_max",
           "ph_optimum", "ph_min", "ph_max",
           "salinity_optimum", "salinity_min", "salinity_max", "oxygen"]


def sample_of(mag_name: str) -> str:
    """'FDZ071-WW16-18(OES00301993)_bin.24' -> 'FDZ071-WW16-18(OES00301993)'.

    Split on the LAST '_bin.' so sample ids containing '_bin' (none observed, but
    the MAG ids are third-party strings) survive intact.
    """
    idx = mag_name.rfind("_bin.")
    return mag_name[:idx] if idx > 0 else mag_name


files = sorted(glob.glob(f"{GS}/chunk_*/*.predictions.tsv"))
print(f"found {len(files)} prediction files", flush=True)
if not files:
    sys.exit("no prediction files; did the array job run?")

cols = ["mag_id", "sample_id"]
for t in TARGETS:
    cols += [t, f"{t}_error", f"{t}_warning"]
cols += ["mag_fna_path"]

n_ok = 0
n_err = 0
n_missing_targets = 0
with open(OUT, "w", newline="") as out:
    w = csv.writer(out, delimiter="\t")
    w.writerow(cols)
    for f in files:
        mag = os.path.basename(f).replace(".predictions.tsv", "")
        rec = {}
        try:
            with open(f) as fh:
                for row in csv.DictReader(fh, delimiter="\t"):
                    rec[row["target"]] = row
        except Exception:
            n_err += 1
            continue
        if not all(t in rec for t in TARGETS):
            n_missing_targets += 1
        row_out = [mag, sample_of(mag)]
        for t in TARGETS:
            rr = rec.get(t, {})
            row_out += [rr.get("value", ""), rr.get("error", ""), rr.get("warning", "")]
        fna = os.path.join(MAG_DIR, mag + ".fa")
        row_out += [fna if os.path.exists(fna) else ""]
        w.writerow(row_out)
        n_ok += 1

print(f"DONE: wrote {n_ok} MAGs ({n_err} unreadable, "
      f"{n_missing_targets} with incomplete target sets) to {OUT}", flush=True)

# Coverage against the full MAG set, and the cold-end distribution that motivated
# this run -- printed here so the miscalibration check is immediate.
all_mags = glob.glob(os.path.join(MAG_DIR, "*.fa"))
print(f"coverage: {n_ok}/{len(all_mags)} MAGs predicted", flush=True)

temps = []
with open(OUT) as fh:
    r = csv.DictReader(fh, delimiter="\t")
    for row in r:
        v = row.get("temperature_optimum", "")
        if v:
            try:
                temps.append(float(v))
            except ValueError:
                pass
if temps:
    temps.sort()
    n = len(temps)
    def q(p):
        return temps[min(n - 1, int(p * n))]
    print(f"predicted temperature_optimum over {n} MAGs: "
          f"min {temps[0]:.1f}  p05 {q(.05):.1f}  median {q(.5):.1f}  "
          f"p95 {q(.95):.1f}  max {temps[-1]:.1f}", flush=True)
    print(f"  <=15C (psychrophile threshold): {sum(t <= 15 for t in temps)}", flush=True)
    print(f"  >=50C (thermophile threshold):  {sum(t >= 50 for t in temps)}", flush=True)
    print(f"  >=80C (hyperthermophile):       {sum(t >= 80 for t in temps)}", flush=True)
