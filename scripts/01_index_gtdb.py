#!/usr/bin/env python3
"""Stage 01 - index GTDB r232 metadata into a representatives table.

Parses the bacterial + archaeal metadata TSVs, filters to species
representatives, extracts environment/QC/taxonomy fields, and writes a parquet
table plus a per-phylum count summary and bar chart.

Runs on biotite against the real metadata (default paths from config), or
locally against a staged sample via --bac / --arc.

Usage
-----
    # on biotite (real data)
    python scripts/01_index_gtdb.py --out results/gtdb_reps_metadata.parquet

    # local validation against staged sample
    python scripts/01_index_gtdb.py \
        --bac data/sample/bac120_metadata_sample.tsv.gz \
        --arc data/sample/ar53_metadata_sample.tsv.gz \
        --out results/gtdb_reps_sample.parquet --no-qc
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from eptrans.config import load_config, repo_root
from eptrans.gtdb import load_representatives


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bac", default=cfg.get_path("biotite.metadata.bacteria"))
    ap.add_argument("--arc", default=cfg.get_path("biotite.metadata.archaea"))
    ap.add_argument("--out", default="results/gtdb_reps_metadata.parquet")
    ap.add_argument("--fig", default="results/gtdb_reps_per_phylum.png")
    ap.add_argument("--no-qc", action="store_true", help="skip completeness/contamination filter")
    args = ap.parse_args()

    df = load_representatives(bac_path=args.bac, arc_path=args.arc, apply_qc=not args.no_qc)
    print(f"[01] representatives: {len(df):,}  "
          f"(Bacteria={int((df['domain']=='Bacteria').sum()):,}, "
          f"Archaea={int((df['domain']=='Archaea').sum()):,})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"[01] wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")

    # Per-phylum summary.
    counts = (df.groupby("phylum").size().sort_values(ascending=False))
    summary = counts.reset_index(name="n_representatives")
    summary_path = out.with_name("gtdb_reps_per_phylum.tsv")
    summary.to_csv(summary_path, sep="\t", index=False)
    print(f"[01] wrote {summary_path}  ({len(summary)} phyla)")

    # Bar chart: top 20 phyla.
    top = counts.head(20)[::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top.index, top.values, color="#3a7ca5")
    ax.set_xlabel("Number of species representatives")
    ax.set_ylabel("GTDB phylum")
    ax.set_title(f"GTDB {cfg.get_path('gtdb.release')} representatives per phylum (top 20)")
    for i, v in enumerate(top.values):
        ax.text(v, i, f" {v:,}", va="center", fontsize=8)
    fig.tight_layout()
    figp = Path(args.fig)
    figp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figp, dpi=150)
    print(f"[01] wrote {figp}")


if __name__ == "__main__":
    main()
