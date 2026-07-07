#!/usr/bin/env python3
"""Stage 01b - metadata-based extremophile flagging.

Applies the curated keyword dictionary (eptrans.binning) to GTDB representatives
and writes per-genome metadata flags + a stacked bar chart of class counts.

Input can be either the full metadata parquet from stage 01, or a compact
columns-only TSV (domain, accession, ncbi_isolation_source, ncbi_organism_name,
gtdb_taxonomy) staged from biotite.

Usage
-----
    python scripts/01b_flag_metadata.py --tsv data/reps_flagcols.tsv \
        --out results/metadata_flags.parquet --fig results/metadata_flags_counts.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eptrans.binning import CLASSES, flag_dataframe
from eptrans.gtdb import expand_taxonomy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tsv", help="compact columns TSV (domain, accession, isolation, organism, taxonomy)")
    ap.add_argument("--parquet", help="full metadata parquet from stage 01 (alternative to --tsv)")
    ap.add_argument("--out", default="results/metadata_flags.parquet")
    ap.add_argument("--fig", default="results/metadata_flags_counts.png")
    args = ap.parse_args()

    if args.tsv:
        df = pd.read_csv(args.tsv, sep="\t", dtype=str,
                         na_values=["none", "None", "NA", ""], keep_default_na=True)
    elif args.parquet:
        df = pd.read_parquet(args.parquet)
    else:
        raise SystemExit("provide --tsv or --parquet")

    # The compact TSV carries a `domain` column; taxonomy expansion also emits
    # `domain` (same Bacteria/Archaea values). Expand, then drop duplicate cols
    # keeping the first occurrence.
    if "phylum" not in df.columns and "gtdb_taxonomy" in df.columns:
        df = expand_taxonomy(df, "gtdb_taxonomy")
        df = df.loc[:, ~df.columns.duplicated()]

    flagged = flag_dataframe(df)

    # Counts.
    n = len(flagged)
    iso_counts = {cls: int(flagged[f"meta_iso_{cls}"].sum()) for cls in CLASSES}
    org_counts = {cls: int(flagged[f"meta_org_{cls}"].sum()) for cls in CLASSES}
    print(f"[01b] representatives: {n:,}")
    print(f"[01b] with isolation_source: {int(df['ncbi_isolation_source'].notna().sum()):,}")
    print(f"[01b] iso-flagged (any class): {int(flagged['meta_iso_any'].sum()):,}")
    print(f"[01b] org-flagged (any class): {int(flagged['meta_org_any'].sum()):,}")
    print("[01b] isolation-source class counts:")
    for cls in CLASSES:
        print(f"        {cls:18s} iso={iso_counts[cls]:>7,}   org={org_counts[cls]:>7,}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    flagged.to_parquet(out, index=False)
    print(f"[01b] wrote {out}")

    # Stacked bar chart: iso vs org evidence per class, split by domain for iso.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: iso-source counts by domain (stacked).
    bac = [int(flagged[(flagged["domain"] == "Bacteria")][f"meta_iso_{c}"].sum()) for c in CLASSES]
    arc = [int(flagged[(flagged["domain"] == "Archaea")][f"meta_iso_{c}"].sum()) for c in CLASSES]
    x = np.arange(len(CLASSES))
    axes[0].bar(x, bac, label="Bacteria", color="#3a7ca5")
    axes[0].bar(x, arc, bottom=bac, label="Archaea", color="#d1495b")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(CLASSES, rotation=40, ha="right")
    axes[0].set_ylabel("Genomes flagged (isolation source)")
    axes[0].set_title("Metadata isolation-source flags")
    axes[0].legend()
    for i, (b, a) in enumerate(zip(bac, arc)):
        if b + a > 0:
            axes[0].text(i, b + a, f"{b+a:,}", ha="center", va="bottom", fontsize=8)

    # Right: iso vs organism-name evidence (grouped).
    w = 0.4
    axes[1].bar(x - w/2, [iso_counts[c] for c in CLASSES], w, label="isolation source", color="#3a7ca5")
    axes[1].bar(x + w/2, [org_counts[c] for c in CLASSES], w, label="organism name", color="#edae49")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(CLASSES, rotation=40, ha="right")
    axes[1].set_ylabel("Genomes flagged")
    axes[1].set_title("Evidence source: isolation vs organism name")
    axes[1].legend()

    fig.suptitle(f"Metadata-based extremophile flags (GTDB reps, n={n:,})")
    fig.tight_layout()
    figp = Path(args.fig)
    figp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figp, dpi=150)
    print(f"[01b] wrote {figp}")


if __name__ == "__main__":
    main()
