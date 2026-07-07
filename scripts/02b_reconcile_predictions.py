#!/usr/bin/env python3
"""Stage 02b - reconcile precomputed GenomeSPOT predictions across releases.

Reuses the GenomeSPOT paper's published predictions (GTDB r214, Supplementary
Data 4) for r232 representatives via three-tier accession matching, computes the
recompute delta, attaches genome absolute paths from genome_index.tsv, and
writes a reconciled TSV (with headers).

Outputs
-------
    <out>                       reconciled predictions TSV (headers; incl. genome paths)
    <out>.delta_accessions.txt  accessions still needing a fresh GenomeSPOT run
    <out>.stats.json            reuse statistics
    <fig>                       reuse/delta summary bar chart

Usage
-----
    python scripts/02b_reconcile_predictions.py \
        --precomputed supplementary_data_4.tsv \
        --reps data/reps_flagcols.tsv \
        --genome-index /path/to/genome_index.tsv \
        --out results/genomespot_reconciled_r232.tsv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from eptrans.config import load_config
from eptrans.reconcile import reconcile, attach_genome_paths, MATCH_LEVELS

# Prediction columns we expect from the precomputed table (paper Supp Data 4).
DEFAULT_PRED_COLS = [
    "oxygen", "temperature_optimum", "temperature_min", "temperature_max",
    "ph_optimum", "ph_min", "ph_max",
    "salinity_optimum", "salinity_min", "salinity_max",
]


def _read_any(path: str) -> pd.DataFrame:
    sep = "," if str(path).endswith(".csv") else "\t"
    return pd.read_csv(path, sep=sep, dtype=str)


def _detect_acc_col(df: pd.DataFrame, hint: str | None) -> str:
    if hint and hint in df.columns:
        return hint
    for cand in ["accession", "ncbi_accession", "genome", "assembly_accession", "gtdb_accession"]:
        if cand in df.columns:
            return cand
    # first column fallback
    return df.columns[0]


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--precomputed", required=True, help="precomputed predictions TSV/CSV (paper Supp Data 4)")
    ap.add_argument("--reps", required=True, help="r232 reps table (parquet or TSV with accession col)")
    ap.add_argument("--genome-index", default=cfg.get_path("biotite.genome_index"))
    ap.add_argument("--reps-acc-col", default="accession")
    ap.add_argument("--precomp-acc-col", default=None, help="auto-detected if omitted")
    ap.add_argument("--pred-cols", nargs="*", default=None,
                    help="prediction columns to carry (auto-intersect with defaults if omitted)")
    ap.add_argument("--out", default="results/genomespot_reconciled_r232.tsv")
    ap.add_argument("--fig", default="results/genomespot_reconcile_summary.png")
    args = ap.parse_args()

    # Load reps.
    reps = (pd.read_parquet(args.reps) if args.reps.endswith(".parquet")
            else pd.read_csv(args.reps, sep="\t", dtype=str))
    # Load precomputed.
    precomp = _read_any(args.precomputed)
    pc_acc = _detect_acc_col(precomp, args.precomp_acc_col)
    print(f"[02b] reps: {len(reps):,} | precomputed: {len(precomp):,} (acc col: '{pc_acc}')")

    # Prediction columns: intersect requested/defaults with what's present.
    if args.pred_cols:
        pred_cols = [c for c in args.pred_cols if c in precomp.columns]
    else:
        pred_cols = [c for c in DEFAULT_PRED_COLS if c in precomp.columns]
        if not pred_cols:  # unknown schema: carry everything except acc col
            pred_cols = [c for c in precomp.columns if c != pc_acc]
    print(f"[02b] carrying {len(pred_cols)} prediction columns: {pred_cols}")

    res = reconcile(reps, precomp, reps_acc_col=args.reps_acc_col,
                    precomp_acc_col=pc_acc, prediction_cols=pred_cols)

    # Attach genome absolute paths (required output).
    reconciled = attach_genome_paths(res.reconciled, args.genome_index, acc_col=args.reps_acc_col)
    n_paths = int(reconciled["genome_fna_path"].notna().sum())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    reconciled.to_csv(out, sep="\t", index=False)

    delta_acc = out.with_name(out.stem + ".delta_accessions.txt")
    res.delta[args.reps_acc_col].to_csv(delta_acc, index=False, header=False)

    stats = dict(res.stats)
    stats["n_genome_paths_attached"] = n_paths
    stats_path = out.with_name(out.stem + ".stats.json")
    json.dump(stats, open(stats_path, "w"), indent=2)

    print(f"[02b] reused: {stats['n_reused']:,} ({stats['reuse_fraction']:.1%})  "
          f"delta: {stats['n_delta']:,}  dropped_precomp: {stats['n_dropped_precomputed']:,}")
    print(f"[02b] reuse by level: {stats['reuse_by_level']}")
    print(f"[02b] genome paths attached: {n_paths:,}/{len(reconciled):,}")
    print(f"[02b] wrote {out}\n[02b] wrote {delta_acc}\n[02b] wrote {stats_path}")

    # Figure: reuse breakdown + delta.
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"reuse:{lvl}" for lvl in MATCH_LEVELS] + ["delta (recompute)"]
    vals = [stats["reuse_by_level"][lvl] for lvl in MATCH_LEVELS] + [stats["n_delta"]]
    colors = ["#2a9d8f", "#8ab17d", "#e9c46a", "#e76f51"]
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylabel("Number of r232 representatives")
    ax.set_title(f"GenomeSPOT prediction reconciliation (r214→r232)\n"
                 f"reused {stats['n_reused']:,} / {stats['n_reps']:,} = {stats['reuse_fraction']:.1%}")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    figp = Path(args.fig)
    figp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figp, dpi=150)
    print(f"[02b] wrote {figp}")


if __name__ == "__main__":
    main()
