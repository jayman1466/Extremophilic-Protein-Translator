#!/usr/bin/env python3
"""Stage 04 - phylogenetically-controlled genome selection.

Selects diversity-capped extremophiles per class and pairs each with a
phylogenetically-close confident-mesophile outgroup (same genus/family/... where
possible), so downstream models cannot separate the extremophile trait by clade
alone.

Outputs
-------
    <out_prefix>.extremophiles.tsv
    <out_prefix>.outgroups.tsv
    <out_prefix>.pairs.tsv
    <out_prefix>.stats.json
    <fig_ranks>     distribution of the phylogenetic rank at which each pair matched
    <fig_spread>    phylum spread of selected extremophiles per class

Usage
-----
    python scripts/04_select_genomes.py \
        --labels results/combined_labels.parquet \
        --out-prefix results/selection \
        --max-per-lineage 5 --lineage-rank family --max-total-per-class 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eptrans.config import load_config
from eptrans.selection import select_with_outgroups

_RANK_ORDER = ["genus", "family", "order", "class", "phylum"]


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", required=True, help="combined labels parquet/TSV (stage 03)")
    ap.add_argument("--out-prefix", default="results/selection")
    ap.add_argument("--max-per-lineage", type=int,
                    default=cfg.get_path("selection.max_per_lineage", 5))
    ap.add_argument("--lineage-rank", default=cfg.get_path("selection.max_per_lineage_rank", "family"))
    ap.add_argument("--max-total-per-class", type=int, default=100)
    ap.add_argument("--classes", default=None,
                    help="comma-separated class list (default: all 6)")
    ap.add_argument("--confidence", default=None,
                    help="confidence tiers to keep. Global: 'high,medium'. Per-class: "
                         "'default=high,medium;thermophile=high' (semicolon-separated, "
                         "class=tiers, 'default' sets the fallback).")
    ap.add_argument("--reuse-outgroups", action="store_true",
                    help="allow one mesophile outgroup to pair across multiple classes "
                         "(deduplicated in the final outgroup set)")
    ap.add_argument("--seed", type=int, default=cfg.get_path("dataset.split_seed", 1466))
    ap.add_argument("--fig-ranks", default="results/selection_match_ranks.png")
    ap.add_argument("--fig-spread", default="results/selection_phylum_spread.png")
    args = ap.parse_args()

    labels = (pd.read_parquet(args.labels) if args.labels.endswith(".parquet")
              else pd.read_csv(args.labels, sep="\t"))

    classes = args.classes.split(",") if args.classes else None
    conf = None
    if args.confidence:
        if "=" in args.confidence:
            # per-class: 'default=high,medium;thermophile=high'
            spec = {}
            for part in args.confidence.split(";"):
                k, v = part.split("=", 1)
                spec[k.strip()] = tuple(v.split(","))
            default = spec.pop("default", None)
            cls_list = classes or ["thermophile", "hyperthermophile", "psychrophile",
                                   "acidophile", "alkaliphile", "halophile"]
            conf = {c: spec.get(c, default) for c in cls_list}
        else:
            conf = tuple(args.confidence.split(","))
    res = select_with_outgroups(
        labels, classes=classes, max_per_lineage=args.max_per_lineage,
        lineage_rank=args.lineage_rank, max_total_per_class=args.max_total_per_class,
        confidence_levels=conf, reuse_outgroups=args.reuse_outgroups, seed=args.seed,
    )

    pref = Path(args.out_prefix)
    pref.parent.mkdir(parents=True, exist_ok=True)
    res.extremophiles.to_csv(f"{pref}.extremophiles.tsv", sep="\t", index=False)
    res.outgroups.to_csv(f"{pref}.outgroups.tsv", sep="\t", index=False)
    res.pairs.to_csv(f"{pref}.pairs.tsv", sep="\t", index=False)
    json.dump(res.stats, open(f"{pref}.stats.json", "w"), indent=2)

    print(f"[04] extremophiles: {res.stats['n_extremophiles']:,} | "
          f"outgroups: {res.stats['n_outgroups']:,} | "
          f"pairs matched: {res.stats['n_pairs_matched']:,} "
          f"(unmatched {res.stats['n_pairs_unmatched']:,})")
    print(f"[04] match-rank distribution: {res.stats['match_rank_dist']}")
    print(f"[04] wrote {pref}.{{extremophiles,outgroups,pairs}}.tsv + stats.json")

    # Figure 1: matched-rank distribution (phylogenetic closeness of pairs).
    ranks = [r for r in _RANK_ORDER if r in res.stats["match_rank_dist"]]
    vals = [res.stats["match_rank_dist"].get(r, 0) for r in ranks]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(ranks)))
    bars = ax.bar(ranks, vals, color=colors)
    ax.set_xlabel("Finest shared rank (extremophile ↔ outgroup)")
    ax.set_ylabel("Number of pairs")
    ax.set_title("Phylogenetic closeness of extremophile–outgroup pairs\n(finer = tighter clade control)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.fig_ranks, dpi=150)
    print(f"[04] wrote {args.fig_ranks}")

    # Figure 2: phylum spread of selected extremophiles per class (diversity check).
    if len(res.extremophiles):
        e = res.extremophiles
        classes = sorted(e["selected_class"].unique())
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        # number of distinct phyla per class
        nphyla = [e[e["selected_class"] == c]["phylum"].nunique() for c in classes]
        ngen = [e[e["selected_class"] == c]["genus"].nunique() for c in classes]
        x = np.arange(len(classes)); w = 0.38
        ax2.bar(x - w/2, nphyla, w, label="distinct phyla", color="#3a7ca5")
        ax2.bar(x + w/2, ngen, w, label="distinct genera", color="#81a4cd")
        ax2.set_xticks(x); ax2.set_xticklabels(classes, rotation=40, ha="right")
        ax2.set_ylabel("Distinct taxa among selected")
        ax2.set_title("Phylogenetic diversity of selected extremophiles per class")
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig(args.fig_spread, dpi=150)
        print(f"[04] wrote {args.fig_spread}")


if __name__ == "__main__":
    main()
