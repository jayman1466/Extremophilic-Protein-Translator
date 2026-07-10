#!/usr/bin/env python3
"""Stage 06 - assemble the labeled secreted-protein dataset with leakage-aware splits.

Joins secreted proteins (stage 05) to genome environmental classes (stage 03),
then assigns leakage-aware train/val/test splits (whole sequence-clusters or
genomes to a single split, stratified by class).

Outputs
-------
    <out>                 dataset parquet + TSV (protein rows + label + split)
    <out>.stats.json      counts by label / split / label x split
    <fig>                 label counts per split (stacked)

Usage
-----
    python scripts/06_assemble_dataset.py \
        --secreted results/secreted_proteins.tsv \
        --labels results/combined_labels.parquet \
        --cluster-map <mmseqs_cluster.tsv> \
        --out results/labeled_dataset.parquet
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
from eptrans.dataset import assemble_dataset


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--secreted", required=True, help="per-protein secreted table (stage 05)")
    ap.add_argument("--labels", required=True, help="combined labels parquet/TSV (stage 03)")
    ap.add_argument("--cluster-map", default=None,
                    help="mmseqs cluster TSV (cols: cluster_rep<TAB>member); optional")
    ap.add_argument("--pairs", default=None,
                    help="stage-04 pairs TSV (extremophile_acc, outgroup_acc); "
                         "co-assigns matched pairs to the same split")
    ap.add_argument("--protein-id-col", default="protein_id")
    ap.add_argument("--genome-col", default="genome")
    ap.add_argument("--multi-label", action="store_true")
    ap.add_argument("--out", default="results/labeled_dataset.parquet")
    ap.add_argument("--fig", default="results/dataset_splits.png")
    args = ap.parse_args()

    secreted = (pd.read_parquet(args.secreted) if args.secreted.endswith(".parquet")
                else pd.read_csv(args.secreted, sep="\t"))
    labels = (pd.read_parquet(args.labels) if args.labels.endswith(".parquet")
              else pd.read_csv(args.labels, sep="\t"))

    cluster_map = None
    if args.cluster_map:
        # mmseqs createtsv output: <cluster_rep>\t<member>
        cm = pd.read_csv(args.cluster_map, sep="\t", header=None, names=["cluster", "member"])
        cluster_map = cm

    pairs = None
    if args.pairs:
        pairs = pd.read_csv(args.pairs, sep="\t", dtype=str)

    res = assemble_dataset(
        secreted, labels, cluster_map=cluster_map, pairs=pairs,
        protein_id_col=args.protein_id_col, genome_col=args.genome_col,
        splits=cfg.get_path("dataset.splits", {"train": 0.8, "val": 0.1, "test": 0.1}),
        seed=cfg.get_path("dataset.split_seed", 1466),
        multi_label=args.multi_label,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.table.to_parquet(out, index=False)
    res.table.to_csv(out.with_suffix(".tsv"), sep="\t", index=False)
    json.dump(res.stats, open(out.with_suffix(".stats.json"), "w"), indent=2, default=str)

    print(f"[06] proteins: {res.stats['n_proteins']:,} | genomes: {res.stats['n_genomes']:,} "
          f"| groups: {res.stats['n_groups']:,} ({res.stats['group_kind']})")
    print(f"[06] label counts: {res.stats['label_counts']}")
    print(f"[06] split counts: {res.stats['split_counts']}")
    print(f"[06] max splits per group (leakage check, must be 1): {res.stats['max_splits_per_group']}")
    assert res.stats["max_splits_per_group"] <= 1, "LEAKAGE: a group spans multiple splits!"
    print(f"[06] wrote {out} and {out.with_suffix('.tsv')}")

    # Protein-level ortholog pairs (cluster regime) -> pairwise margin loss input.
    if res.protein_pairs is not None and len(res.protein_pairs):
        pp_path = out.with_name(out.stem + "_protein_pairs.tsv")
        res.protein_pairs.to_csv(pp_path, sep="\t", index=False)
        print(f"[06] protein pairs: {res.stats['n_protein_pairs']:,} "
              f"({res.stats['protein_pairs_same_split']:,} same-split) -> {pp_path}")

    # Figure: label counts per split.
    tab = res.table
    labels_u = sorted(tab["label"].unique())
    splits_u = ["train", "val", "test"]
    x = np.arange(len(labels_u))
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(labels_u))
    colors = {"train": "#2a9d8f", "val": "#e9c46a", "test": "#e76f51"}
    for sp in splits_u:
        vals = [int(((tab["label"] == lab) & (tab["split"] == sp)).sum()) for lab in labels_u]
        ax.bar(x, vals, bottom=bottom, label=sp, color=colors.get(sp))
        bottom += np.array(vals)
    ax.set_xticks(x); ax.set_xticklabels(labels_u, rotation=40, ha="right")
    ax.set_ylabel("Secreted proteins")
    ax.set_title(f"Labeled dataset: {res.stats['n_proteins']:,} secreted proteins "
                 f"({res.stats['group_kind']}-level leakage-aware splits)")
    ax.legend(title="split")
    for i, tot in enumerate(bottom):
        if tot > 0:
            ax.text(i, tot, f"{int(tot):,}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    figp = Path(args.fig)
    fig.savefig(figp, dpi=150)
    print(f"[06] wrote {figp}")


if __name__ == "__main__":
    main()
