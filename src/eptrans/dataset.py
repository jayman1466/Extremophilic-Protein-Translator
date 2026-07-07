"""Labeled dataset assembly with leakage-aware splits.

Combines the three upstream products into a protein-level supervised dataset:

    secreted proteins (stage 05)  x  genome environmental class (stage 03/04)

Each secreted protein inherits the extremophile class of its source genome
(or ``mesophile`` for a confident-mesophile outgroup). The resulting table is
the training data for a protein language model / classifier.

Leakage control
---------------
Two independent leakage risks are controlled:

1. **Sequence-similarity leakage** - near-identical secreted proteins recur
   across genomes. If a protein lands in train and its near-duplicate in test,
   the test score is inflated. We therefore assign whole **sequence clusters**
   (mmseqs2, default 50% id / 80% cov) to a single split. If no cluster map is
   supplied we fall back to **genome-level** grouping (all proteins of one
   genome share a split), which still prevents genome memorization but not
   cross-genome homology leakage.

2. **Class imbalance across splits** - splits are drawn per class so the
   train/val/test class proportions match the overall proportions
   (stratified group split).

The phylogenetically-matched extremophile/outgroup pairing from stage 04 is the
*class-balance* control (trait decorrelated from clade); this module adds the
*evaluation* control (no leaked homologs).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MESOPHILE_LABEL = "mesophile"


@dataclass
class SplitResult:
    table: pd.DataFrame     # protein rows with a `split` column
    stats: dict


def assign_labels(
    secreted: pd.DataFrame,
    genome_labels: pd.DataFrame,
    genome_col: str = "genome",
    label_acc_col: str = "accession",
    multi_label: bool = False,
) -> pd.DataFrame:
    """Attach environmental class label to each secreted protein.

    Args:
        secreted: per-protein table from stage 05 (must have ``genome_col`` and a
            protein id column). Genome ids may carry GTDB prefixes.
        genome_labels: combined-labels table (stage 03) with per-genome
            ``final_<class>`` booleans and/or ``confident_mesophile``.
        multi_label: if True, keep the full ``;``-joined class list per genome;
            else use the single ``final_label`` (first class) / mesophile.

    Returns:
        secreted table + ``label`` column (and ``is_mesophile``).
    """
    from .gtdb import bare_accession
    gl = genome_labels.copy()
    gl["_bare"] = gl[label_acc_col].astype(str).map(bare_accession)

    class_cols = [c for c in gl.columns if c.startswith("final_") and gl[c].dtype == bool]
    classes = [c[len("final_"):] for c in class_cols]

    def _label_for_row(row):
        labels = [cls for cls, col in zip(classes, class_cols) if bool(row.get(col))]
        if labels:
            return ";".join(sorted(labels)) if multi_label else sorted(labels)[0]
        if bool(row.get("confident_mesophile", False)):
            return MESOPHILE_LABEL
        return None

    gl["_label"] = gl.apply(_label_for_row, axis=1)
    label_map = dict(zip(gl["_bare"], gl["_label"]))

    out = secreted.copy()
    out["_bare"] = out[genome_col].astype(str).map(bare_accession)
    out["label"] = out["_bare"].map(label_map)
    out["is_mesophile"] = out["label"] == MESOPHILE_LABEL
    out = out.drop(columns=["_bare"])
    # keep only labeled proteins (drop genomes with no class and not mesophile)
    out = out[out["label"].notna()].reset_index(drop=True)
    return out


def stratified_group_split(
    df: pd.DataFrame,
    group_col: str,
    label_col: str = "label",
    splits: dict | None = None,
    seed: int = 1466,
) -> pd.DataFrame:
    """Assign whole groups to train/val/test, stratified by (group-majority) label.

    Every row in a group gets the same split. Groups are stratified by each
    group's majority label so class proportions are preserved across splits.
    """
    splits = splits or {"train": 0.8, "val": 0.1, "test": 0.1}
    order = ["train", "val", "test"]
    fracs = [splits.get(s, 0.0) for s in order]

    # majority label per group
    grp_label = (df.groupby(group_col)[label_col]
                 .agg(lambda s: s.value_counts().index[0]))
    rng = np.random.default_rng(seed)

    group_split: dict = {}
    for lab, groups in grp_label.groupby(grp_label):
        g = list(groups.index)
        rng.shuffle(g)
        n = len(g)
        n_train = int(round(n * fracs[0]))
        n_val = int(round(n * fracs[1]))
        # ensure test gets the remainder
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        assign = (["train"] * n_train + ["val"] * n_val
                  + ["test"] * (n - n_train - n_val))
        for grp, sp in zip(g, assign):
            group_split[grp] = sp

    out = df.copy()
    out["split"] = out[group_col].map(group_split)
    return out


def assemble_dataset(
    secreted: pd.DataFrame,
    genome_labels: pd.DataFrame,
    cluster_map: pd.DataFrame | None = None,
    protein_id_col: str = "protein_id",
    genome_col: str = "genome",
    seq_col: str | None = None,
    splits: dict | None = None,
    seed: int = 1466,
    multi_label: bool = False,
) -> SplitResult:
    """Full assembly: label + leakage-aware split.

    Args:
        cluster_map: optional DataFrame with columns [member, cluster] mapping
            each protein (``{genome}~{protein_id}`` or protein_id) to an mmseqs
            cluster. When absent, groups = genomes.
    """
    labeled = assign_labels(secreted, genome_labels, genome_col=genome_col,
                            multi_label=multi_label)

    # tagged id used for cluster lookup + as the dataset key
    labeled["tagged_id"] = (labeled[genome_col].astype(str) + "~"
                            + labeled[protein_id_col].astype(str))

    if cluster_map is not None and len(cluster_map):
        cmap = dict(zip(cluster_map["member"].astype(str),
                        cluster_map["cluster"].astype(str)))
        # try tagged id first, then bare protein id
        def _grp(row):
            return cmap.get(row["tagged_id"]) or cmap.get(str(row[protein_id_col])) or row["tagged_id"]
        labeled["group"] = labeled.apply(_grp, axis=1)
        group_col = "group"
        group_kind = "sequence_cluster"
    else:
        group_col = genome_col
        group_kind = "genome"

    out = stratified_group_split(labeled, group_col=group_col, label_col="label",
                                 splits=splits, seed=seed)

    stats = {
        "n_proteins": int(len(out)),
        "n_genomes": int(out[genome_col].nunique()),
        "n_groups": int(out[group_col].nunique()),
        "group_kind": group_kind,
        "label_counts": out["label"].value_counts().to_dict(),
        "split_counts": out["split"].value_counts().to_dict(),
        "split_by_label": (out.groupby(["split", "label"]).size()
                           .unstack(fill_value=0).to_dict()),
    }
    # leakage assertion: no group spans multiple splits
    spans = out.groupby(group_col)["split"].nunique()
    stats["max_splits_per_group"] = int(spans.max()) if len(spans) else 0
    return SplitResult(table=out, stats=stats)
