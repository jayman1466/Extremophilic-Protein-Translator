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
    protein_pairs: pd.DataFrame | None = None  # ortholog pairs for L_pair (cluster regime)


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
    # Carry the genome's environmental-label confidence tier onto each protein so
    # downstream training can weight examples (high > medium). For mesophiles this
    # is the confident_mesophile tier; for extremophiles the final_confidence tier.
    if "final_confidence" in gl.columns:
        conf_map = dict(zip(gl["_bare"], gl["final_confidence"]))
        out["label_confidence"] = out["_bare"].map(conf_map)
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


def _coassign_matched_pairs(
    labeled: pd.DataFrame,
    group_col: str,
    genome_col: str,
    pairs: pd.DataFrame,
) -> pd.Series:
    """Union base groups so each extremophile and its matched mesophile outgroup
    share a split group.

    Uses union-find over the base groups: for every (extremophile, outgroup)
    pairing, all base groups containing proteins of either genome are merged into
    one component. Whole components are then assigned to a single split, so a
    matched pair can never straddle train/val/test. Reused outgroups form a
    "star" (one mesophile + the extremophiles it anchors) that lands together.

    Returns a Series (indexed like ``labeled``) of the merged group id.
    """
    from .gtdb import bare_accession
    # genome -> set of base groups its proteins occupy
    genome_groups: dict[str, set] = {}
    bare = labeled[genome_col].astype(str).map(bare_accession)
    for g, grp in zip(bare, labeled[group_col].astype(str)):
        genome_groups.setdefault(g, set()).add(grp)

    parent: dict = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        parent[find(a)] = find(b)

    for e_raw, m_raw in zip(pairs["extremophile_acc"], pairs["outgroup_acc"]):
        e, m = bare_accession(e_raw), bare_accession(m_raw)
        # skip unmatched extremophiles (empty outgroup) — nothing to co-assign
        if not e or not m:
            continue
        grps = list(genome_groups.get(e, set()) | genome_groups.get(m, set()))
        for k in range(1, len(grps)):
            union(grps[0], grps[k])

    return labeled[group_col].astype(str).map(lambda g: find(g))


def _derive_protein_pairs(
    labeled: pd.DataFrame,
    group_col: str,
    genome_col: str,
    pairs: pd.DataFrame,
) -> "pd.DataFrame":
    """Derive protein-level ortholog pairs = (cluster INTERSECT matched-genome-pair).

    For each matched (extremophile, outgroup) genome pair, find sequence clusters
    that contain a protein from BOTH genomes; those co-clustered proteins are the
    ortholog pair (>=50% id by construction). Within a (cluster, genome-pair) we
    take one protein per genome (highest cs_prob as a stable tie-break — a cheap
    proxy for reciprocal-best; a full RBH would need alignment scores). Both
    members share a cluster, hence the same split, so test-fold pairs feed the
    pairwise margin loss (Section 12, L_pair).

    Returns columns: class, cluster, ext_acc, outgroup_acc, ext_id, outgroup_id.
    """
    from .gtdb import bare_accession
    lab = labeled.copy()
    lab["_bare"] = lab[genome_col].astype(str).map(bare_accession)
    # index: (bare genome, cluster) -> best tagged_id
    sort_col = "cs_prob" if "cs_prob" in lab.columns else group_col
    lab = lab.sort_values(sort_col, ascending=False)
    best = (lab.groupby(["_bare", group_col])["tagged_id"].first())

    rows = []
    for cls, e_raw, m_raw in zip(pairs.get("class", [None] * len(pairs)),
                                 pairs["extremophile_acc"], pairs["outgroup_acc"]):
        e, m = bare_accession(e_raw), bare_accession(m_raw)
        if not e or not m:
            continue
        try:
            e_clusters = set(best.loc[e].index)
            m_clusters = set(best.loc[m].index)
        except KeyError:
            continue
        for cl in (e_clusters & m_clusters):
            rows.append({"class": cls, "cluster": cl, "ext_acc": e, "outgroup_acc": m,
                         "ext_id": best.loc[(e, cl)], "outgroup_id": best.loc[(m, cl)]})
    return pd.DataFrame(rows, columns=["class", "cluster", "ext_acc", "outgroup_acc",
                                       "ext_id", "outgroup_id"])


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
    pairs: pd.DataFrame | None = None,
) -> SplitResult:
    """Full assembly: label + leakage-aware split.

    Args:
        cluster_map: optional DataFrame with columns [member, cluster] mapping
            each protein (``{genome}~{protein_id}`` or protein_id) to an mmseqs
            cluster. When absent, groups = genomes.
        pairs: optional stage-04 pairs table (cols extremophile_acc,
            outgroup_acc). When supplied, matched extremophile/outgroup genomes
            are co-assigned to the same split (union-find over base groups) so a
            pair never straddles train/val/test — preserving the matched contrast
            within each fold.
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

    # Regime switch (see design doc Section 14):
    #  - genome grouping (no cluster map): co-assign matched pairs by UNIONing
    #    genome groups (bounded star components) so a pair shares a split.
    #  - cluster grouping (cluster map present): DO NOT union genomes (would
    #    blow up via conserved-family transitive closure). Split on clusters
    #    directly; matched orthologs co-cluster and land together for free.
    #    Protein-level pairs are derived separately (protein_pairs attribute).
    protein_pairs = None
    if pairs is not None and len(pairs) and group_kind.startswith("genome"):
        labeled["_split_group"] = _coassign_matched_pairs(
            labeled, group_col, genome_col, pairs)
        split_group_col = "_split_group"
        group_kind += "+matched_pairs"
    else:
        split_group_col = group_col
        if pairs is not None and len(pairs) and group_kind.startswith("sequence_cluster"):
            protein_pairs = _derive_protein_pairs(labeled, group_col, genome_col, pairs)

    out = stratified_group_split(labeled, group_col=split_group_col, label_col="label",
                                 splits=splits, seed=seed)

    stats = {
        "n_proteins": int(len(out)),
        "n_genomes": int(out[genome_col].nunique()),
        "n_groups": int(out[split_group_col].nunique()),
        "group_kind": group_kind,
        "label_counts": out["label"].value_counts().to_dict(),
        "split_counts": out["split"].value_counts().to_dict(),
        "split_by_label": (out.groupby(["split", "label"]).size()
                           .unstack(fill_value=0).to_dict()),
    }
    # leakage assertion: no split-group spans multiple splits
    spans = out.groupby(split_group_col)["split"].nunique()
    stats["max_splits_per_group"] = int(spans.max()) if len(spans) else 0
    # largest merged component (diagnostic: pair co-assignment can grow groups)
    comp_sizes = out.groupby(split_group_col)[genome_col].nunique()
    stats["max_genomes_per_group"] = int(comp_sizes.max()) if len(comp_sizes) else 0
    # protein-level ortholog pairs (cluster regime): annotate split + count
    if protein_pairs is not None and len(protein_pairs):
        split_of = dict(zip(out["tagged_id"], out["split"]))
        protein_pairs = protein_pairs.copy()
        protein_pairs["ext_split"] = protein_pairs["ext_id"].map(split_of)
        protein_pairs["out_split"] = protein_pairs["outgroup_id"].map(split_of)
        stats["n_protein_pairs"] = int(len(protein_pairs))
        stats["protein_pairs_same_split"] = int(
            (protein_pairs["ext_split"] == protein_pairs["out_split"]).sum())
    return SplitResult(table=out, stats=stats, protein_pairs=protein_pairs)
