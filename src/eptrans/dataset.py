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


def merge_cluster_maps(maps: dict, members: "pd.Index | list") -> "pd.Series":
    """Union-find merge of several cluster maps into one leakage-safe grouping.

    Returns member -> merged_group_id, where two proteins share a merged group iff
    they are co-clustered in AT LEAST ONE input map (i.e. the transitive closure
    over the union of all co-clustering relations).

    WHY THIS IS REQUIRED rather than just using the coarsest map: clustering at a
    lower identity is NOT guaranteed to be a coarsening of clustering at a higher
    one. Measured on the real 158-genome probe proteome (323,178 proteins,
    --cov-mode 0): of 34,752 multi-member 50%-identity clusters, 1,421 (4.09%) are
    SPLIT across two or more 40%-identity clusters. mmseqs cascaded clustering
    picks different representatives at different thresholds, so the partitions
    cross rather than nest.

    Consequence: if the split were grouped on the 40% map alone, those 1,421
    clusters would have members landing in different folds while still being
    50%-identical to each other -- exactly the train/test leakage the
    cluster-level split exists to prevent. Merging both maps makes the grouping
    at least as coarse as every map it was built from, so no co-clustering
    relation in ANY map can straddle a fold boundary.
    """
    parent = {m: m for m in members}

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:      # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for series in maps.values():
        first_of_cluster = {}
        for mem, cl in series.items():
            if mem not in parent:
                continue
            if cl in first_of_cluster:
                union(first_of_cluster[cl], mem)
            else:
                first_of_cluster[cl] = mem
    return pd.Series({m: find(m) for m in parent}, dtype=object)


def _derive_protein_pairs(
    labeled: pd.DataFrame,
    group_col: str,
    genome_col: str,
    pairs: pd.DataFrame,
    max_pairs_per_cluster_class: int | None = None,
    seed: int = 1466,
    tiebreak: str = "auto",
    scope_by_class: dict | None = None,
    default_scope: str = "secreted",
    secreted_col: str = "is_secreted",
    cluster_col_by_scope: dict | None = None,
) -> "pd.DataFrame":
    """Derive protein-level ortholog pairs = (cluster INTERSECT matched-genome-pair).

    For each matched (extremophile, outgroup) genome pair, find sequence clusters
    that contain a protein from BOTH genomes; those co-clustered proteins are the
    ortholog pair (>=50% id by construction). Within a (cluster, genome-pair) we
    take one protein per genome (highest cs_prob as a stable tie-break — a cheap
    proxy for reciprocal-best; a full RBH would need alignment scores). Both
    members share a cluster, hence the same split, so test-fold pairs feed the
    pairwise margin loss (Section 12, L_pair).

    Args:
        max_pairs_per_cluster_class: if set, emit at most this many pairs per
            (cluster, class) cell — a REDUNDANCY cap for whole-proteome scope.

            The pair set is a sparse (clusters x matched-genome-pairs) matrix. A
            cluster's contribution scales with its prevalence f = fraction of
            genomes carrying it. On the secreted set this self-limits: max
            observed prevalence is 849/7,268 = 11.7% (no universal secreted
            protein exists), largest single-cluster contribution 201 pairs.
            Under whole-proteome scope, core families (Bac120 markers, ribosome,
            tRNA synthetases, RNAP, chaperones) sit at f~1.0 and each intersect
            EVERY matched genome pair — 2,687 pairs from one cluster, 13x the
            current maximum, in a regime with zero representatives in the
            current data.

            Those pairs are REDUNDANT, not uninformative: thermophile EF-Tu vs
            mesophile EF-Tu is a real thermostability contrast, and for the
            temperature classes it is exactly the signal whole-proteome scope
            exists to capture. So the correct instrument caps duplicate votes
            within a family rather than dropping the family (do NOT filter
            clusters on prevalence — that deletes the signal).

            Stratification by class is REQUIRED, not cosmetic. Matched genome
            pairs are severely imbalanced (halophile 1,472; thermophile 560;
            acidophile 341; alkaliphile 292; hyperthermophile 22 = 0.8%). An
            UNSTRATIFIED cap of k drops hyperthermophile entirely with
            probability (1 - 22/2687)^k: 0.85 at k=20, 0.66 at k=50; k>=365
            (14% of all genome pairs, barely a cap) is needed for <5% risk.
            Capping per (cluster, class) makes that risk zero.
        seed: RNG seed for the subsample, so pair derivation stays reproducible.
        tiebreak: which within-cluster representative rule to use.
            'auto' (default) uses cs_prob when it is populated for >=50% of rows
            (i.e. secretome input) and the deterministic id order otherwise;
            'cs_prob' forces the SignalP proxy and raises if the column is absent;
            'deterministic' forces lexicographic id order. See the note in the
            body — cs_prob is meaningless outside the secretome.

    Returns columns: class, cluster, ext_acc, outgroup_acc, ext_id, outgroup_id.
    """
    from .gtdb import bare_accession
    lab = labeled.copy()
    lab["_bare"] = lab[genome_col].astype(str).map(bare_accession)

    # Within-cluster representative choice: one protein per (genome, cluster).
    #
    # cs_prob (SignalP cleavage-site probability) is a cheap proxy for
    # reciprocal-best, but it is ONLY meaningful on the secretome, where every
    # protein carries a signal peptide by construction (r232: non-null for all
    # 1,985,508 rows, median 0.966, IQR 0.838-0.979). Under whole_proteome scope
    # ~89% of proteins are SignalP class OTHER with no cleavage site, so cs_prob
    # is 0/NaN for most of them and sorting on it degenerates to arbitrary order
    # while LOOKING principled.
    #
    # So the tie-break is scope-conditional: use cs_prob only when asked for it
    # (secretome scope), and otherwise fall back to an explicit, documented
    # deterministic order rather than a silently-meaningless numeric sort. A real
    # orthology criterion (reciprocal best hit / bidirectional coverage) is the
    # correct long-term fix for whole_proteome scope and is not implemented here.
    if tiebreak == "auto":
        frac_usable = 0.0
        if "cs_prob" in lab.columns and len(lab):
            frac_usable = float((lab["cs_prob"].fillna(0) > 0).mean())
        use_cs = frac_usable >= 0.5
    elif tiebreak == "cs_prob":
        if "cs_prob" not in lab.columns:
            raise ValueError("tiebreak='cs_prob' but the labeled table has no cs_prob column")
        use_cs = True
    elif tiebreak == "deterministic":
        use_cs = False
    else:
        raise ValueError(f"tiebreak must be 'auto', 'cs_prob' or 'deterministic'; got {tiebreak!r}")

    if use_cs:
        lab = lab.sort_values("cs_prob", ascending=False, kind="mergesort")
    else:
        # Stable, reproducible, and honest about being arbitrary: lexicographic
        # on the tagged id. Does NOT pretend to approximate reciprocal-best.
        lab = lab.sort_values("tagged_id", ascending=True, kind="mergesort")
    # ---- PER-CLASS PROTEIN SCOPE (config dataset.protein_scope) ----
    # Scope is per class and comes from config, NOT from a rule about phenotype
    # families. As committed: halophile/acidophile/alkaliphile are `secreted` (the
    # extracellular interface is where pH and salt adaptation acts), and
    # hyperthermophile/psychrophile are `whole_proteome` (thermostability is a
    # global property of the fold). THERMOPHILE IS ALSO A TEMPERATURE CLASS BUT IS
    # SCOPED `secreted` -- deliberately held at the default on 2026-08-04 pending
    # the large-cluster question, so "temperature => whole proteome" is NOT a valid
    # generalization to encode anywhere. A single
    # `labeled` table therefore cannot serve both: derivation must filter per
    # class, not once up front.
    #
    # This matters because outgroups are REUSED across classes. One mesophile can
    # be the outgroup for a halophile (secreted scope) and a hyperthermophile
    # (whole scope) simultaneously, so eligibility is a property of the
    # (protein, class) pair -- not of the protein. Filtering the input table once
    # would either starve the secreted classes of their scope restriction or
    # cap the temperature classes at the secretome.
    # Scope filtering is OPT-IN: with scope_by_class=None the input table is taken
    # as already being the intended scope (the historical contract -- stage 06 was
    # fed a secreted-only table), so every caller that predates this parameter
    # keeps its exact behaviour. Only an explicit scope map activates filtering.
    if scope_by_class is None:
        scope_by_class, scope_active = {}, False
    else:
        scope_by_class, scope_active = dict(scope_by_class), True

    classes_present = (list(pd.Series(pairs["class"]).dropna().unique())
                       if "class" in pairs else [None])
    scopes_needed = (set(scope_by_class.get(c, default_scope) for c in classes_present)
                     if scope_active else {"_asis"})
    if scope_active and "secreted" in scopes_needed and secreted_col not in lab.columns:
        need = sorted(c for c in classes_present
                      if scope_by_class.get(c, default_scope) == "secreted")
        raise ValueError(
            f"protein scope 'secreted' requested for {need} but the labeled table has no "
            f"{secreted_col!r} column. Pass a whole-proteome table carrying that boolean, "
            f"or set those classes to 'whole_proteome'. Refusing to emit whole-proteome "
            f"pairs for a secreted-scope class.")

    # One representative map per SCOPE, built once and reused across classes.
    #
    # cluster_col_by_scope lets each scope cluster at its OWN identity threshold
    # (measured decision 2026-08-04: whole_proteome classes at 40%, secreted at
    # 50%). Absent it, every scope shares `group_col` and behaviour is unchanged.
    ccol = dict(cluster_col_by_scope or {})
    best_by_scope = {}
    for sc in scopes_needed:
        sub = lab
        if sc == "secreted":
            sub = lab[lab[secreted_col].fillna(False).astype(bool)]
        gc = ccol.get(sc, group_col)
        if gc not in sub.columns:
            raise ValueError(f"cluster column {gc!r} for scope {sc!r} not in the labeled table")
        # PERF: the loop below does 2 partial and 2 scalar lookups per emitted pair.
        # A (_bare, cluster) MultiIndex Series from groupby().first() is NOT lexsorted,
        # so each .loc degrades to a scan. Measured on a 4M-entry index: partial .loc
        # 3.61 ms, scalar .loc 420 us; sort_index() alone gives 8.6x on the scalar path,
        # and a plain dict-of-dicts gives ~9,000x on partials for a ~6 s one-time build.
        # At ~1.2M whole-proteome pairs the .loc form costs ~17 min per million pairs and
        # scales with output size, which is what made run 1164635 take hours.
        _g = sub.groupby(["_bare", gc])["tagged_id"].first()
        _d: dict[str, dict] = {}
        for (_b, _cl), _tid in zip(_g.index, _g.values):
            _d.setdefault(_b, {})[_cl] = _tid
        best_by_scope[sc] = _d

    def _scope_of(cls):
        return scope_by_class.get(cls, default_scope) if scope_active else "_asis"

    # PERF: parallel per-column lists rather than one dict per row (measured 2.24x
    # faster and 4.7x less peak memory at 1.2M rows). Secondary to the lookup fix above.
    c_cls: list = []; c_clu: list = []; c_e: list = []; c_m: list = []
    c_eid: list = []; c_mid: list = []; c_sc: list = []; c_col: list = []
    for cls, e_raw, m_raw in zip(pairs.get("class", [None] * len(pairs)),
                                 pairs["extremophile_acc"], pairs["outgroup_acc"]):
        e, m = bare_accession(e_raw), bare_accession(m_raw)
        if not e or not m:
            continue
        sc = _scope_of(cls)
        best = best_by_scope[sc]
        e_map = best.get(e)
        m_map = best.get(m)
        if not e_map or not m_map:
            continue
        # iterate the smaller map and probe the larger: O(min) instead of building
        # both key sets and intersecting them
        if len(e_map) > len(m_map):
            small, large = m_map, e_map
        else:
            small, large = e_map, m_map
        col = ccol.get(sc, group_col)
        for cl in small:
            if cl in large:
                c_cls.append(cls); c_clu.append(cl); c_e.append(e); c_m.append(m)
                c_eid.append(e_map[cl]); c_mid.append(m_map[cl])
                c_sc.append(sc); c_col.append(col)
    out = pd.DataFrame({"class": c_cls, "cluster": c_clu, "ext_acc": c_e,
                        "outgroup_acc": c_m, "ext_id": c_eid, "outgroup_id": c_mid,
                        "scope": c_sc, "cluster_col": c_col},
                       columns=["class", "cluster", "ext_acc", "outgroup_acc",
                                "ext_id", "outgroup_id", "scope", "cluster_col"])

    k = max_pairs_per_cluster_class
    if k is not None and len(out):
        # Subsample within each (cluster, class) cell. Deterministic given seed.
        rng = np.random.default_rng(seed)
        keep = []
        for _, idx in out.groupby(["cluster", "class"], dropna=False).indices.items():
            if len(idx) <= k:
                keep.append(idx)
            else:
                keep.append(rng.choice(idx, size=k, replace=False))
        out = out.iloc[np.sort(np.concatenate(keep))].reset_index(drop=True)
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
    pairs: pd.DataFrame | None = None,
    max_pairs_per_cluster_class: int | None = None,
    tiebreak: str = "auto",
    scope_by_class: dict | None = None,
    default_scope: str = "secreted",
    secreted_col: str = "is_secreted",
    cluster_maps: dict | None = None,
    cluster_col_by_scope: dict | None = None,
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

    # ---- MULTI-THRESHOLD CLUSTERING ----
    # cluster_maps = {name: DataFrame[member, cluster]} attaches one column per
    # named map (e.g. 'id50', 'id40'), so pair derivation can use a different
    # threshold per scope. The SPLIT is grouped on the union-find merge of every
    # map (see merge_cluster_maps): the partitions cross rather than nest, so
    # grouping on any single map would leak.
    if cluster_maps:
        series = {}
        for name, cm in cluster_maps.items():
            if cm is None or not len(cm):
                continue
            d = dict(zip(cm["member"].astype(str), cm["cluster"].astype(str)))
            col = f"cluster_{name}"
            labeled[col] = [
                d.get(t) or d.get(str(p)) or t
                for t, p in zip(labeled["tagged_id"], labeled[protein_id_col])
            ]
            series[name] = pd.Series(
                dict(zip(labeled["tagged_id"], labeled[col])), dtype=object)
        if not series:
            raise ValueError("cluster_maps was given but every map was empty")
        merged = merge_cluster_maps(series, list(labeled["tagged_id"]))
        labeled["group"] = labeled["tagged_id"].map(merged)
        group_col = "group"
        group_kind = "sequence_cluster_merged(" + "+".join(sorted(series)) + ")"
    elif cluster_map is not None and len(cluster_map):
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
            protein_pairs = _derive_protein_pairs(
                labeled, group_col, genome_col, pairs,
                max_pairs_per_cluster_class=max_pairs_per_cluster_class, seed=seed,
                tiebreak=tiebreak, scope_by_class=scope_by_class,
                default_scope=default_scope, secreted_col=secreted_col,
                cluster_col_by_scope=cluster_col_by_scope)

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
