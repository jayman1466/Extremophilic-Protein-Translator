"""Unit tests for eptrans.dataset (labeling + leakage-aware splits)."""
import pandas as pd

from eptrans.dataset import (
    assign_labels, stratified_group_split, assemble_dataset, MESOPHILE_LABEL,
)


def _genome_labels():
    df = pd.DataFrame({
        "accession": ["RS_GCF_000001.1", "GB_GCA_000002.1", "RS_GCF_000003.1"],
        "final_thermophile": [True, False, False],
        "final_halophile": [False, False, False],
        "confident_mesophile": [False, True, False],
        "final_confidence": ["high", "none", "none"],
    })
    return df


def _secreted():
    # genome 3 has no class and is not mesophile -> its proteins dropped
    return pd.DataFrame({
        "genome": ["RS_GCF_000001.1", "RS_GCF_000001.1", "GB_GCA_000002.1", "RS_GCF_000003.1"],
        "protein_id": ["c1_1", "c1_2", "c2_1", "c3_1"],
        "signalp_class": ["SP", "LIPO", "TAT", "SP"],
    })


def test_assign_labels():
    out = assign_labels(_secreted(), _genome_labels())
    lab = dict(zip(out["protein_id"], out["label"]))
    assert lab["c1_1"] == "thermophile"
    assert lab["c1_2"] == "thermophile"
    assert lab["c2_1"] == MESOPHILE_LABEL
    assert "c3_1" not in lab  # unlabeled genome dropped
    assert out["is_mesophile"].sum() == 1


def test_label_confidence_propagates():
    out = assign_labels(_secreted(), _genome_labels())
    conf = dict(zip(out["protein_id"], out["label_confidence"]))
    # thermophile genome was high-confidence -> both its proteins inherit "high"
    assert conf["c1_1"] == "high"
    assert conf["c1_2"] == "high"


def test_no_group_spans_split():
    # many genomes, ensure no genome's proteins land in >1 split
    rows = []
    for g in range(30):
        cls = "thermophile" if g % 2 == 0 else "halophile"
        for p in range(5):
            rows.append({"genome": f"G{g}", "protein_id": f"p{p}", "label": cls})
    df = pd.DataFrame(rows)
    out = stratified_group_split(df, group_col="genome", seed=1)
    spans = out.groupby("genome")["split"].nunique()
    assert spans.max() == 1  # no leakage


def test_stratification_all_labels_in_train():
    rows = []
    for g in range(40):
        cls = ["thermophile", "halophile", MESOPHILE_LABEL][g % 3]
        for p in range(3):
            rows.append({"genome": f"G{g}", "protein_id": f"p{p}", "label": cls})
    df = pd.DataFrame(rows)
    out = stratified_group_split(df, group_col="genome", seed=7)
    train_labels = set(out[out["split"] == "train"]["label"])
    assert {"thermophile", "halophile", MESOPHILE_LABEL} <= train_labels


def test_assemble_with_cluster_map_prevents_homolog_leak():
    # two genomes share a homologous protein cluster -> must not span splits
    rows = []
    for g in range(20):
        for p in range(4):
            rows.append({"genome": f"G{g}", "protein_id": f"p{p}"})
    secreted = pd.DataFrame(rows)
    gl = pd.DataFrame({
        "accession": [f"G{g}" for g in range(20)],
        "final_thermophile": [g % 2 == 0 for g in range(20)],
        "confident_mesophile": [g % 2 == 1 for g in range(20)],
    })
    # cluster map: put p0 of every genome into ONE shared cluster
    members, clusters = [], []
    for g in range(20):
        for p in range(4):
            members.append(f"G{g}~p{p}")
            clusters.append("SHARED" if p == 0 else f"G{g}~p{p}")
    cmap = pd.DataFrame({"member": members, "cluster": clusters})
    res = assemble_dataset(secreted, gl, cluster_map=cmap, seed=3)
    assert res.stats["group_kind"] == "sequence_cluster"
    assert res.stats["max_splits_per_group"] == 1
    # the SHARED cluster's rows are all in one split
    shared = res.table[res.table["group"] == "SHARED"]
    assert shared["split"].nunique() == 1


def test_assemble_genome_fallback():
    res = assemble_dataset(_secreted(), _genome_labels(), cluster_map=None, seed=1)
    assert res.stats["group_kind"] == "genome"
    assert res.stats["max_splits_per_group"] == 1
    assert res.stats["n_proteins"] == 3  # c3_1 dropped


def test_matched_pairs_coassigned_to_same_split():
    """Each extremophile and its matched mesophile outgroup must share a split."""
    import pandas as pd
    # 10 extremophile genomes (even ids, thermophile) each paired to a distinct
    # mesophile outgroup (odd ids). Without co-assignment the independent genomes
    # could split apart; with pairs they must not.
    rows = []
    for g in range(20):
        for p in range(3):
            rows.append({"genome": f"G{g:03d}.1", "protein_id": f"p{p}"})
    secreted = pd.DataFrame(rows)
    genome_labels = pd.DataFrame({
        "accession": [f"G{g:03d}.1" for g in range(20)],
        "final_thermophile": [g % 2 == 0 for g in range(20)],
        "confident_mesophile": [g % 2 == 1 for g in range(20)],
        "final_confidence": ["high" if g % 2 == 0 else "none" for g in range(20)],
    })
    pairs = pd.DataFrame({
        "extremophile_acc": [f"G{g:03d}.1" for g in range(0, 20, 2)],
        "outgroup_acc":     [f"G{g:03d}.1" for g in range(1, 20, 2)],
    })
    res = assemble_dataset(secreted, genome_labels, pairs=pairs,
                           genome_col="genome", seed=3)
    tab = res.table
    # for every pair, extremophile and outgroup share the same split
    split_of = dict(zip(tab["genome"], tab["split"]))
    for e, m in zip(pairs["extremophile_acc"], pairs["outgroup_acc"]):
        assert split_of[e] == split_of[m], f"pair {e}/{m} split apart"
    assert res.stats["max_splits_per_group"] == 1
    assert "matched_pairs" in res.stats["group_kind"]


def test_cluster_regime_derives_protein_pairs_no_genome_union():
    """With a cluster map, split on clusters (no genome union) and derive
    protein-level ortholog pairs from cluster INTERSECT matched-genome-pair."""
    import pandas as pd
    # 2 matched pairs; each pair shares ONE orthologous cluster (co-clustered),
    # plus genome-private singleton proteins in their own clusters.
    rows, cmap = [], []
    for gi, g in enumerate(["E0.1", "M0.1", "E1.1", "M1.1"]):
        for p in range(2):
            pid = f"c{p}"
            rows.append({"genome": g, "protein_id": pid, "cs_prob": 0.9})
            tagged = f"{g}~{pid}"
            # p0 of a pair shares a cluster; p1 is private
            pair_idx = gi // 2
            clu = f"ortho_{pair_idx}" if p == 0 else f"priv_{g}_{p}"
            cmap.append({"cluster": clu, "member": tagged})
    secreted = pd.DataFrame(rows)
    cluster_map = pd.DataFrame(cmap)
    genome_labels = pd.DataFrame({
        "accession": ["E0.1", "M0.1", "E1.1", "M1.1"],
        "final_thermophile": [True, False, True, False],
        "confident_mesophile": [False, True, False, True],
        "final_confidence": ["high", "none", "high", "none"],
    })
    pairs = pd.DataFrame({
        "class": ["thermophile", "thermophile"],
        "extremophile_acc": ["E0.1", "E1.1"],
        "outgroup_acc": ["M0.1", "M1.1"],
    })
    res = assemble_dataset(secreted, genome_labels, cluster_map=cluster_map,
                           pairs=pairs, genome_col="genome", seed=5)
    # cluster regime: group_kind must NOT carry the genome-union tag
    assert res.stats["group_kind"] == "sequence_cluster"
    assert "matched_pairs" not in res.stats["group_kind"]
    # 2 ortholog pairs derived (one shared cluster per matched pair)
    assert res.protein_pairs is not None
    assert res.stats["n_protein_pairs"] == 2
    # each derived pair co-clusters -> same split guaranteed
    assert res.stats["protein_pairs_same_split"] == 2
