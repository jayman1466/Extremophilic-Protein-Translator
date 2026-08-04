"""Multi-threshold clustering: per-scope thresholds + leakage-safe merged split."""
import pandas as pd, pytest
from eptrans.dataset import merge_cluster_maps, _derive_protein_pairs, assemble_dataset


def test_merge_is_transitive_closure_over_all_maps():
    # map A co-clusters (p1,p2); map B co-clusters (p2,p3). No single map joins
    # p1 and p3, but the merge must -- otherwise p1 and p3 could land in
    # different folds while being linked through p2.
    A = pd.Series({"p1": "a1", "p2": "a1", "p3": "a2"})
    B = pd.Series({"p1": "b1", "p2": "b2", "p3": "b2"})
    g = merge_cluster_maps({"A": A, "B": B}, ["p1", "p2", "p3"])
    assert g["p1"] == g["p2"] == g["p3"]


def test_merge_keeps_unrelated_proteins_apart():
    A = pd.Series({"p1": "a1", "p2": "a1", "q1": "a9"})
    B = pd.Series({"p1": "b1", "p2": "b1", "q1": "b9"})
    g = merge_cluster_maps({"A": A, "B": B}, ["p1", "p2", "q1"])
    assert g["p1"] == g["p2"]
    assert g["q1"] != g["p1"]


def test_merge_handles_the_non_nesting_case():
    """The measured hazard: one 50% cluster split across two 40% clusters.
    The merge must keep all four proteins together."""
    id50 = pd.Series({"w": "c1", "x": "c1", "y": "c1", "z": "c1"})
    id40 = pd.Series({"w": "d1", "x": "d1", "y": "d2", "z": "d2"})
    g = merge_cluster_maps({"id50": id50, "id40": id40}, list("wxyz"))
    assert len({g[k] for k in "wxyz"}) == 1


def test_merge_is_at_least_as_coarse_as_every_input():
    id50 = pd.Series({"w": "c1", "x": "c1", "y": "c2", "z": "c2"})
    id40 = pd.Series({"w": "d1", "x": "d2", "y": "d1", "z": "d2"})
    g = merge_cluster_maps({"a": id50, "b": id40}, list("wxyz"))
    for m in (id50, id40):
        for cl in set(m):
            mems = [k for k, v in m.items() if v == cl]
            assert len({g[k] for k in mems}) == 1, (cl, mems)


def _lab():
    rows = []
    for gen in ("E_HOT", "E_SALT", "M_OUT"):
        rows.append(dict(genome=gen, tagged_id=f"{gen}~s", cluster_id50="s50",
                         cluster_id40="s40", is_secreted=True, cs_prob=0.9))
        # a cytoplasmic protein that only co-clusters at 40%
        rows.append(dict(genome=gen, tagged_id=f"{gen}~c", is_secreted=False,
                         cs_prob=float("nan"),
                         cluster_id50=f"uniq_{gen}",   # singleton at 50%
                         cluster_id40="c40"))          # co-clusters at 40%
    return pd.DataFrame(rows)

PAIRS = pd.DataFrame([
    {"class": "hyperthermophile", "extremophile_acc": "E_HOT", "outgroup_acc": "M_OUT"},
    {"class": "halophile", "extremophile_acc": "E_SALT", "outgroup_acc": "M_OUT"},
])
SCOPE = {"hyperthermophile": "whole_proteome", "halophile": "secreted"}
CCOL = {"whole_proteome": "cluster_id40", "secreted": "cluster_id50"}


def test_each_scope_uses_its_own_threshold():
    out = _derive_protein_pairs(_lab(), "cluster_id50", "genome", PAIRS,
                                tiebreak="deterministic", scope_by_class=SCOPE,
                                cluster_col_by_scope=CCOL)
    hyp = out[out["class"] == "hyperthermophile"]
    hal = out[out["class"] == "halophile"]
    # whole_proteome at 40% recovers the cytoplasmic pair that is a singleton at 50%
    assert "c40" in set(hyp.cluster)
    # secreted class stays on the 50% map and only gets the secreted cluster
    assert set(hal.cluster) == {"s50"}


def test_cluster_col_recorded_per_row():
    out = _derive_protein_pairs(_lab(), "cluster_id50", "genome", PAIRS,
                                tiebreak="deterministic", scope_by_class=SCOPE,
                                cluster_col_by_scope=CCOL)
    m = dict(zip(out["class"], out.cluster_col))
    assert m["hyperthermophile"] == "cluster_id40"
    assert m["halophile"] == "cluster_id50"


def test_unknown_cluster_column_raises():
    with pytest.raises(ValueError, match="cluster column"):
        _derive_protein_pairs(_lab(), "cluster_id50", "genome", PAIRS,
                              tiebreak="deterministic", scope_by_class=SCOPE,
                              cluster_col_by_scope={"whole_proteome": "nope",
                                                    "secreted": "cluster_id50"})


def test_assemble_groups_split_on_merged_map():
    lab = _lab().rename(columns={"tagged_id": "protein_id"})
    lab["protein_id"] = [p.split("~")[1] for p in lab["protein_id"]]
    # assign_labels reads final_<class> BOOLEAN columns, not a joined string
    labels = pd.DataFrame({
        "accession": ["E_HOT", "E_SALT", "M_OUT"],
        "final_hyperthermophile": [True, False, False],
        "final_halophile": [False, True, False],
        "confident_mesophile": [False, False, True]})
    cm50 = pd.DataFrame({"member": [f"{g}~{p}" for g, p in zip(lab.genome, lab.protein_id)],
                         "cluster": lab.cluster_id50})
    cm40 = pd.DataFrame({"member": [f"{g}~{p}" for g, p in zip(lab.genome, lab.protein_id)],
                         "cluster": lab.cluster_id40})
    res = assemble_dataset(lab, labels, cluster_maps={"id50": cm50, "id40": cm40},
                           pairs=PAIRS, multi_label=True)
    assert res.stats["group_kind"].startswith("sequence_cluster_merged")
    # the 3 cytoplasmic proteins are singletons at 50% but one cluster at 40%,
    # so the merged grouping must put them in ONE group -> one split
    grp = dict(zip(res.table.tagged_id, res.table.group))
    assert len({grp[f"{g}~c"] for g in ("E_HOT", "E_SALT", "M_OUT")}) == 1
