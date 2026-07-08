"""Unit tests for eptrans.selection (diversity cap + matched outgroups)."""
import pandas as pd
import pytest

from eptrans.selection import select_extremophiles, find_outgroup, select_with_outgroups
from eptrans.gtdb import GTDB_RANKS


def _make_labels():
    """Synthetic labels: two families of thermophiles + matched mesophiles."""
    rows = []
    # 10 thermophiles in family FamA (should be capped), 3 in FamB
    for i in range(10):
        rows.append(dict(accession=f"E_A{i}", family="FamA", genus=f"GenA{i%2}",
                         order="OrdA", **{"class": "ClsA"}, phylum="PhyA", domain="Bacteria",
                         final_thermophile=True, final_confidence="high", confident_mesophile=False))
    for i in range(3):
        rows.append(dict(accession=f"E_B{i}", family="FamB", genus=f"GenB{i}",
                         order="OrdB", **{"class": "ClsB"}, phylum="PhyB", domain="Bacteria",
                         final_thermophile=True, final_confidence="medium", confident_mesophile=False))
    # mesophiles: one in GenA0 (genus match), one in FamB (family match), one far
    rows.append(dict(accession="M_genusA", family="FamA", genus="GenA0", order="OrdA",
                     **{"class": "ClsA"}, phylum="PhyA", domain="Bacteria",
                     final_thermophile=False, final_confidence="none", confident_mesophile=True))
    rows.append(dict(accession="M_famB", family="FamB", genus="GenBx", order="OrdB",
                     **{"class": "ClsB"}, phylum="PhyB", domain="Bacteria",
                     final_thermophile=False, final_confidence="none", confident_mesophile=True))
    df = pd.DataFrame(rows)
    # species rank (required by select_with_outgroups' GTDB_RANKS check)
    df["species"] = df["accession"] + "_sp"
    # add remaining rank cols not set as False class flags
    for cls in ["hyperthermophile", "psychrophile", "acidophile", "alkaliphile", "halophile"]:
        df[f"final_{cls}"] = False
    return df


def test_diversity_cap():
    df = _make_labels()
    sel = select_extremophiles(df, "thermophile", max_per_lineage=3, lineage_rank="family")
    # FamA capped at 3 (of 10), FamB all 3 -> 6 total
    assert (sel["family"] == "FamA").sum() == 3
    assert (sel["family"] == "FamB").sum() == 3
    assert len(sel) == 6


def test_max_total():
    df = _make_labels()
    sel = select_extremophiles(df, "thermophile", max_per_lineage=100,
                               lineage_rank="family", max_total=5)
    assert len(sel) == 5


def test_confidence_filter_excludes_tiers():
    df = _make_labels()
    # high-only must drop the medium FamB members entirely (not just deprioritise)
    sel = select_extremophiles(df, "thermophile", max_per_lineage=100,
                               lineage_rank="family", confidence_levels=("high",))
    assert set(sel["final_confidence"]) == {"high"}
    assert (sel["family"] == "FamB").sum() == 0   # FamB was medium


def test_per_class_confidence_dict():
    df = _make_labels()
    # dict form: thermophile high-only -> FamB (medium) excluded
    res = select_with_outgroups(df, classes=["thermophile"], max_per_lineage=100,
                                lineage_rank="family",
                                confidence_levels={"thermophile": ("high",)})
    assert set(res.extremophiles["final_confidence"]) == {"high"}


def test_outgroup_reuse_across_classes():
    df = _make_labels()
    # make the same mesophile a valid genus match for two classes
    df.loc[df["accession"] == "E_A0", "final_acidophile"] = True
    df.loc[df["accession"] == "E_A0", "final_confidence"] = "high"
    # with reuse, the shared-genus mesophile M_genusA can pair in both classes
    res = select_with_outgroups(df, classes=["thermophile", "acidophile"],
                                max_per_lineage=100, lineage_rank="family",
                                reuse_outgroups=True)
    pairs = res.pairs
    reused = pairs[pairs["outgroup_acc"] == "M_genusA"]
    assert reused["class"].nunique() >= 1  # M_genusA can appear for >=1 class
    # outgroup set is deduplicated even if reused across classes
    assert res.outgroups["accession"].is_unique


def test_confidence_preference():
    df = _make_labels()
    # high-confidence FamA should be picked before medium FamB when capped tight
    sel = select_extremophiles(df, "thermophile", max_per_lineage=100,
                               lineage_rank="domain", max_total=10)
    # first picks should be high confidence
    assert sel.iloc[0]["final_confidence"] == "high"


def test_find_outgroup_genus_first():
    df = _make_labels()
    meso = df[df["confident_mesophile"]].copy()
    erow = df[df["accession"] == "E_A0"].iloc[0]  # GenA0
    idx, rank = find_outgroup(erow, meso, set())
    assert rank == "genus"
    assert meso.loc[idx, "accession"] == "M_genusA"


def test_find_outgroup_walks_up():
    df = _make_labels()
    meso = df[df["confident_mesophile"]].copy()
    erow = df[df["accession"] == "E_B0"].iloc[0]  # FamB, GenB0 (no genus mesophile)
    idx, rank = find_outgroup(erow, meso, set())
    assert rank == "family"
    assert meso.loc[idx, "accession"] == "M_famB"


def test_outgroup_not_reused():
    df = _make_labels()
    meso = df[df["confident_mesophile"]].copy()
    used = set()
    e0 = df[df["accession"] == "E_A0"].iloc[0]
    idx0, _ = find_outgroup(e0, meso, used)
    used.add(idx0)
    # another GenA0 extremophile can't reuse the same outgroup -> must walk up or fail
    e2 = df[df["accession"] == "E_A2"].iloc[0]  # GenA0 too (i%2)
    idx2, rank2 = find_outgroup(e2, meso, used)
    assert idx2 != idx0  # different (or None)


def test_select_with_outgroups_end_to_end():
    df = _make_labels()
    res = select_with_outgroups(df, classes=["thermophile"], max_per_lineage=3,
                                lineage_rank="family", max_total_per_class=None)
    assert res.stats["n_extremophiles"] == 6
    assert res.stats["n_pairs_matched"] >= 1
    assert "matched_rank" in res.pairs.columns


def test_missing_taxonomy_raises():
    df = pd.DataFrame({"accession": ["x"], "final_thermophile": [True],
                       "confident_mesophile": [False]})
    with pytest.raises(ValueError):
        select_with_outgroups(df, classes=["thermophile"])
