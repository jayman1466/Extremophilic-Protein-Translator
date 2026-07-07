"""Unit tests for eptrans.reconcile (cross-release accession matching)."""
import pandas as pd

from eptrans.reconcile import _keys, reconcile, attach_genome_paths


def test_keys_strip_prefix_and_levels():
    k = _keys("RS_GCF_000005845.2")
    assert k["exact"] == "GCF_000005845.2"
    assert k["noversion"] == "GCF_000005845"
    assert k["assembly"] == "000005845"


def test_keys_gca_gcf_share_assembly():
    # GenBank and RefSeq copies of one assembly share the numeric id
    assert _keys("GB_GCA_000005845.1")["assembly"] == _keys("RS_GCF_000005845.2")["assembly"]


def _reps():
    return pd.DataFrame({
        "accession": [
            "RS_GCF_000005845.2",   # exact match
            "GB_GCA_000008085.1",   # version-bump match
            "RS_GCF_000009999.1",   # GCA<->GCF swap match
            "GB_GCA_111111111.1",   # no match -> delta
        ],
        "domain": ["Bacteria", "Archaea", "Bacteria", "Bacteria"],
    })


def _precomp():
    return pd.DataFrame({
        "accession": [
            "GCF_000005845.2",      # exact
            "GCA_000008085.3",      # bumped version of rep 2
            "GCA_000009999.1",      # GCF rep 3 is GCA here (swap)
            "GCA_222222222.1",      # not a rep in r232 -> dropped
        ],
        "temperature_optimum": [37.0, 85.0, 4.0, 30.0],
        "oxygen": ["tolerant", "not tolerant", "tolerant", "tolerant"],
    })


def test_reconcile_levels():
    res = reconcile(_reps(), _precomp())
    lv = res.reconciled.set_index("accession")["genomespot_match_level"].to_dict()
    assert lv["RS_GCF_000005845.2"] == "exact"
    assert lv["GB_GCA_000008085.1"] == "noversion"
    assert lv["RS_GCF_000009999.1"] == "assembly"
    assert pd.isna(lv["GB_GCA_111111111.1"])


def test_reconcile_counts():
    res = reconcile(_reps(), _precomp())
    assert res.stats["n_reused"] == 3
    assert res.stats["n_delta"] == 1
    assert res.stats["n_dropped_precomputed"] == 1
    assert res.stats["reuse_by_level"] == {"exact": 1, "noversion": 1, "assembly": 1}


def test_reconcile_values_carried():
    res = reconcile(_reps(), _precomp())
    r = res.reconciled.set_index("accession")
    assert r.loc["RS_GCF_000005845.2", "precomp_temperature_optimum"] == 37.0
    assert r.loc["GB_GCA_000008085.1", "precomp_temperature_optimum"] == 85.0
    # delta row has no carried value
    assert pd.isna(r.loc["GB_GCA_111111111.1", "precomp_temperature_optimum"])


def test_delta_frame_is_unmatched_only():
    res = reconcile(_reps(), _precomp())
    assert list(res.delta["accession"]) == ["GB_GCA_111111111.1"]


def test_dropped_precomputed():
    res = reconcile(_reps(), _precomp())
    assert list(res.dropped_precomputed["accession"]) == ["GCA_222222222.1"]


def test_attach_genome_paths(tmp_path):
    idx = tmp_path / "genome_index.tsv"
    idx.write_text(
        "GCF_000005845.2\t/abs/GCF_000005845.2_genomic.fna.gz\n"
        "GCA_000008085.1\t/abs/GCA_000008085.1_genomic.fna.gz\n"
    )
    df = pd.DataFrame({"accession": ["RS_GCF_000005845.2", "GB_GCA_111111111.1"]})
    out = attach_genome_paths(df, str(idx))
    assert out.loc[0, "genome_fna_path"] == "/abs/GCF_000005845.2_genomic.fna.gz"
    assert pd.isna(out.loc[1, "genome_fna_path"])
