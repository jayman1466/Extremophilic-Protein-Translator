"""Tests for stage 03b: pooled measured-OGT merge and the psychrophile rubric."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "03b_merge_measured_ogt.py"
_spec = importlib.util.spec_from_file_location("merge_ogt", _SCRIPT)
merge_ogt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(merge_ogt)


def _row(**kw):
    base = dict(ogt_measured=np.nan, ogt_n_sources=np.nan,
                tmin_measured=np.nan, meta_cold=False)
    base.update(kw)
    return pd.Series(base)


class TestNormaliseSpecies:
    def test_strips_strain_suffix_to_binomial(self):
        assert merge_ogt.normalise_species("Acetobacter aceti NBRC 14818") == "acetobacter aceti"

    def test_drops_candidatus_prefix(self):
        assert merge_ogt.normalise_species("Candidatus Pelagibacter ubique") == "pelagibacter ubique"

    def test_returns_none_for_unusable(self):
        assert merge_ogt.normalise_species("Pseudomonas") is None
        assert merge_ogt.normalise_species(None) is None
        assert merge_ogt.normalise_species(np.nan) is None

    def test_extracts_species_from_gtdb_taxonomy(self):
        tax = "d__Bacteria;p__Pseudomonadota;c__Gammaproteobacteria;s__Escherichia coli"
        assert merge_ogt.species_from_gtdb_taxonomy(tax) == "escherichia coli"

    def test_placeholder_taxonomy_without_species_is_none(self):
        assert merge_ogt.species_from_gtdb_taxonomy("d__Bacteria;p__X;s__") is None


class TestConventionalTemps:
    def test_excludes_only_the_four_incubator_defaults(self):
        df = pd.DataFrame({"t": [25.0, 28.0, 30.0, 37.0, 26.0, 29.5, 15.0]})
        kept = merge_ogt.drop_conventional(df, "t").t.tolist()
        assert kept == [26.0, 29.5, 15.0]


class TestRubric:
    """The rubric must never consult a GenomeSPOT prediction."""

    def test_strict_cold_with_habitat_is_high(self):
        assert merge_ogt.classify(_row(ogt_measured=8.0, ogt_n_sources=1, meta_cold=True)) == "high"

    def test_strict_cold_corroborated_is_high_without_habitat(self):
        assert merge_ogt.classify(_row(ogt_measured=8.0, ogt_n_sources=2)) == "high"

    def test_strict_cold_single_source_is_medium(self):
        assert merge_ogt.classify(_row(ogt_measured=8.0, ogt_n_sources=1)) == "medium"

    def test_lenient_band_is_inclusive_at_20(self):
        """63 r232 genomes sit at exactly 20.0 C; a strict '<' dropped all of them."""
        assert merge_ogt.classify(_row(ogt_measured=20.0, ogt_n_sources=2, meta_cold=True)) == "high"
        assert merge_ogt.classify(_row(ogt_measured=20.0, ogt_n_sources=2)) == "medium"

    def test_tmin_below_freezing_promotes_within_lenient_band(self):
        """Cryobacterium arcticum: Topt 20.0, Tmin -6 -> high on Tmin corroboration."""
        assert merge_ogt.classify(
            _row(ogt_measured=20.0, ogt_n_sources=3, tmin_measured=-6.0)) == "high"

    def test_tmin_never_admits_a_genome_on_its_own(self):
        """Tmin<=4 holds for 17% of clear mesophiles: eurythermy, not cold optimum."""
        assert merge_ogt.classify(
            _row(ogt_measured=27.5, ogt_n_sources=2, tmin_measured=0.0)) == "none"
        assert merge_ogt.classify(
            _row(ogt_measured=30.0, ogt_n_sources=1, tmin_measured=-5.0)) == "none"

    def test_tmin_above_freezing_does_not_promote(self):
        assert merge_ogt.classify(
            _row(ogt_measured=18.0, ogt_n_sources=2, tmin_measured=4.0)) == "medium"

    def test_psychrotolerant_band_requires_habitat(self):
        assert merge_ogt.classify(_row(ogt_measured=22.0, ogt_n_sources=2, meta_cold=True)) == "medium"
        assert merge_ogt.classify(_row(ogt_measured=22.0, ogt_n_sources=2)) == "none"

    def test_measured_warm_overrides_cold_metadata(self):
        """The point of measuring: cold-sounding source, warm measured optimum."""
        assert merge_ogt.classify(
            _row(ogt_measured=30.0, ogt_n_sources=2, meta_cold=True)) == "none"

    def test_metadata_only_is_low(self):
        assert merge_ogt.classify(_row(meta_cold=True)) == "low"

    def test_no_evidence_at_all_is_none(self):
        assert merge_ogt.classify(_row()) == "none"


class TestPooling:
    def test_mean_and_spread_across_sources(self):
        frames = [
            pd.DataFrame({"sp_norm": ["a", "b"], "ogt_tempura": [10.0, 40.0]}),
            pd.DataFrame({"sp_norm": ["a", "c"], "ogt_madin": [14.0, 22.0]}),
        ]
        pooled = merge_ogt.pool_sources(frames).set_index("sp_norm")
        assert pooled.loc["a", "ogt_n_sources"] == 2
        assert pooled.loc["a", "ogt_measured"] == pytest.approx(12.0)
        assert pooled.loc["a", "ogt_spread"] == pytest.approx(4.0)
        assert pooled.loc["b", "ogt_n_sources"] == 1
        assert pooled.loc["b", "ogt_spread"] == pytest.approx(0.0)


class TestMergeIntoLabels:
    def _labels(self):
        return pd.DataFrame({
            "accession": ["GB_1", "GB_2", "GB_3"],
            "gtdb_taxonomy": [
                "d__Bacteria;s__Cryobacterium arcticum",
                "d__Bacteria;s__Escherichia coli",
                "d__Bacteria;s__",                      # placeholder: no species key
            ],
            "meta_iso_psychrophile": [True, False, True],
            "meta_org_psychrophile": [False, False, False],
        })

    def _pooled(self):
        return merge_ogt.pool_sources([
            pd.DataFrame({"sp_norm": ["cryobacterium arcticum", "escherichia coli"],
                          "ogt_tempura": [20.0, 37.5]}),
            pd.DataFrame({"sp_norm": ["cryobacterium arcticum", "escherichia coli"],
                          "ogt_madin": [20.0, 37.0]}),
        ]).assign(tmin_measured=[-6.0, 8.0])

    def test_row_count_preserved_and_tiers_assigned(self):
        merged, stats = merge_ogt.merge_into_labels(self._labels(), self._pooled())
        assert len(merged) == 3
        assert stats["genomes_without_species_key"] == 1
        tiers = dict(zip(merged.accession, merged.psy_conf_measured))
        assert tiers["GB_1"] == "high"    # 20.0 C + cold habitat
        assert tiers["GB_2"] == "none"    # measured 37.25 C overrides nothing to claim
        assert tiers["GB_3"] == "low"     # habitat only, unmatched species

    def test_unmatched_species_are_reported_not_dropped(self):
        pooled = self._pooled()
        extra = pd.DataFrame({"sp_norm": ["nonexistent species"], "ogt_tempura": [5.0],
                              "ogt_n_sources": [1], "ogt_measured": [5.0],
                              "ogt_spread": [0.0], "tmin_measured": [np.nan]})
        pooled = pd.concat([pooled, extra], ignore_index=True)
        _, stats = merge_ogt.merge_into_labels(self._labels(), pooled)
        assert stats["pooled_species_unmatched"] >= 1

    def test_rerun_refreshes_rather_than_suffixing_columns(self):
        """The stage writes into its own input; a second run must not duplicate."""
        pooled = self._pooled()
        first, _ = merge_ogt.merge_into_labels(self._labels(), pooled)
        second, _ = merge_ogt.merge_into_labels(first, pooled)
        assert not [c for c in second.columns if c.endswith(("_x", "_y"))]
        assert second.shape[1] == first.shape[1]
        pd.testing.assert_series_equal(first.psy_conf_measured, second.psy_conf_measured)
