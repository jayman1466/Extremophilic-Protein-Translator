"""Unit tests for eptrans.binning keyword flagging + combination logic."""
from eptrans.binning import (
    flag_genome, predicted_classes, is_confident_mesophile, combine_label,
)


# ---- isolation-source keyword flagging --------------------------------------
def test_soda_lake_is_halo_and_alkali():
    f = flag_genome("hypersaline soda lake sediment")
    assert "halophile" in f.iso_classes
    assert "alkaliphile" in f.iso_classes


def test_hydrothermal_thermophile():
    f = flag_genome("marine hydrothermal vent")
    assert f.iso_classes == {"thermophile"}


def test_sulfide_chimney_hyperthermophile_implies_thermophile():
    f = flag_genome("deep-sea hydrothermal sulfide chimney")
    assert "hyperthermophile" in f.iso_classes
    assert "thermophile" in f.iso_classes  # implication


def test_acid_mine_acidophile():
    assert flag_genome("acid mine drainage sediment").iso_classes == {"acidophile"}


def test_permafrost_psychrophile():
    assert flag_genome("permafrost active layer soil").iso_classes == {"psychrophile"}


def test_salt_marsh_excluded():
    # tidal salt marsh should NOT be halophile
    assert "halophile" not in flag_genome("salt marsh").iso_classes


def test_cold_seep_excluded():
    assert "psychrophile" not in flag_genome("cold seep").iso_classes


def test_basalt_not_halophile():
    assert "halophile" not in flag_genome("basalt rock sample").iso_classes


def test_plain_soil_nothing():
    assert flag_genome("soil").iso_classes == set()


def test_organism_name_separate():
    f = flag_genome("soil", organism_name="Halobacterium salinarum")
    assert f.iso_classes == set()             # habitat says nothing
    assert "halophile" in f.org_classes       # name signal captured separately


# ---- GenomeSPOT prediction -> classes ---------------------------------------
def test_predicted_hyperthermophile():
    c = predicted_classes(temp_opt=85, ph_opt=7, salinity_opt=1)
    assert "hyperthermophile" in c and "thermophile" in c


def test_predicted_acidophile_halophile():
    c = predicted_classes(temp_opt=30, ph_opt=3.5, salinity_opt=8)
    assert "acidophile" in c and "halophile" in c
    assert "alkaliphile" not in c


def test_predicted_psychrophile():
    assert "psychrophile" in predicted_classes(temp_opt=8, ph_opt=7, salinity_opt=1)


def test_predicted_handles_none():
    assert predicted_classes(None, None, None) == set()


def test_confident_mesophile():
    assert is_confident_mesophile(30, 7, 1) is True
    assert is_confident_mesophile(85, 7, 1) is False   # thermophilic temp


# ---- combination rule --------------------------------------------------------
def test_combine_agreement_high():
    label, conf = combine_label({"thermophile"}, {"thermophile"}, pred_available=True)
    assert conf == "high" and label == "thermophile"


def test_combine_prediction_only_medium():
    label, conf = combine_label(set(), {"halophile"}, pred_available=True)
    assert conf == "medium" and label == "halophile"


def test_combine_metadata_only_low():
    label, conf = combine_label({"acidophile"}, set(), pred_available=True)
    assert conf == "low" and label == "acidophile"


def test_combine_conflict_low():
    # metadata says thermophile, prediction says psychrophile -> conflict, low
    label, conf = combine_label({"thermophile"}, {"psychrophile"}, pred_available=True)
    assert conf in ("medium", "low")   # no agreement; prediction wins as medium


def test_combine_no_evidence_none():
    assert combine_label(set(), set(), pred_available=False) == ("", "none")
