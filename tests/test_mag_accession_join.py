"""MAG accession must match the ingested proteome accession, not mag_id."""
import pandas as pd, pytest, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "m03c", Path(__file__).resolve().parents[1] / "scripts/03c_merge_deepsea_mags.py")
m03c = importlib.util.module_from_spec(spec); spec.loader.exec_module(m03c)
from eptrans.gtdb import bare_accession

# Use the REAL config thresholds -- an earlier version of this test hand-wrote a
# flattened dict and crashed on th["temperature"], which is nested.
from eptrans.config import load_config
TH = load_config()["thresholds"]

def _mags():
    return pd.DataFrame({
        "mag_id": ["10A(CNS0876618)_bin.16", "FDZ071-WW16-18(OES00301993)_bin.24",
                   "clean_bin.7"],
        "domain": ["d__Bacteria"]*3, "phylum": ["p__X"]*3, "class": ["c__X"]*3,
        "order": ["o__X"]*3, "family": ["f__X"]*3, "genus": ["g__X"]*3,
        "species": ["s__X"]*3, "sample_id": ["S1", "S2", "S3"],
        "ecosystem": ["HV"]*3, "isolation_source": ["hydrothermal vent"]*3,
        "temperature_optimum": [70.0, 30.0, 12.0], "temperature_min": [50.0, 10.0, 2.0],
        "ph_optimum": [7.0]*3, "salinity_optimum": [3.0]*3,
        "temperature_c": [None]*3, "insitu_temp_usable": [""]*3,
        "depth_m": [2000.0]*3, "metadata_classes": ["thermophile", "", "psychrophile"]})

MAP = {"10A(CNS0876618)_bin.16": "CU_CUST_000000001.1",
       "FDZ071-WW16-18(OES00301993)_bin.24": "CU_CUST_000000002.1"}


def test_accession_comes_from_id_map_not_mag_id():
    out = m03c.build_mag_rows(_mags(), TH, id_map=MAP)
    assert out["accession"].iloc[0] == "CU_CUST_000000001.1"
    assert "(" not in out["accession"].iloc[0]


def test_unmapped_mag_falls_back_and_is_flagged():
    out = m03c.build_mag_rows(_mags(), TH, id_map=MAP)
    assert out["has_proteome"].tolist() == [True, True, False]
    assert out["accession"].iloc[2] == "CU_clean_bin.7"


def test_accession_joins_to_protein_genome_via_bare_accession():
    """The actual failure mode: assign_labels matches on bare_accession equality."""
    out = m03c.build_mag_rows(_mags(), TH, id_map=MAP)
    protein_genome = "CU_CUST_000000001.1"          # from the real FASTA headers
    assert bare_accession(out["accession"].iloc[0]) == bare_accession(protein_genome)


def test_without_id_map_the_join_would_fail():
    """Regression guard: documents the bug so it cannot silently return."""
    out = m03c.build_mag_rows(_mags(), TH, id_map=None)
    assert bare_accession(out["accession"].iloc[0]) != bare_accession("CU_CUST_000000001.1")
    assert not out["has_proteome"].any()


def test_parenthesised_ids_never_reach_the_accession_column():
    out = m03c.build_mag_rows(_mags(), TH, id_map=MAP)
    mapped = out[out["has_proteome"]]
    assert not mapped["accession"].str.contains(r"[()]").any()
