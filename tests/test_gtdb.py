"""Unit tests for eptrans.gtdb (offline: accession + taxonomy helpers)."""
from eptrans.gtdb import (
    bare_accession, source_prefix, accession_root, parse_gtdb_taxonomy, GTDB_RANKS,
)


def test_bare_accession():
    assert bare_accession("RS_GCF_000005845.2") == "GCF_000005845.2"
    assert bare_accession("GB_GCA_000008085.1") == "GCA_000008085.1"
    assert bare_accession("GCF_000005845.2") == "GCF_000005845.2"  # already bare


def test_source_prefix():
    assert source_prefix("RS_GCF_000005845.2") == "RS_"
    assert source_prefix("GB_GCA_000008085.1") == "GB_"
    assert source_prefix("GCF_000005845.2") is None


def test_accession_root():
    assert accession_root("GCF_000005845.2") == ("GCF", "000005845")
    assert accession_root("RS_GCF_000005845.2") == ("GCF", "000005845")
    assert accession_root("GB_GCA_000008085.1") == ("GCA", "000008085")


def test_parse_taxonomy_full():
    t = ("d__Archaea;p__Methanobacteriota;c__Methanobacteria;o__Methanobacteriales;"
         "f__Methanobacteriaceae;g__Methanobrevibacter;s__Methanobrevibacter smithii")
    d = parse_gtdb_taxonomy(t)
    assert d["domain"] == "Archaea"
    assert d["phylum"] == "Methanobacteriota"
    assert d["species"] == "Methanobrevibacter smithii"
    assert set(d.keys()) == set(GTDB_RANKS)


def test_parse_taxonomy_empty_ranks():
    d = parse_gtdb_taxonomy("d__Bacteria;p__Pseudomonadota;c__;o__;f__;g__;s__")
    assert d["phylum"] == "Pseudomonadota"
    assert d["class"] == ""
    assert d["species"] == ""


def test_parse_taxonomy_bad_input():
    d = parse_gtdb_taxonomy(None)
    assert all(v == "" for v in d.values())
    assert set(d.keys()) == set(GTDB_RANKS)
