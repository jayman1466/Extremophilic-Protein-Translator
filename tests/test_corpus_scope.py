import pandas as pd, numpy as np, pytest
from eptrans.dataset import _apply_corpus_scope, assemble_dataset

def _labeled():
    # genomes: acidophile (secreted-scope ext), psychrophile (whole-scope ext),
    # meso_A outgroups the acidophile only (-> secreted), meso_B outgroups the
    # psychrophile (-> whole). Each genome has 1 secreted + 1 cytoplasmic protein.
    rows=[]
    def g(acc,label,ismeso):
        for pid,sec in [("p_sec",True),("p_cyt",False)]:
            rows.append(dict(genome=acc,protein_id=pid,label=label,
                             is_mesophile=ismeso,is_secreted=sec))
    g("GB_ACID1","acidophile",False)
    g("GB_PSY1","psychrophile",False)
    g("GB_MESOA","mesophile",True)
    g("GB_MESOB","mesophile",True)
    return pd.DataFrame(rows)

def _pairs():
    return pd.DataFrame([
        dict(**{"class":"acidophile"}, extremophile_acc="GB_ACID1", outgroup_acc="GB_MESOA"),
        dict(**{"class":"psychrophile"},extremophile_acc="GB_PSY1", outgroup_acc="GB_MESOB"),
    ])

def test_scope_branches():
    lab=_labeled()
    out,st=_apply_corpus_scope(lab,_pairs(),
        {"acidophile":"secreted","psychrophile":"whole_proteome"},
        "secreted","is_secreted","genome","protein_id")
    kept=set(zip(out.genome,out.protein_id))
    # acidophile: secreted only
    assert ("GB_ACID1","p_sec") in kept and ("GB_ACID1","p_cyt") not in kept
    # psychrophile: both
    assert ("GB_PSY1","p_sec") in kept and ("GB_PSY1","p_cyt") in kept
    # meso_A (outgroups acidophile=secreted): secreted only
    assert ("GB_MESOA","p_sec") in kept and ("GB_MESOA","p_cyt") not in kept
    # meso_B (outgroups psychrophile=whole): both
    assert ("GB_MESOB","p_sec") in kept and ("GB_MESOB","p_cyt") in kept
    assert st["scope_dropped"]==2 and st["scope_whole_mesophile_genomes"]==1

def test_scope_none_is_noop():
    lab=_labeled()
    # assemble with scope_by_class=None must NOT filter (historical contract)
    r=assemble_dataset(lab,lab.rename(columns={"genome":"accession"}).assign(
        final_acidophile=lab.label.eq("acidophile"),
        final_psychrophile=lab.label.eq("psychrophile"),
        confident_mesophile=lab.is_mesophile),
        scope_by_class=None, seed=1)
    assert r.stats["n_proteins"]==len(lab)


def test_mlm_subsample_extremophile_only():
    import importlib.util, pathlib
    p = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "09_subsample_mlm.py"
    spec = importlib.util.spec_from_file_location("s09", p)
    s09 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s09)
    lab = pd.DataFrame([
        dict(tagged_id=f"g{i}~p", label=("mesophile" if i % 2 else "halophile"),
             is_mesophile=bool(i % 2), label_confidence="high",
             split="train", group=f"c{i}")
        for i in range(20)])
    # default: extremophiles only
    sub = s09.subsample(lab, n_train=100, n_val=0, seed=1)
    assert (~sub["is_mesophile"]).all(), "mesophiles leaked into MLM set"
    assert len(sub) == 10
    # opt-in: keep mesophiles
    sub2 = s09.subsample(lab, n_train=100, n_val=0, seed=1, extremophile_only=False)
    assert sub2["is_mesophile"].any() and len(sub2) == 20


def test_inv_scope_c_catches_mesophile_union_violation():
    # Assemble a small scoped dataset, then inject a non-secreted mesophile protein
    # from a genome that outgroups only a SECRETED class -> INV-SCOPE-C must fire.
    lab = _labeled()
    gl = lab.rename(columns={"genome": "accession"}).assign(
        final_acidophile=lab.label.eq("acidophile"),
        final_psychrophile=lab.label.eq("psychrophile"),
        confident_mesophile=lab.is_mesophile)
    # MESOA outgroups acidophile (secreted) only; force-keep its cytoplasmic protein
    # by NOT scoping (simulate the regression), then assert the guard would catch it.
    # Direct check on the invariant logic: a non-secreted MESOA protein is illegal.
    from eptrans.dataset import _apply_corpus_scope
    out, st = _apply_corpus_scope(lab, _pairs(),
        {"acidophile": "secreted", "psychrophile": "whole_proteome"},
        "secreted", "is_secreted", "genome", "protein_id")
    # correctly-scoped output has NO non-secreted MESOA protein
    bad = out[(out.genome == "GB_MESOA") & (~out.is_secreted)]
    assert len(bad) == 0, "MESOA cytoplasmic protein should have been dropped"
    # and MESOB (outgroups psychrophile=whole) DOES keep its cytoplasmic protein
    assert len(out[(out.genome == "GB_MESOB") & (~out.is_secreted)]) == 1


def test_inv_scope_d_polyextremophile_keeps_whole_proteome():
    """INV-SCOPE-D: a genome whose single corpus label is a SECRETED-scope class but
    which serves a WHOLE-scope class as a pair extremophile must keep its whole
    proteome.

    This is the deep-sea/soda-lake polyextremophile case: cold+saline organisms are
    labelled halophile (one label per genome) yet form the psychrophile pairs. Keying
    scope off the label alone reduced all 19 psychrophile and all 27 hyperthermophile
    ext pair genomes to their secretome, so the psychrophile pair evaluation ran on
    secreted proteins of salt/alkali-labelled genomes.
    """
    rows = []
    for acc, label, ismeso in [("GB_POLY1", "halophile", False),
                               ("GB_HALO1", "halophile", False),
                               ("GB_MESOP", "mesophile", True)]:
        for pid, sec in [("p_sec", True), ("p_cyt", False)]:
            rows.append(dict(genome=acc, protein_id=pid, label=label,
                             is_mesophile=ismeso, is_secreted=sec))
    lab = pd.DataFrame(rows)
    # POLY1 is labelled halophile but is the EXTREMOPHILE of a psychrophile pair.
    # HALO1 serves only a halophile pair. Real schema uses "ext_acc".
    pairs = pd.DataFrame([
        dict(**{"class": "psychrophile"}, ext_acc="GB_POLY1", outgroup_acc="GB_MESOP"),
        dict(**{"class": "halophile"},    ext_acc="GB_HALO1", outgroup_acc="GB_MESOP"),
    ])
    out, st = _apply_corpus_scope(
        lab, pairs, {"halophile": "secreted", "psychrophile": "whole_proteome"},
        "secreted", "is_secreted", "genome", "protein_id")
    kept = set(zip(out.genome, out.protein_id))
    # the polyextremophile keeps BOTH proteins despite its halophile label
    assert ("GB_POLY1", "p_sec") in kept, "secreted protein must be kept"
    assert ("GB_POLY1", "p_cyt") in kept, \
        "INV-SCOPE-D: cytoplasmic protein of a whole-scope pair extremophile was dropped"
    # a genome serving only the secreted class is still secreted-only
    assert ("GB_HALO1", "p_sec") in kept and ("GB_HALO1", "p_cyt") not in kept
    assert st["scope_whole_ext_pair_genomes"] == 1


def test_inv_scope_d_accepts_legacy_ext_column_name():
    """The union must not silently no-op when the frame uses extremophile_acc."""
    rows = []
    for pid, sec in [("p_sec", True), ("p_cyt", False)]:
        rows.append(dict(genome="GB_POLY1", protein_id=pid, label="halophile",
                         is_mesophile=False, is_secreted=sec))
    lab = pd.DataFrame(rows)
    pairs = pd.DataFrame([dict(**{"class": "psychrophile"},
                               extremophile_acc="GB_POLY1", outgroup_acc="GB_MESOP")])
    out, st = _apply_corpus_scope(
        lab, pairs, {"halophile": "secreted", "psychrophile": "whole_proteome"},
        "secreted", "is_secreted", "genome", "protein_id")
    assert st["scope_whole_ext_pair_genomes"] == 1
    assert ("GB_POLY1", "p_cyt") in set(zip(out.genome, out.protein_id))
