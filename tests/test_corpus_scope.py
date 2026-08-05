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
