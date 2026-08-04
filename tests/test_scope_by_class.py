"""Per-class protein scope in pair derivation (config dataset.protein_scope)."""
import pandas as pd, pytest
from eptrans.dataset import _derive_protein_pairs

def _labeled():
    # genome G_EXT_H (halophile, secreted scope), G_EXT_T (hyperthermophile, whole),
    # G_OUT shared as the outgroup for BOTH -- the reuse case that forces per-class
    # filtering. Each genome has one secreted protein (cluster c_sec) and one
    # non-secreted protein (cluster c_who).
    rows = []
    for g in ("G_EXT_H", "G_EXT_T", "G_OUT"):
        rows.append(dict(genome=g, tagged_id=f"{g}~sec", cluster="c_sec",
                         is_secreted=True, cs_prob=0.9))
        rows.append(dict(genome=g, tagged_id=f"{g}~who", cluster="c_who",
                         is_secreted=False, cs_prob=float("nan")))
    return pd.DataFrame(rows)

PAIRS = pd.DataFrame([
    dict(**{"class": "halophile"}, extremophile_acc="G_EXT_H", outgroup_acc="G_OUT"),
    dict(**{"class": "hyperthermophile"}, extremophile_acc="G_EXT_T", outgroup_acc="G_OUT"),
])
SCOPE = {"halophile": "secreted", "hyperthermophile": "whole_proteome"}

def _derive(**kw):
    return _derive_protein_pairs(_labeled(), "cluster", "genome", PAIRS,
                                 tiebreak="deterministic", **kw)

def test_secreted_class_gets_only_secreted_clusters():
    out = _derive(scope_by_class=SCOPE)
    hal = out[out["class"] == "halophile"]
    assert set(hal.cluster) == {"c_sec"}, set(hal.cluster)

def test_whole_proteome_class_gets_both_clusters():
    out = _derive(scope_by_class=SCOPE)
    hyp = out[out["class"] == "hyperthermophile"]
    assert set(hyp.cluster) == {"c_sec", "c_who"}, set(hyp.cluster)

def test_shared_outgroup_serves_both_scopes_simultaneously():
    # the whole point: G_OUT is the outgroup for both classes, at different scopes
    out = _derive(scope_by_class=SCOPE)
    assert (out.outgroup_acc == "G_OUT").all()
    assert len(out) == 3  # halophile 1 (sec) + hyperthermophile 2 (sec+who)

def test_scope_column_records_which_rule_applied():
    out = _derive(scope_by_class=SCOPE)
    assert dict(zip(out["class"], out.scope))["halophile"] == "secreted"
    assert dict(zip(out["class"], out.scope))["hyperthermophile"] == "whole_proteome"

def test_default_scope_applies_to_unlisted_class():
    out = _derive(scope_by_class={"halophile": "secreted"}, default_scope="whole_proteome")
    hyp = out[out["class"] == "hyperthermophile"]
    assert set(hyp.cluster) == {"c_sec", "c_who"}  # unlisted -> default whole_proteome

def test_missing_secreted_column_raises_not_silently_wrong():
    lab = _labeled().drop(columns=["is_secreted"])
    with pytest.raises(ValueError, match="is_secreted"):
        _derive_protein_pairs(lab, "cluster", "genome", PAIRS,
                              tiebreak="deterministic", scope_by_class=SCOPE)

def test_whole_proteome_only_needs_no_secreted_column():
    lab = _labeled().drop(columns=["is_secreted"])
    p = PAIRS[PAIRS["class"] == "hyperthermophile"]
    out = _derive_protein_pairs(lab, "cluster", "genome", p, tiebreak="deterministic",
                                scope_by_class={"hyperthermophile": "whole_proteome"})
    assert set(out.cluster) == {"c_sec", "c_who"}

def test_scope_none_is_opt_out_input_taken_as_is():
    """scope_by_class=None must NOT filter: the historical contract is that the
    caller already passed a table of the intended scope. Regression guard for
    every pre-existing caller of _derive_protein_pairs."""
    out = _derive_protein_pairs(_labeled(), "cluster", "genome", PAIRS,
                                tiebreak="deterministic")
    assert set(out.cluster) == {"c_sec", "c_who"}
    assert set(out.scope) == {"_asis"}

def test_scope_none_works_without_secreted_column():
    lab = _labeled().drop(columns=["is_secreted"])
    out = _derive_protein_pairs(lab, "cluster", "genome", PAIRS, tiebreak="deterministic")
    assert len(out) == 4  # 2 classes x 2 clusters, unfiltered

def test_explicit_empty_map_activates_default_scope():
    """{} is DIFFERENT from None: it opts in with every class on default_scope."""
    out = _derive_protein_pairs(_labeled(), "cluster", "genome", PAIRS,
                                tiebreak="deterministic", scope_by_class={})
    assert set(out.cluster) == {"c_sec"}
