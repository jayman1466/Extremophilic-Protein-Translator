"""Reconcile precomputed GenomeSPOT predictions across GTDB releases.

The GenomeSPOT paper (Barnum et al. 2024, bioRxiv 2024.03.22.586313) applied its
models to the species representatives of GTDB release **r214** (order ~85k
genomes; see the paper for the exact figure). The per-genome predictions themselves are NOT
shipped in the GenomeSPOT repo; the ``analyze_all_species`` notebook reads them
from a local ``data/predictions_gtdb/`` directory and merges them into a table
the notebook names ``supplementary_data_4.tsv`` (presumed to be the paper's
Supplementary Data 4 - this has not been independently verified here, as the
bioRxiv supplementary files were not retrievable from this environment). The
precomputed predictions must therefore be supplied to this module by the user.

Our pipeline targets GTDB **r232** (199,923 reps). Rather than recompute
everything, we reuse whatever precomputed predictions are provided where the
*same assembly* is still a representative in r232, and compute the *delta* - the
r232 reps that need a fresh GenomeSPOT run. This module is agnostic to the exact
provenance/schema of the precomputed table: it matches on accession and carries
over whichever prediction columns are present.

Accession reconciliation
------------------------
NCBI assembly accessions look like ``GCA_000005845.2`` (GenBank) or
``GCF_000005845.2`` (RefSeq). GTDB prefixes them ``GB_`` / ``RS_``. Between
releases the same organism can appear under:

* the **same** accession                         -> exact reuse
* a **bumped version** (``...845.1`` -> ``...845.2``) -> same assembly, reuse
* a **GenBank<->RefSeq swap** (``GCA_...`` <-> ``GCF_...``) - GenBank and
  RefSeq copies of one assembly share the **same 9-digit numeric id**, so the
  numeric root bridges them -> reuse

We therefore match on three keys, strongest first, and record which level
matched so the reuse can be audited:

1. ``exact``   - full bare accession incl. source prefix and version
2. ``noversion`` - source (GCA/GCF) + numeric, ignoring ``.version``
3. ``assembly`` - numeric id only (bridges GCA<->GCF and version)

A r232 rep with no match at any level is part of the **recompute delta**.
A precomputed row matching no r232 rep is **dropped** (organism not a rep in r232).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .gtdb import bare_accession, accession_root

MATCH_LEVELS = ["exact", "noversion", "assembly"]


def _keys(accession: str) -> dict[str, str]:
    """Compute the three reconciliation keys for one accession."""
    bare = bare_accession(accession)                 # e.g. GCF_000005845.2
    kind, numeric = accession_root(accession)         # ('GCF', '000005845')
    stem = bare.split(".")[0]                          # GCF_000005845 (no version)
    return {
        "exact": bare,
        "noversion": stem,
        "assembly": numeric,   # bridges GCA<->GCF
    }


@dataclass
class ReconcileResult:
    """Outcome of reconciling r232 reps against precomputed predictions."""
    reconciled: pd.DataFrame       # r232 reps with reused predictions (+ match level) or NaN
    delta: pd.DataFrame            # r232 reps needing recompute (no match)
    dropped_precomputed: pd.DataFrame  # precomputed rows not matching any r232 rep
    stats: dict


def build_precomputed_index(precomp: pd.DataFrame, acc_col: str) -> dict[str, dict[str, int]]:
    """Index precomputed rows by each match key -> row position.

    Returns {level: {key: iloc}}. For a given level, later duplicate keys
    overwrite earlier ones (rare; logged in stats via collision count).
    """
    index: dict[str, dict[str, int]] = {lvl: {} for lvl in MATCH_LEVELS}
    collisions = {lvl: 0 for lvl in MATCH_LEVELS}
    for i, acc in enumerate(precomp[acc_col].astype(str)):
        ks = _keys(acc)
        for lvl in MATCH_LEVELS:
            k = ks[lvl]
            if k in index[lvl]:
                collisions[lvl] += 1
            index[lvl][k] = i
    return index, collisions


def reconcile(
    reps: pd.DataFrame,
    precomp: pd.DataFrame,
    reps_acc_col: str = "accession",
    precomp_acc_col: str = "accession",
    prediction_cols: list[str] | None = None,
) -> ReconcileResult:
    """Reconcile r232 reps against precomputed (older-release) predictions.

    Args:
        reps: r232 representatives; must have ``reps_acc_col`` (GTDB accession,
            with or without GB_/RS_ prefix).
        precomp: precomputed predictions; must have ``precomp_acc_col``.
        prediction_cols: columns to carry over from precomp. If None, all
            precomp columns except the accession column are carried.

    Returns:
        ReconcileResult with reconciled/delta/dropped frames + stats.
    """
    if prediction_cols is None:
        prediction_cols = [c for c in precomp.columns if c != precomp_acc_col]

    index, collisions = build_precomputed_index(precomp, precomp_acc_col)
    precomp_reset = precomp.reset_index(drop=True)

    match_level: list[str | None] = []
    matched_precomp_iloc: list[int | None] = []
    used_precomp_rows: set[int] = set()

    for acc in reps[reps_acc_col].astype(str):
        ks = _keys(acc)
        hit_iloc = None
        hit_level = None
        for lvl in MATCH_LEVELS:
            k = ks[lvl]
            if k in index[lvl]:
                hit_iloc = index[lvl][k]
                hit_level = lvl
                break
        match_level.append(hit_level)
        matched_precomp_iloc.append(hit_iloc)
        if hit_iloc is not None:
            used_precomp_rows.add(hit_iloc)

    out = reps.copy().reset_index(drop=True)
    out["genomespot_match_level"] = match_level
    out["genomespot_reused"] = [lvl is not None for lvl in match_level]

    # Attach carried prediction columns from matched precomp rows.
    for col in prediction_cols:
        vals = []
        src = precomp_reset[col]
        for iloc in matched_precomp_iloc:
            vals.append(src.iloc[iloc] if iloc is not None else pd.NA)
        out[f"precomp_{col}"] = vals

    delta = out[~out["genomespot_reused"]].copy()
    dropped_mask = ~precomp_reset.index.isin(used_precomp_rows)
    dropped = precomp_reset[dropped_mask].copy()

    stats = {
        "n_reps": int(len(reps)),
        "n_precomputed": int(len(precomp)),
        "n_reused": int(out["genomespot_reused"].sum()),
        "n_delta": int(len(delta)),
        "reuse_by_level": {lvl: int(sum(1 for x in match_level if x == lvl)) for lvl in MATCH_LEVELS},
        "n_dropped_precomputed": int(len(dropped)),
        "precomp_key_collisions": collisions,
        "reuse_fraction": round(float(out["genomespot_reused"].mean()), 4) if len(out) else 0.0,
    }
    return ReconcileResult(reconciled=out, delta=delta, dropped_precomputed=dropped, stats=stats)


def attach_genome_paths(
    df: pd.DataFrame,
    genome_index_path: str,
    acc_col: str = "accession",
    out_col: str = "genome_fna_path",
) -> pd.DataFrame:
    """Add absolute genome .fna.gz paths from genome_index.tsv (keyed on bare acc)."""
    idx: dict[str, str] = {}
    with open(genome_index_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                idx[parts[0]] = parts[1]
    out = df.copy()
    out[out_col] = [idx.get(bare_accession(a)) for a in out[acc_col].astype(str)]
    return out
