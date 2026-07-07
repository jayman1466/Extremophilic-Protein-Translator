"""Phylogenetically-controlled genome selection.

Two competing goals, per the project design:

1. **Diversity** - extremophiles for a given class should span the tree, not
   cluster in one or two clades. Otherwise the model learns "is this genome a
   Thermotoga?" rather than "is this genome thermophilic?".
2. **Matched outgroups** - each extremophile should be paired with a mesophile
   that is phylogenetically *close* (same genus/family/...), so that when the
   model is trained on extremophile-vs-mesophile it cannot separate the two by
   clade alone. The trait varies *within* a clade in the training data.

Strategy
--------
Extremophile selection (per class):
    * cap the number of genomes drawn from any one lineage at a chosen rank
      (default: family) so no clade dominates - "diversity capping".
    * prefer high-confidence labels (metadata + prediction agree).

Outgroup selection (per selected extremophile):
    * find confident mesophiles in the SAME lineage at the finest rank possible,
      walking up genus -> family -> order -> class until a match is found.
    * record the matched rank so the phylogenetic distance of each pair is known.

The result is a balanced, clade-matched selection where the extremophile trait
is decorrelated from deep phylogeny as much as the data allows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import load_config
from .gtdb import GTDB_RANKS

# Ranks from finest to coarsest for outgroup matching.
_RANK_ORDER = ["species", "genus", "family", "order", "class", "phylum", "domain"]


@dataclass
class SelectionResult:
    extremophiles: pd.DataFrame       # selected extremophile genomes (+ class, confidence)
    outgroups: pd.DataFrame           # selected mesophile outgroups (+ matched rank, partner)
    pairs: pd.DataFrame               # extremophile <-> outgroup pairing table
    stats: dict = field(default_factory=dict)


def _confidence_rank(conf: str) -> int:
    return {"high": 0, "medium": 1, "low": 2, "none": 3}.get(conf, 3)


def select_extremophiles(
    df: pd.DataFrame,
    cls: str,
    max_per_lineage: int = 5,
    lineage_rank: str = "family",
    max_total: int | None = None,
    confidence_col: str = "final_confidence",
    seed: int = 1466,
) -> pd.DataFrame:
    """Diversity-capped selection of extremophiles for one class.

    Draws at most ``max_per_lineage`` genomes from each ``lineage_rank`` group,
    preferring higher-confidence labels, until ``max_total`` is reached.
    """
    cls_col = f"final_{cls}"
    pool = df[df[cls_col].fillna(False)].copy()
    if pool.empty:
        return pool

    rng = np.random.default_rng(seed)
    pool["_conf_rank"] = pool[confidence_col].map(_confidence_rank) if confidence_col in pool else 1
    pool["_jitter"] = rng.random(len(pool))
    # Sort so best-confidence first, random within tier.
    pool = pool.sort_values(["_conf_rank", "_jitter"])

    selected_idx = []
    per_lineage: dict[str, int] = {}
    for idx, row in pool.iterrows():
        lin = row.get(lineage_rank, "") or "__NA__"
        if per_lineage.get(lin, 0) >= max_per_lineage:
            continue
        selected_idx.append(idx)
        per_lineage[lin] = per_lineage.get(lin, 0) + 1
        if max_total and len(selected_idx) >= max_total:
            break

    out = pool.loc[selected_idx].drop(columns=["_conf_rank", "_jitter"], errors="ignore").copy()
    out["selected_class"] = cls
    return out


def find_outgroup(
    extremophile_row: pd.Series,
    mesophile_pool: pd.DataFrame,
    used_outgroups: set,
    match_ranks: list[str] | None = None,
) -> tuple[int | None, str | None]:
    """Find the phylogenetically-closest unused confident mesophile.

    Walks from finest rank (genus) up to coarsest, returning the first pool
    member sharing the extremophile's taxon at that rank. Returns
    (pool_index, matched_rank) or (None, None).
    """
    match_ranks = match_ranks or ["genus", "family", "order", "class", "phylum"]
    for rank in match_ranks:
        taxon = extremophile_row.get(rank)
        if not taxon or taxon == "":
            continue
        cand = mesophile_pool[(mesophile_pool[rank] == taxon)
                              & (~mesophile_pool.index.isin(used_outgroups))]
        if not cand.empty:
            return cand.index[0], rank
    return None, None


def select_with_outgroups(
    labels: pd.DataFrame,
    classes: list[str] | None = None,
    max_per_lineage: int = 5,
    lineage_rank: str = "family",
    max_total_per_class: int | None = 100,
    mesophile_col: str = "confident_mesophile",
    acc_col: str = "accession",
    seed: int = 1466,
) -> SelectionResult:
    """Full phylo-controlled selection: diverse extremophiles + matched outgroups.

    Args:
        labels: combined-labels table (stage 03) with per-class ``final_<cls>``
            booleans, taxonomy rank columns, and a ``confident_mesophile`` column.
    """
    cfg = load_config()
    classes = classes or ["thermophile", "hyperthermophile", "psychrophile",
                          "acidophile", "alkaliphile", "halophile"]

    # Ensure taxonomy rank columns exist.
    missing = [r for r in GTDB_RANKS if r not in labels.columns]
    if missing:
        raise ValueError(f"labels missing taxonomy columns {missing}; run expand_taxonomy first")

    mesophile_pool = labels[labels[mesophile_col].fillna(False)].copy()

    all_extremo = []
    all_outgroups = []
    pair_rows = []
    used_outgroups: set = set()

    for cls in classes:
        extremo = select_extremophiles(
            labels, cls, max_per_lineage=max_per_lineage, lineage_rank=lineage_rank,
            max_total=max_total_per_class, seed=seed,
        )
        if extremo.empty:
            continue
        all_extremo.append(extremo)

        for eidx, erow in extremo.iterrows():
            oidx, matched_rank = find_outgroup(erow, mesophile_pool, used_outgroups)
            if oidx is not None:
                used_outgroups.add(oidx)
                orow = mesophile_pool.loc[oidx]
                all_outgroups.append(oidx)
                pair_rows.append({
                    "class": cls,
                    "extremophile_acc": erow[acc_col],
                    "extremophile_confidence": erow.get("final_confidence"),
                    "outgroup_acc": orow[acc_col],
                    "matched_rank": matched_rank,
                    f"shared_{matched_rank}": erow.get(matched_rank),
                })
            else:
                pair_rows.append({
                    "class": cls,
                    "extremophile_acc": erow[acc_col],
                    "extremophile_confidence": erow.get("final_confidence"),
                    "outgroup_acc": None,
                    "matched_rank": None,
                })

    extremo_df = (pd.concat(all_extremo, ignore_index=True) if all_extremo
                  else pd.DataFrame(columns=labels.columns))
    outgroup_df = (mesophile_pool.loc[sorted(used_outgroups)].copy() if used_outgroups
                   else pd.DataFrame(columns=labels.columns))
    pairs_df = pd.DataFrame(pair_rows)

    stats = {
        "n_extremophiles": int(len(extremo_df)),
        "n_outgroups": int(len(outgroup_df)),
        "n_pairs_matched": int(pairs_df["outgroup_acc"].notna().sum()) if len(pairs_df) else 0,
        "n_pairs_unmatched": int(pairs_df["outgroup_acc"].isna().sum()) if len(pairs_df) else 0,
        "by_class": {},
        "match_rank_dist": {},
    }
    if len(pairs_df):
        for cls in classes:
            sub = pairs_df[pairs_df["class"] == cls]
            if len(sub):
                stats["by_class"][cls] = {
                    "extremophiles": int(len(sub)),
                    "matched": int(sub["outgroup_acc"].notna().sum()),
                }
        stats["match_rank_dist"] = (pairs_df["matched_rank"].value_counts(dropna=True).to_dict())

    return SelectionResult(extremo_df, outgroup_df, pairs_df, stats)
