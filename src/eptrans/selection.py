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
    confidence_levels: tuple | None = None,
    max_per_sample: int | None = None,
    sample_col: str = "source_sample_id",
    seed: int = 1466,
) -> pd.DataFrame:
    """Diversity-capped selection of extremophiles for one class.

    Draws at most ``max_per_lineage`` genomes from each ``lineage_rank`` group,
    preferring higher-confidence labels, until ``max_total`` is reached.

    If ``confidence_levels`` is given (e.g. ``("high", "medium")``), genomes
    outside those tiers are excluded entirely (not just deprioritised).

    ``max_per_sample`` additionally caps how many genomes may come from one
    environmental sample, identified by ``sample_col``. This exists because
    metagenome-assembled genomes are labelled from SAMPLE-level metadata: the
    deep-sea set contributes 4,084 MAGs from only 858 samples, and its worst
    single sample yields 38 thermophile MAGs spanning 30 distinct families. Those
    38 pass a per-family cap untouched while resting on ONE environmental
    observation -- a single temperature reading -- so counting them as 38
    independent pieces of evidence is pseudoreplication. The per-lineage cap
    cannot catch this: it controls phylogenetic redundancy, and co-occurring bins
    from one sample are phylogenetically diverse by construction.

    Rows with no value in ``sample_col`` (every isolate genome in GTDB, which is
    its own sample) are exempt: NaN means "not from a shared metagenome", not
    "unknown sample". Passing ``max_per_sample=None`` disables the cap entirely
    and reproduces pre-MAG behaviour exactly.
    """
    cls_col = f"final_{cls}"
    pool = df[df[cls_col].fillna(False)].copy()
    if confidence_levels is not None and confidence_col in pool:
        pool = pool[pool[confidence_col].isin(confidence_levels)]
    if pool.empty:
        return pool

    rng = np.random.default_rng(seed)
    pool["_conf_rank"] = pool[confidence_col].map(_confidence_rank) if confidence_col in pool else 1
    pool["_jitter"] = rng.random(len(pool))
    # Sort so best-confidence first, random within tier.
    pool = pool.sort_values(["_conf_rank", "_jitter"])

    selected_idx = []
    per_lineage: dict[str, int] = {}
    per_sample: dict[str, int] = {}
    for idx, row in pool.iterrows():
        lin = row.get(lineage_rank, "") or "__NA__"
        if per_lineage.get(lin, 0) >= max_per_lineage:
            continue
        # Per-sample cap: only for genomes that declare a source sample. A blank
        # means an isolate genome (its own sample), which must not be pooled into
        # one shared bucket -- doing so would cap ALL of GTDB at max_per_sample.
        smp = row.get(sample_col, None) if sample_col in pool.columns else None
        has_sample = smp is not None and pd.notna(smp) and str(smp) != ""
        if max_per_sample is not None and has_sample:
            if per_sample.get(smp, 0) >= max_per_sample:
                continue
        selected_idx.append(idx)
        per_lineage[lin] = per_lineage.get(lin, 0) + 1
        if max_per_sample is not None and has_sample:
            per_sample[smp] = per_sample.get(smp, 0) + 1
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
    confidence_levels: tuple | None = None,
    reuse_outgroups: bool = False,
    max_per_sample: int | None = None,
    sample_col: str = "source_sample_id",
    require_col: str | None = None,
    seed: int = 1466,
) -> SelectionResult:
    """Full phylo-controlled selection: diverse extremophiles + matched outgroups.

    Args:
        labels: combined-labels table (stage 03) with per-class ``final_<cls>``
            booleans, taxonomy rank columns, and a ``confident_mesophile`` column.
        max_per_sample: cap on genomes drawn from any one environmental sample
            (see ``select_extremophiles``). Applies to the extremophile draw only;
            mesophile outgroups are matched individually by taxonomy, so a sample
            cannot dominate them the same way. ``None`` disables it.
        require_col: name of a boolean column that a genome must satisfy to be
            selectable at all, applied to BOTH extremophiles and outgroups.

            WHY THIS EXISTS: pass ``"has_proteome"``. A genome with no proteome
            file on disk contributes no protein sequences, so any pair built on it
            derives ZERO protein pairs -- silently, since nothing downstream errors
            on an absent FASTA entry. Measured on the 2026-08-04 run: 10 selected
            MAGs had ``has_proteome == False``, voiding 10 genome pairs including
            **2 of the only 12 high-confidence psychrophile pairs (17% of that
            tier)**. Filtering here rather than downstream means the diversity caps
            and outgroup matching spend their budget on usable genomes and pick
            replacements, instead of the pairs evaporating after assembly.
    """
    cfg = load_config()
    classes = classes or ["thermophile", "hyperthermophile", "psychrophile",
                          "acidophile", "alkaliphile", "halophile"]

    # Ensure taxonomy rank columns exist.
    missing = [r for r in GTDB_RANKS if r not in labels.columns]
    if missing:
        raise ValueError(f"labels missing taxonomy columns {missing}; run expand_taxonomy first")

    # Drop unusable genomes BEFORE any selection, so caps and matching spend their
    # budget on genomes that can actually contribute proteins (see require_col).
    if require_col:
        if require_col not in labels.columns:
            raise ValueError(f"require_col={require_col!r} not in labels columns")
        # NULL means "not annotated by this ingest", NOT "absent". Only the custom
        # MAG ingest writes has_proteome; all 901,341 GTDB rows carry None, and every
        # GTDB representative does have a proteome. Treating null as False would drop
        # the entire GTDB catalogue -- so null is retained and only an explicit
        # False excludes.
        col = labels[require_col]
        keep = col.isna() | col.fillna(False).astype(bool)
        n_drop = int((~keep).sum())
        if n_drop:
            print(f"[selection] require_col={require_col}: dropping {n_drop:,} of "
                  f"{len(labels):,} genomes explicitly flagged as having no proteome "
                  f"(nulls retained as unannotated)")
        labels = labels[keep].copy()

    mesophile_pool = labels[labels[mesophile_col].fillna(False)].copy()

    all_extremo = []
    all_outgroups = []
    pair_rows = []
    used_outgroups: set = set()     # global set of outgroup indices ever chosen

    for cls in classes:
        # When reuse_outgroups, each class starts fresh so the same mesophile can
        # pair with extremophiles in multiple classes (reused, counted once in the
        # final deduplicated outgroup set). Otherwise outgroups are used-once.
        used_this_class: set = set() if reuse_outgroups else used_outgroups
        # confidence_levels may be a single tuple (all classes) or a dict of
        # per-class tuples (e.g. thermophile high-only, others high+medium).
        conf_cls = (confidence_levels.get(cls) if isinstance(confidence_levels, dict)
                    else confidence_levels)
        extremo = select_extremophiles(
            labels, cls, max_per_lineage=max_per_lineage, lineage_rank=lineage_rank,
            max_total=max_total_per_class, confidence_levels=conf_cls,
            max_per_sample=max_per_sample, sample_col=sample_col, seed=seed,
        )
        if extremo.empty:
            continue
        all_extremo.append(extremo)

        for eidx, erow in extremo.iterrows():
            oidx, matched_rank = find_outgroup(erow, mesophile_pool, used_this_class)
            if oidx is not None:
                used_this_class.add(oidx)
                used_outgroups.add(oidx)
                orow = mesophile_pool.loc[oidx]
                all_outgroups.append(oidx)
                pair_rows.append({
                    "class": cls,
                    "extremophile_acc": erow[acc_col],
                    "extremophile_confidence": erow.get("final_confidence"),
                    "outgroup_acc": orow[acc_col],
                    "outgroup_confidence": orow.get("final_confidence"),
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
