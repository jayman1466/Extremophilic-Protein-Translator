#!/usr/bin/env python3
"""Stage 03 - combine metadata flags + GenomeSPOT predictions into final labels.

For each r232 representative, reconcile the two independent evidence sources:
  * metadata isolation-source flags (stage 01b)
  * GenomeSPOT predicted classes (reconciled/recomputed, stage 02/02b)

into a final extremophile label per class with a confidence tier:
  high   - metadata and prediction agree
  medium - prediction only
  low    - metadata only, or metadata/prediction conflict
  none   - no evidence

Also derives a `confident_mesophile` flag (all predicted optima inside the
mesophile envelope AND no metadata extremophile flag) for outgroup selection.

Outputs
-------
    <out>                    combined labels parquet + TSV
    <fig_counts>             final class counts by confidence tier
    <fig_agree>              metadata vs prediction agreement matrix (per class)

Usage
-----
    python scripts/03_combine_bins.py \
        --flags results/metadata_flags.parquet \
        --predictions results/genomespot_reconciled_r232.tsv \
        --out results/combined_labels.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eptrans.config import load_config
from eptrans.binning import (
    CLASSES, predicted_classes, is_confident_mesophile, combine_label,
)


# Map GenomeSPOT precomputed/recomputed columns -> the three optima we threshold.
PRED_TEMP = ["precomp_temperature_optimum", "temperature_optimum"]
PRED_PH = ["precomp_ph_optimum", "ph_optimum"]
PRED_SAL = ["precomp_salinity_optimum", "salinity_optimum"]


def _first_present(df: pd.DataFrame, cands: list[str]) -> pd.Series:
    for c in cands:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    return pd.Series([np.nan] * len(df), index=df.index)


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flags", required=True, help="metadata flags parquet (stage 01b)")
    ap.add_argument("--predictions", default=None,
                    help="reconciled/predicted GenomeSPOT table (stage 02b); optional")
    ap.add_argument("--acc-col", default="accession")
    ap.add_argument("--bare-join", action="store_true",
                    help="join flags<->predictions on bare accession (strip GB_/RS_ prefix); "
                         "use when predictions carry bare accessions (aggregated GenomeSPOT TSV)")
    ap.add_argument("--out", default="results/combined_labels.parquet")
    ap.add_argument("--fig-counts", default="results/combined_label_counts.png")
    ap.add_argument("--fig-agree", default="results/combined_agreement.png")
    args = ap.parse_args()

    flags = pd.read_parquet(args.flags)
    df = flags.copy()

    # Merge predictions if provided.
    have_pred = False
    if args.predictions:
        pred = (pd.read_parquet(args.predictions) if args.predictions.endswith(".parquet")
                else pd.read_csv(args.predictions, sep="\t"))
        pred.columns = [c.strip() for c in pred.columns]  # tolerate CRLF-trailing header
        keep = [args.acc_col] + [c for c in pred.columns
                                 if c in set(PRED_TEMP + PRED_SAL + PRED_PH)
                                 or c in ("genomespot_reused", "genomespot_match_level", "genome_fna_path")]
        pred = pred[keep].drop_duplicates(args.acc_col)
        if args.bare_join:
            # normalize both sides to bare accession (strips GB_/RS_ + keeps version).
            # The aggregated GenomeSPOT TSV uses bare accessions; metadata flags keep
            # the GTDB prefix. Join on the shared bare form.
            from eptrans.gtdb import bare_accession
            df["_bare"] = df[args.acc_col].map(bare_accession)
            pred = pred.rename(columns={args.acc_col: "_pred_acc"})
            pred["_bare"] = pred["_pred_acc"].map(bare_accession)
            pred = pred.drop(columns=["_pred_acc"])
            df = df.merge(pred, on="_bare", how="left").drop(columns=["_bare"])
        else:
            df = df.merge(pred, on=args.acc_col, how="left")
        have_pred = True

    temp = _first_present(df, PRED_TEMP)
    ph = _first_present(df, PRED_PH)
    sal = _first_present(df, PRED_SAL)
    pred_available = have_pred & (temp.notna() | ph.notna() | sal.notna())

    # Per-genome derivation.
    final_labels, confidences, pred_class_join, meso_flags = [], [], [], []
    for i in range(len(df)):
        meta_c = set(str(df.iloc[i].get("meta_iso_classes", "") or "").split(";")) - {""}
        if pred_available.iloc[i]:
            pc = predicted_classes(temp.iloc[i], ph.iloc[i], sal.iloc[i], cfg=cfg)
            meso = is_confident_mesophile(temp.iloc[i], ph.iloc[i], sal.iloc[i], cfg=cfg)
        else:
            pc = set()
            meso = False
        label, conf = combine_label(meta_c, pc, bool(pred_available.iloc[i]))
        final_labels.append(label)
        confidences.append(conf)
        pred_class_join.append(";".join(sorted(pc)))
        # confident mesophile: predicted mesophile AND no metadata extremophile flag
        meso_flags.append(bool(meso and not meta_c))

    df["pred_classes"] = pred_class_join
    df["final_label"] = final_labels
    df["final_confidence"] = confidences
    df["confident_mesophile"] = meso_flags

    # Per-class boolean membership in the final label.
    for cls in CLASSES:
        df[f"final_{cls}"] = [cls in (lbl.split(";") if lbl else []) for lbl in final_labels]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    df.to_csv(out.with_suffix(".tsv"), sep="\t", index=False)

    # Report.
    conf_counts = pd.Series(confidences).value_counts().to_dict()
    print(f"[03] genomes: {len(df):,} | predictions merged: {have_pred}")
    print(f"[03] confidence tiers: {conf_counts}")
    print(f"[03] confident mesophiles: {int(df['confident_mesophile'].sum()):,}")
    for cls in CLASSES:
        n = int(df[f"final_{cls}"].sum())
        print(f"[03]   final {cls:18s}: {n:,}")
    print(f"[03] wrote {out} and {out.with_suffix('.tsv')}")

    # Figure 1: final class counts stacked by confidence.
    tiers = ["high", "medium", "low"]
    tier_colors = {"high": "#2a9d8f", "medium": "#e9c46a", "low": "#e76f51"}
    x = np.arange(len(CLASSES))
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(CLASSES))
    for tier in tiers:
        vals = [int(((df[f"final_{c}"]) & (df["final_confidence"] == tier)).sum()) for c in CLASSES]
        ax.bar(x, vals, bottom=bottom, label=f"{tier} confidence", color=tier_colors[tier])
        bottom += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=40, ha="right")
    ax.set_ylabel("Genomes with final class label")
    ax.set_title(f"Combined extremophile labels by confidence tier (n={len(df):,})")
    ax.legend()
    for i, tot in enumerate(bottom):
        if tot > 0:
            ax.text(i, tot, f"{int(tot):,}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    figp = Path(args.fig_counts)
    figp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figp, dpi=150)
    print(f"[03] wrote {figp}")

    # Figure 2: metadata vs prediction agreement per class (only if predictions present).
    if have_pred:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        both, meta_only, pred_only = [], [], []
        for cls in CLASSES:
            m = df["meta_iso_classes"].fillna("").str.split(";").apply(lambda xs: cls in xs)
            p = df["pred_classes"].fillna("").str.split(";").apply(lambda xs: cls in xs)
            both.append(int((m & p).sum()))
            meta_only.append(int((m & ~p).sum()))
            pred_only.append(int((~m & p).sum()))
        w = 0.6
        ax2.bar(x, both, w, label="both agree", color="#2a9d8f")
        ax2.bar(x, meta_only, w, bottom=both, label="metadata only", color="#264653")
        ax2.bar(x, pred_only, w, bottom=np.array(both)+np.array(meta_only),
                label="prediction only", color="#e9c46a")
        ax2.set_xticks(x); ax2.set_xticklabels(CLASSES, rotation=40, ha="right")
        ax2.set_ylabel("Genomes"); ax2.set_title("Evidence agreement per class")
        ax2.legend()
        fig2.tight_layout()
        figp2 = Path(args.fig_agree)
        fig2.savefig(figp2, dpi=150)
        print(f"[03] wrote {figp2}")


if __name__ == "__main__":
    main()
