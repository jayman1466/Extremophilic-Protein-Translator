#!/usr/bin/env python
"""Stage 03c -- merge the deep-sea MAG set into the combined labels table.

Appends 4,084 metagenome-assembled genomes (Guo et al. 2026, Cell Host & Microbe)
to the r232 representative-genome labels, so they enter phylogeny-controlled
selection alongside GTDB genomes.

WHAT THE MAGs BRING, AND WHAT THEY DO NOT
-----------------------------------------
Measured over all 4,084 under the standard rubric (metadata + GenomeSPOT):

    class              high  medium    low
    thermophile         590     272  1,191
    halophile             0      81      0
    hyperthermophile      0      40    242
    acidophile            0      34      0
    alkaliphile           0       5      0
    psychrophile          2       3  1,376

Thermophile earns `high` the ordinary way: hydrothermal-vent metadata and hot
GenomeSPOT predictions agree. The pH and salinity classes cannot reach `high`
at all -- marine sediment metadata carries no acidic/alkaline/saline vocabulary,
so those calls are prediction-only, which is exactly what `medium` denotes.

The psychrophile column is the important non-contribution. 1,417 MAGs carry
cold-habitat keywords, and 1,376 of them stay at `low`, because:
  * GenomeSPOT will not corroborate cold (2 of 201 MAGs with MEASURED ambient
    <=15 C are predicted <=15 C; mean over-prediction +23.5 C), and
  * the measured-OGT rubric of stage 03b keys on cultivated species names, while
    43% of these MAGs have no species assignment at all.
So this dataset is thermophile-heavy by construction. Do not expect it to fix
the psychrophile sample-size problem; it does not.

WHY IN-SITU TEMPERATURE DOES NOT PROMOTE COLD MAGs
--------------------------------------------------
Tested against the hot end as a control, on the same `insitu_temp_usable` tag:
    ambient >=50 C (n=12) : GenomeSPOT predicts >=50 C for 11/12 (92%)
    ambient <=15 C (n=201): predicts <=15 C for 2/201 (1.0%)
Ambient temperature is corroborative where it is hot and uninformative where it
is cold. Promoting cold MAGs on ambient alone would rest `high` on habitat
evidence measured at a 0.14-0.63% hit rate across four independent populations.
They stay at `low`, which is the honest tier, and remain useful for MLM
pretraining where label noise is tolerable -- not for labelled pair evaluation.

THE PER-SAMPLE CAP
------------------
MAG labels are SAMPLE-level, not genome-level: 4,084 MAGs come from 858 samples,
and the worst single sample contributes 38 thermophile MAGs across 30 distinct
families. Those 38 sail through a per-family cap while resting on ONE
environmental observation. This stage therefore writes `source_sample_id` for
every MAG and leaves it blank for GTDB isolates, which is the column
`select_extremophiles(max_per_sample=...)` keys on. Blank means "its own
sample" and is exempt -- pooling blanks into one bucket would cap all of GTDB.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eptrans.binning import combine_label
from eptrans.gtdb import load_config

RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]
CLASSES = ["thermophile", "hyperthermophile", "psychrophile",
           "acidophile", "alkaliphile", "halophile"]
MAG_PREFIX = "CU_"          # same scheme as other custom genomes


def _threshold_predictions(row: pd.Series, th: dict) -> set[str]:
    """Apply config thresholds to one row's GenomeSPOT predictions."""
    out: set[str] = set()
    t, p, s = row.get("temperature_optimum"), row.get("ph_optimum"), row.get("salinity_optimum")
    if pd.notna(t):
        if t >= th["temperature"]["thermophile_min_opt"]:
            out.add("thermophile")
        if t >= th["temperature"]["hyperthermophile_min_opt"]:
            out.add("hyperthermophile")
        if t <= th["temperature"]["psychrophile_max_opt"]:
            out.add("psychrophile")
    if pd.notna(p):
        if p <= th["ph"]["acidophile_max_opt"]:
            out.add("acidophile")
        if p >= th["ph"]["alkaliphile_min_opt"]:
            out.add("alkaliphile")
    if pd.notna(s) and s >= th["salinity"]["halophile_min_opt"]:
        out.add("halophile")
    return out


def _is_mesophile(row: pd.Series, th: dict) -> bool:
    """Confident mesophile: every axis predicted inside the moderate band."""
    m = th["mesophile"]
    t, p, s = row.get("temperature_optimum"), row.get("ph_optimum"), row.get("salinity_optimum")
    if pd.isna(t) or pd.isna(p) or pd.isna(s):
        return False
    return (m["temp_min_opt"] <= t <= m["temp_max_opt"]
            and m["ph_min_opt"] <= p <= m["ph_max_opt"]
            and s <= m["salinity_max_opt"])


def build_mag_rows(merged: pd.DataFrame, th: dict,
                   id_map: dict | None = None) -> pd.DataFrame:
    """Turn the merged MAG table into label-table rows.

    id_map: {original_mag_id -> assigned CU_CUST_xxx.1 accession} from the ingest
    (mag_ingest/map.tsv). REQUIRED whenever the MAGs have been ingested, because
    the accession the proteins are keyed by is NOT derivable from mag_id.

    THE BUG THIS EXISTS TO PREVENT. The original code wrote
    `accession = "CU_" + mag_id`, giving e.g. `CU_10A(CNS0876618)_bin.16`. But the
    ingest assigns sequential accessions, so the proteome file, the FASTA headers
    and every SignalP prediction use `CU_CUST_000000001.1`. Those two accession
    spaces share NO key, and assign_labels joins on bare_accession equality --
    so every MAG protein would silently fail to get a label and be dropped from
    the dataset with no error raised. 1,445 of 4,084 mag_id values (35.4%) also
    contain parentheses, which is what made the renaming necessary in the first
    place (GTDB-Tk's own genome-ID validator rejects them).
    """
    out = pd.DataFrame(index=merged.index)
    mid = merged["mag_id"].astype(str)
    if id_map:
        mapped = mid.map(id_map)
        out["accession"] = mapped.where(mapped.notna(), MAG_PREFIX + mid)
        out["has_proteome"] = mapped.notna()
    else:
        out["accession"] = MAG_PREFIX + mid
        out["has_proteome"] = False
    out["mag_id"] = merged["mag_id"]
    out["source_dataset"] = "deepsea_dsgc_2026"
    out["source_sample_id"] = merged["sample_id"]       # drives max_per_sample
    out["gtdb_representative"] = "f"                    # MAGs are not GTDB reps

    for r in RANKS:
        out[r] = merged[r]

    for c in ["temperature_optimum", "temperature_min", "ph_optimum", "salinity_optimum"]:
        out[c] = merged[c]
    out["insitu_temperature_c"] = merged["temperature_c"]
    out["insitu_temp_usable"] = merged["insitu_temp_usable"]
    out["isolation_source"] = merged["isolation_source"]
    out["ecosystem"] = merged["ecosystem"]
    out["depth_m"] = merged["depth_m"]

    meta = merged["metadata_classes"].fillna("").apply(
        lambda s: {t.strip() for t in str(s).split(",") if t.strip()})
    pred = merged.apply(lambda r: _threshold_predictions(r, th), axis=1)
    avail = merged["temperature_optimum"].notna()

    combined = [combine_label(m, p, a) for m, p, a in zip(meta, pred, avail)]
    out["final_label"] = [c[0] for c in combined]
    out["final_confidence"] = [c[1] for c in combined]
    for cls in CLASSES:
        out[f"final_{cls}"] = [cls in c[0].split(";") for c in combined]
    out["confident_mesophile"] = merged.apply(lambda r: _is_mesophile(r, th), axis=1)
    # Cold MAGs cannot reach the measured-OGT rubric (uncultivated, no species
    # match), so this column is explicitly `none` rather than silently absent.
    out["psy_conf_measured"] = "none"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=None, help="input combined-labels parquet")
    ap.add_argument("--mags", required=True, help="merged deep-sea MAG TSV (taxonomy + GenomeSPOT)")
    ap.add_argument("--out", default=None, help="output parquet")
    ap.add_argument("--stats", default=None, help="optional JSON stats path")
    ap.add_argument("--id-map", default=None,
                    help="mag_ingest/map.tsv (original_mag_id<TAB>assigned CU_CUST accession). "
                         "REQUIRED once the MAGs are ingested: the accession the proteins "
                         "are keyed by is not derivable from mag_id, and without it every "
                         "MAG protein silently fails the label join.")
    ap.add_argument("--require-proteome", action="store_true",
                    help="keep only MAGs present in --id-map, i.e. those with an "
                         "actual proteome on disk. Without it, un-ingested MAGs are "
                         "retained as label-only rows (usable for counting, never "
                         "selectable as a training genome).")
    args = ap.parse_args()

    cfg = load_config()
    th = cfg["thresholds"]
    repo = Path(__file__).resolve().parents[1]
    labels_path = Path(args.labels) if args.labels else repo / "results/combined_labels_r232.parquet"
    out_path = Path(args.out) if args.out else repo / "results/combined_labels_r232_plus_deepsea.parquet"

    labels = pd.read_parquet(labels_path)
    mags = pd.read_csv(args.mags, sep="\t", low_memory=False)

    # Idempotence: drop any previously appended rows from this dataset so a
    # re-run replaces them instead of duplicating (same discipline as 03b).
    if "source_dataset" in labels.columns:
        before = len(labels)
        labels = labels[labels["source_dataset"] != "deepsea_dsgc_2026"].copy()
        if before != len(labels):
            print(f"[03c] dropped {before - len(labels):,} previously appended MAG rows")

    id_map = None
    if args.id_map:
        m = pd.read_csv(args.id_map, sep="\t", header=None,
                        names=["orig", "assigned"], dtype=str)
        if m["orig"].duplicated().any() or m["assigned"].duplicated().any():
            raise SystemExit(f"[03c] --id-map is not injective: "
                             f"{int(m['orig'].duplicated().sum())} duplicate originals, "
                             f"{int(m['assigned'].duplicated().sum())} duplicate assigned")
        id_map = dict(zip(m["orig"], m["assigned"]))
        print(f"[03c] id map: {len(id_map):,} ingested MAGs "
              f"(e.g. {m['orig'].iloc[0]} -> {m['assigned'].iloc[0]})")
    else:
        print("[03c] WARNING: no --id-map. Accessions will be built from mag_id, which "
              "does NOT match the ingested proteome accessions; MAG proteins will not "
              "join to labels.")

    mag_rows = build_mag_rows(mags, th, id_map=id_map)
    n_prot = int(mag_rows["has_proteome"].sum())
    print(f"[03c] MAG rows {len(mag_rows):,}; with proteome on disk {n_prot:,}; "
          f"label-only {len(mag_rows) - n_prot:,}")
    if args.require_proteome:
        mag_rows = mag_rows[mag_rows["has_proteome"]].copy()
        print(f"[03c] --require-proteome: kept {len(mag_rows):,}")
    if id_map and n_prot == 0:
        raise SystemExit("[03c] id map supplied but matched ZERO mag_id values -- "
                         "the map's first column must be the ORIGINAL mag_id")

    # GTDB genomes are their own sample: leave the column blank, never a shared token.
    for col in ["source_dataset", "source_sample_id", "mag_id",
                "insitu_temperature_c", "insitu_temp_usable", "ecosystem", "depth_m"]:
        if col not in labels.columns:
            labels[col] = np.nan
    if "source_dataset" in labels.columns:
        labels["source_dataset"] = labels["source_dataset"].fillna("gtdb_r232")

    merged = pd.concat([labels, mag_rows], ignore_index=True, sort=False)

    assert merged["accession"].is_unique, "accession collision after merge"
    n_mag = int((merged["source_dataset"] == "deepsea_dsgc_2026").sum())
    assert n_mag == len(mags), f"expected {len(mags)} MAG rows, got {n_mag}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)

    tiers = {}
    for cls in CLASSES:
        sub = mag_rows[mag_rows[f"final_{cls}"]]
        tiers[cls] = sub["final_confidence"].value_counts().to_dict()
    stats = {
        "rows_total": len(merged),
        "rows_gtdb": int((merged["source_dataset"] == "gtdb_r232").sum()),
        "rows_deepsea": n_mag,
        "deepsea_samples": int(mag_rows["source_sample_id"].nunique()),
        "mags_per_sample_max": int(mag_rows["source_sample_id"].value_counts().max()),
        "deepsea_tiers_by_class": tiers,
        "deepsea_confident_mesophile": int(mag_rows["confident_mesophile"].sum()),
        "output": str(out_path),
    }
    print(json.dumps(stats, indent=2))
    if args.stats:
        Path(args.stats).write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
