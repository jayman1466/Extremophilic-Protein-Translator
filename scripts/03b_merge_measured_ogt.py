#!/usr/bin/env python3
"""Stage 03b: pool measured optimal growth temperatures and derive psychrophile tiers.

WHY THIS STAGE EXISTS
---------------------
GenomeSPOT cannot label psychrophiles. Measured across four independent
populations of cold-habitat genomes, its predicted temperature optimum reaches
<=15 C for well under 1% of them:

    cryo-source r232 (permafrost/glacier/ice)  n=1,452   0.41%
    hadal/abyssal/deep-sea sediment r232       n=  639   0.63%
    TEMPURA measured-cold r232                 n=  443   0.45%
    deep-sea MAGs (this project)               n=1,416   0.14%

The mechanism is regression shrinkage, not sampling: predictions regress onto
the training mean with slope 0.846 and a fixed point at 33.0 C, so a measured
5 C organism is predicted at 9.3 C and the whole cold tail collapses toward
ambient. GenomeSPOT's own authors report RMSE 14 C below 15 C against a
mean-predictor baseline of 12.46 -- worse than guessing. Two other groups
reproduce the failure with different features and different training sets
(Sauer & Wang 2019; Toki et al. 2026), and OGTFinder 2025 reports "poor fit for
psychrophiles" with 58 cold examples out of 6,401 modelled prokaryotes.

So for the cold class we substitute MEASURED optima for predicted ones. This
stage assembles them and writes a rubric that never consults GenomeSPOT.

SOURCES (four, pooled by normalised binomial species name)
----------------------------------------------------------
    TEMPURA    Sato et al. 2020, Microbes Environ 35:ME20074       8,638 species
    Madin      Madin et al. 2020, Sci Data 7:170                  11,629
    Toki       Toki et al. 2026, mSystems, OGT.csv                 3,131
    OGTFinder  Colette et al. 2025 (bioRxiv), Type=='optimum'      3,168
                 -- itself pooling BacDive, ThermoBase, aciDB,
                    MediaDB, Lyubetsky et al. 2020

Each contributes cold species the others lack (uniquely <=15 C: 30 / 37 / 14 /
12), so none is redundant. Pooled: 16,497 species, of which 9,444 join GTDB
r232 representatives.

NOT ADDED: Sauer & Wang 2015 (Biophys J 109:1420), Toki's dataset I at 11,004
species. PMC4601007 is not open access and efetch exposes no supplementary
material. Its value would be corroboration rather than coverage -- 57.9% of
pooled species rest on a single measurement -- since marginal cold yield per
added source has been falling (+47, +15, +11 species at <=15 C).

JOIN CAVEAT
-----------
The join is on SPECIES NAME, not accession. Measured for TEMPURA: 13 genomes
match by accession versus 4,920 by name. Name joins are release-sensitive
(Toki keyed to GTDB r207, we are on r232) and 86.3% of r232 species carry
placeholder names, so unmatched counts are REPORTED, never silently dropped.

WHAT ABOUT Tmin?
----------------
An earlier design used "Tmin <= 4 AND Topt <= 20" as a second route into the
class. Measured against the pooled data, that rule earns no independent place:

  * As an INDEPENDENT criterion it is wrong. Tmin <= 4 holds for 19.5% of all
    r232 genomes with a measured Tmin, and for 17.1% of clear mesophiles
    (Topt 25-40 C). Dropping the Topt ceiling admits 849 genomes at median
    Topt 27.5 C. Tmin <= 4 marks EURYTHERMY -- tolerates cold -- not a cold
    optimum, so those are not psychrophiles.
  * PAIRED with a Topt ceiling it is nearly redundant: 149 genomes pass
    "Tmin<=4 AND Topt<=20" versus 130 passing "Topt<20" alone, and all 41 of
    the difference sit at exactly Topt 20.0. That gap was an inclusive-versus-
    exclusive boundary artifact of the OGT rule, not information from Tmin.
  * Coverage is TEMPURA-only, 52.4% of pooled species, so any rule requiring
    Tmin silently halves the eligible set.

Tmin is therefore RETAINED but DEMOTED, in one narrow role: corroboration at
the 15-20 C boundary, where Tmin <= 0 (growth at or below freezing) promotes
medium to high. That is a claim about a measurement the OGT bands genuinely
cannot make, and it recovers organisms like Cryobacterium arcticum and
Tomitella biformata (Tmin -6 and -5 C, Topt exactly 20.0). It promotes 20
genomes. It never admits a genome on its own.

The boundary artifact it exposed is fixed here: the second band is Topt <= 20,
not < 20. 63 genomes sit at exactly 20.0 -- Cryobacterium, Psychromonas,
Paenisporosarcina, Aequorivita -- and excluding them on a strict inequality was
an error rather than a judgement.

OUTPUTS
-------
    data/ogt/pooled_measured_ogt.tsv          species-keyed reference (checked in)
    results/combined_labels_r232.parquet      genome-keyed, gains ogt_* columns
    results/measured_ogt_merge.stats.json     join diagnostics

Deliberately NOT written to work/genome_index.tsv: that file is a headerless
two-column accession->path resolver consumed by shell loops (awk lookups in the
ANI and GenomeSPOT jobs). Adding columns would break every consumer, and it is
keyed by genome while this data is keyed by species.

Usage:
    python scripts/03b_merge_measured_ogt.py --config config/config.yaml
    python scripts/03b_merge_measured_ogt.py --dry-run     # report joins, write nothing
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- constants

#: Exact incubation temperatures excluded as laboratory convention rather than
#: measurement. Toki et al. observed these four dominate every OGT database
#: because they are default incubator settings; treating them as measured optima
#: would inject a mesophile spike at 25/28/30/37 C.
CONVENTIONAL_TEMPS = {25.0, 28.0, 30.0, 37.0}

#: Psychrophile threshold (C). 15 is the project-wide definition; note Toki et
#: al. use <20 C, so their reported recall figures are not directly comparable.
STRICT_C = 15.0
#: Second band, INCLUSIVE -- see "boundary artifact" in the module docstring.
LENIENT_C = 20.0
#: Psychrotolerant band: only reached with corroborating cold habitat metadata.
TOLERANT_C = 25.0
#: Tmin at or below freezing, the only role Tmin retains.
TMIN_FREEZING_C = 0.0

SOURCE_COLS = ["ogt_tempura", "ogt_madin", "ogt_toki", "ogt_ogtfinder"]


# ---------------------------------------------------------------- helpers

def normalise_species(name: object) -> str | None:
    """Reduce a strain/species string to a lowercase binomial for joining.

    Drops the "Candidatus " prefix and any strain suffix, keeping the first two
    whitespace-separated tokens. Returns None for anything that cannot yield a
    binomial, so callers can count unmatched rows instead of joining on junk.
    """
    if not isinstance(name, str):
        return None
    cleaned = re.sub(r"\s+", " ", name.strip())
    cleaned = re.sub(r"^Candidatus\s+", "", cleaned, flags=re.IGNORECASE)
    parts = cleaned.split()
    return " ".join(parts[:2]).lower() if len(parts) >= 2 else None


def species_from_gtdb_taxonomy(taxonomy: object) -> str | None:
    """Pull the s__ field out of a GTDB taxonomy string and normalise it."""
    if not isinstance(taxonomy, str):
        return None
    match = re.search(r"s__([^;]+)", taxonomy)
    return normalise_species(match.group(1)) if match else None


def drop_conventional(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    """Remove rows whose temperature is exactly a conventional incubator setting."""
    return frame[~frame[col].isin(CONVENTIONAL_TEMPS)]


def classify(row: pd.Series) -> str:
    """Assign a psychrophile confidence tier. GenomeSPOT is never consulted.

    high    measured <=15 C with either a cold habitat or >=2 independent sources
            measured <=20 C with a cold habitat, or with Tmin <=0 C
    medium  measured <=15 C from a single source
            measured <=20 C with no corroboration
            measured <25 C with a cold habitat (psychrotolerant)
    low     cold habitat only, no measurement at all
    none    measured >=25 C (this OVERRIDES cold-sounding metadata), or nothing

    The >=25 C override matters: 61 r232 genomes have cold isolation sources but
    a measured optimum at or above 25 C, and are currently eligible for
    selection.
    """
    ogt = row.get("ogt_measured")
    n_src = row.get("ogt_n_sources")
    tmin = row.get("tmin_measured")
    cold_habitat = bool(row.get("meta_cold", False))

    if pd.isna(ogt):
        return "low" if cold_habitat else "none"

    if ogt <= STRICT_C:
        if cold_habitat:
            return "high"
        return "high" if (pd.notna(n_src) and n_src >= 2) else "medium"

    if ogt <= LENIENT_C:
        if cold_habitat:
            return "high"
        if pd.notna(tmin) and tmin <= TMIN_FREEZING_C:
            return "high"
        return "medium"

    if ogt < TOLERANT_C:
        return "medium" if cold_habitat else "none"

    return "none"


# ---------------------------------------------------------------- loaders

def load_tempura(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw["sp_norm"] = raw["genus_and_species"].map(normalise_species)
    raw = raw.dropna(subset=["sp_norm"])
    raw = drop_conventional(raw, "Topt_ave")
    return (raw.groupby("sp_norm")
               .agg(ogt_tempura=("Topt_ave", "median"),
                    tmin_measured=("Tmin", "median"))
               .reset_index())


def load_madin(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    raw["sp_norm"] = raw["species"].map(normalise_species)
    raw = raw.dropna(subset=["sp_norm", "growth_tmp"])
    raw = drop_conventional(raw, "growth_tmp")
    return (raw.groupby("sp_norm").growth_tmp.median()
               .rename("ogt_madin").reset_index())


def load_toki(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw["sp_norm"] = raw["species"].map(normalise_species)
    raw = raw.dropna(subset=["sp_norm", "OGT"])
    return (raw.groupby("sp_norm").OGT.median()
               .rename("ogt_toki").reset_index())


def load_ogtfinder(path: Path) -> pd.DataFrame:
    """OGTFinder ships growth AND optimum rows; only optima are OGT.

    Conflating cultivation temperature with an optimum is exactly the error the
    conventional-temperature exclusion guards against, so filter on Type first.
    TEMPURA rows are dropped to keep the pooled sources independent -- OGTFinder
    redistributes TEMPURA verbatim, and double-counting it would make a single
    measurement look corroborated.
    """
    raw = pd.read_csv(path, sep="\t", low_memory=False)
    raw["Temp"] = pd.to_numeric(raw["Temp"], errors="coerce")
    raw = raw[(raw["Type"] == "optimum") & raw["Temp"].notna()]
    raw = raw[raw["Source"].str.lower() != "tempura"]
    raw["sp_norm"] = raw["species"].map(normalise_species)
    raw = raw.dropna(subset=["sp_norm"])
    return (raw.groupby("sp_norm")
               .agg(ogt_ogtfinder=("Temp", "median"),
                    ogt_ogtfinder_sources=("Source", lambda s: ";".join(sorted(set(s)))))
               .reset_index())


# ---------------------------------------------------------------- pooling

def pool_sources(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Outer-join per-source tables and summarise agreement across them."""
    pooled = frames[0]
    for nxt in frames[1:]:
        pooled = pooled.merge(nxt, on="sp_norm", how="outer")

    present = [c for c in SOURCE_COLS if c in pooled.columns]
    pooled["ogt_n_sources"] = pooled[present].notna().sum(axis=1)
    pooled["ogt_measured"] = pooled[present].mean(axis=1)
    pooled["ogt_spread"] = pooled[present].max(axis=1) - pooled[present].min(axis=1)
    return pooled.sort_values("sp_norm").reset_index(drop=True)


def merge_into_labels(labels: pd.DataFrame, pooled: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Attach pooled OGT columns to the genome-keyed label table by species name."""
    if "gtdb_taxonomy" not in labels.columns:
        raise KeyError("label table lacks gtdb_taxonomy; cannot derive species key")

    labels = labels.copy()

    # Idempotence: this stage writes back into its own input, so a second run
    # would collide on the columns the first run added and pandas would silently
    # suffix them to ogt_measured_x / ogt_measured_y. Drop any prior output
    # columns first so re-running refreshes them instead of duplicating.
    stale = [c for c in (*SOURCE_COLS, "ogt_measured", "ogt_n_sources", "ogt_spread",
                         "tmin_measured", "psy_conf_measured", "meta_cold", "sp_norm")
             if c in labels.columns]
    if stale:
        print(f"refreshing {len(stale)} existing column(s) from a previous run: "
              + ", ".join(stale))
        labels = labels.drop(columns=stale)

    labels["sp_norm"] = labels["gtdb_taxonomy"].map(species_from_gtdb_taxonomy)

    carry = ["sp_norm", "ogt_measured", "ogt_n_sources", "ogt_spread",
             "tmin_measured", *[c for c in SOURCE_COLS if c in pooled.columns]]
    merged = labels.merge(pooled[carry], on="sp_norm", how="left")

    if len(merged) != len(labels):
        raise AssertionError(
            f"merge changed row count {len(labels)} -> {len(merged)}; "
            "the species key is not unique in the pooled table"
        )

    psy_cols = [c for c in ("meta_iso_psychrophile", "meta_org_psychrophile")
                if c in merged.columns]
    merged["meta_cold"] = (merged[psy_cols].fillna(False).any(axis=1)
                           if psy_cols else False)
    merged["psy_conf_measured"] = merged.apply(classify, axis=1)

    tiers = merged.psy_conf_measured.value_counts()
    named = merged.sp_norm.notna()
    stats = {
        "genomes_total": int(len(merged)),
        "genomes_with_species_key": int(named.sum()),
        "genomes_without_species_key": int((~named).sum()),
        "genomes_with_measured_ogt": int(merged.ogt_measured.notna().sum()),
        "pooled_species_total": int(len(pooled)),
        "pooled_species_matched_to_r232": int(
            pooled.sp_norm.isin(merged.loc[named, "sp_norm"]).sum()),
        "pooled_species_unmatched": int(
            (~pooled.sp_norm.isin(merged.loc[named, "sp_norm"])).sum()),
        "tiers": {k: int(tiers.get(k, 0)) for k in ("high", "medium", "low", "none")},
        "genomes_le_15C": int((merged.ogt_measured <= STRICT_C).sum()),
        "genomes_le_20C": int((merged.ogt_measured <= LENIENT_C).sum()),
        "promoted_by_tmin": int(
            ((merged.ogt_measured > STRICT_C) & (merged.ogt_measured <= LENIENT_C)
             & ~merged.meta_cold & (merged.tmin_measured <= TMIN_FREEZING_C)).sum()),
        "measured_warm_overriding_cold_metadata": int(
            ((merged.ogt_measured >= TOLERANT_C) & merged.meta_cold).sum()),
        "multi_source_species": int((pooled.ogt_n_sources >= 2).sum()),
        "multi_source_median_spread_C": float(
            pooled.loc[pooled.ogt_n_sources >= 2, "ogt_spread"].median()),
        "multi_source_disagree_over_10C": int(
            (pooled.loc[pooled.ogt_n_sources >= 2, "ogt_spread"] > 10).sum()),
    }
    return merged.drop(columns=["sp_norm"]), stats


# ---------------------------------------------------------------- entrypoint

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--ogt-dir", default="data/ogt",
                    help="directory holding the four downloaded source files")
    ap.add_argument("--labels", default="results/combined_labels_r232.parquet")
    ap.add_argument("--out-labels", default=None,
                    help="defaults to --labels (in-place update)")
    ap.add_argument("--out-pooled", default="data/ogt/pooled_measured_ogt.tsv")
    ap.add_argument("--out-stats", default="results/measured_ogt_merge.stats.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="report join diagnostics and write nothing")
    args = ap.parse_args(argv)

    ogt_dir = Path(args.ogt_dir)
    wanted = {
        "tempura": (ogt_dir / "tempura.csv", load_tempura),
        "madin": (ogt_dir / "madin_condensed_traits.csv", load_madin),
        "toki": (ogt_dir / "toki_OGT.csv", load_toki),
        "ogtfinder": (ogt_dir / "ogtfinder_growth_temp.tsv", load_ogtfinder),
    }

    frames, loaded, missing = [], [], []
    for name, (path, loader) in wanted.items():
        if path.exists():
            frame = loader(path)
            frames.append(frame)
            loaded.append(f"{name}={len(frame)}")
        else:
            missing.append(f"{name} ({path})")

    if not frames:
        print("ERROR no source files found under " + str(ogt_dir), file=sys.stderr)
        print("expected: " + ", ".join(p.name for p, _ in wanted.values()), file=sys.stderr)
        return 2
    if missing:
        # Proceeding on a subset is legitimate, but it changes ogt_n_sources and
        # therefore which genomes reach `high`. Say so loudly.
        print("WARNING missing sources, corroboration tiers will differ: "
              + "; ".join(missing), file=sys.stderr)

    print("loaded species per source: " + ", ".join(loaded))

    pooled = pool_sources(frames)
    print(f"pooled species: {len(pooled)}")

    labels = pd.read_parquet(args.labels)
    merged, stats = merge_into_labels(labels, pooled)
    stats["sources_loaded"] = loaded
    stats["sources_missing"] = missing

    print(json.dumps(stats, indent=2))

    if args.dry_run:
        print("dry run: nothing written")
        return 0

    out_pooled = Path(args.out_pooled)
    out_pooled.parent.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(out_pooled, sep="\t", index=False)

    out_labels = Path(args.out_labels or args.labels)
    merged.to_parquet(out_labels, index=False)

    out_stats = Path(args.out_stats)
    out_stats.parent.mkdir(parents=True, exist_ok=True)
    out_stats.write_text(json.dumps(stats, indent=2) + "\n")

    print(f"wrote {out_pooled} ({len(pooled)} species)")
    print(f"wrote {out_labels} ({len(merged)} genomes, {merged.shape[1]} columns)")
    print(f"wrote {out_stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
