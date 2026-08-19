#!/usr/bin/env python3
"""Compute per-phenotype LigandMPNN bias_AA vectors from the labeled dataset.

For each of the six extremophile classes, we join the labeled_dataset parquet
(which carries {tagged_id, label, is_mesophile}) against a FASTA carrying the
sequences (`{genome}~{protein_id}` headers to match `tagged_id`) and compute:

    bias_AA[phenotype][aa] = log( freq_ext_aa / freq_meso_aa )

where freq_ext_aa is the fraction of that AA across the concatenation of all
sequences whose `label` contains `phenotype`, and freq_meso_aa is the same
over `is_mesophile == True`. Both reference pools use the same clustered
labeled_dataset so they inherit the QC filters (SignalP secreted, clustered
at 50%, split assignments) that trained the mhk32 classifier heads.

This is a static property of the dataset. It only needs to be recomputed if
the dataset changes (new r-release, augment-audit expansion, relabeling) or
a new phenotype is added.

Runtime memory: streams the FASTA (single pass), keeps only per-phenotype
20-dim counters. On biotite, expect ~5 min end-to-end for ~2M sequences.

Emits data/bias_aa_by_phenotype.json. Print a compact rank table for sanity.

Usage
-----
  python scripts/prep_bias_aa.py \\
      --parquet results/labeled_dataset_r232_clustered.parquet \\
      --faa     $PERSIST/secreted_proteins_r232.faa \\
      --out     data/bias_aa_by_phenotype.json

The FASTA is on biotite at $PERSIST/secreted_proteins_r232.faa (~865 MB).
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# LigandMPNN's alphabet order (documented in the LigandMPNN --bias_AA docs).
# We emit the JSON with these keys in this order so downstream consumers can
# convert to a numpy vector without re-sorting.
AA20 = list("ACDEFGHIKLMNPQRSTVWY")
PHENOTYPES = [
    "thermophile",
    "hyperthermophile",
    "psychrophile",
    "acidophile",
    "alkaliphile",
    "halophile",
]


def parse_labels(label_str: str) -> set[str]:
    """Labels are ';'-joined class strings (e.g. 'hyperthermophile;thermophile').

    Return the set of phenotype tokens present.
    """
    if not isinstance(label_str, str) or not label_str:
        return set()
    return {tok.strip() for tok in label_str.split(";") if tok.strip()}


def _open_faa(path: str):
    """Open a FASTA (plain or .gz) as text."""
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"))
    return open(path, "r")


def iter_fasta(path: str):
    """Yield (header_id, sequence) pairs. header_id is the token AFTER '>' up to
    the first whitespace. Sequence is uppercase A-Z only (X and gaps dropped)."""
    header = None
    buf: list[str] = []
    with _open_faa(path) as fh:
        for line in fh:
            if not line:
                continue
            if line[0] == ">":
                if header is not None:
                    yield header, "".join(buf)
                # Header id is everything up to whitespace, minus the '>'.
                header = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip().upper())
        if header is not None:
            yield header, "".join(buf)


def count_composition(seq: str, counter: dict[str, int]) -> int:
    """Add each residue in seq to counter[aa]. Return the number of counted AAs
    (excluding X, B, Z, J, U, O, *, and other non-canonical)."""
    n = 0
    for c in seq:
        if c in counter:  # only canonical 20 are pre-seeded
            counter[c] += 1
            n += 1
    return n


def compute_bias(
    parquet: str,
    faa: str,
    pseudocount: float = 1.0,
    verbose: bool = True,
) -> dict:
    """Two passes conceptually, one pass in practice: build id -> label set map
    from the parquet in memory (light: ~2M rows x 3 cols), then stream FASTA."""
    if verbose:
        print(f"[bias_aa] reading {parquet}", flush=True)
    df = pd.read_parquet(parquet, columns=["tagged_id", "label", "is_mesophile"])
    if verbose:
        print(f"[bias_aa]   rows={len(df):,}", flush=True)

    # Map tagged_id -> (phenotype_set, is_meso). Duplicates would be a bug
    # upstream; if present, first wins.
    id_map: dict[str, tuple[set[str], bool]] = {}
    for tid, lab, meso in zip(df["tagged_id"], df["label"], df["is_mesophile"]):
        if tid in id_map:
            continue
        id_map[tid] = (parse_labels(lab), bool(meso))
    if verbose:
        print(f"[bias_aa]   distinct tagged_ids={len(id_map):,}", flush=True)

    # Per-phenotype counters, + one for mesophiles.
    counters: dict[str, dict[str, int]] = {
        p: {aa: 0 for aa in AA20} for p in PHENOTYPES
    }
    counters["mesophile"] = {aa: 0 for aa in AA20}
    n_prot: dict[str, int] = {p: 0 for p in PHENOTYPES}
    n_prot["mesophile"] = 0
    n_aa: dict[str, int] = {p: 0 for p in PHENOTYPES}
    n_aa["mesophile"] = 0

    matched = 0
    unmatched = 0
    seen = 0
    if verbose:
        print(f"[bias_aa] streaming {faa}", flush=True)

    for header, seq in iter_fasta(faa):
        seen += 1
        entry = id_map.get(header)
        if entry is None:
            unmatched += 1
            if verbose and unmatched <= 5:
                print(f"[bias_aa]   unmatched header (first {unmatched}): {header}", flush=True)
            continue
        phenos, is_meso = entry
        if not phenos and not is_meso:
            unmatched += 1
            continue
        matched += 1
        # Accumulate one composition histogram, then add it to each pool this
        # sequence belongs to. (One sequence, N labels → counted in N pools.)
        local: dict[str, int] = {aa: 0 for aa in AA20}
        local_total = count_composition(seq, local)
        for p in phenos:
            if p not in counters:
                continue  # ignore non-target labels
            for aa in AA20:
                counters[p][aa] += local[aa]
            n_prot[p] += 1
            n_aa[p] += local_total
        if is_meso:
            for aa in AA20:
                counters["mesophile"][aa] += local[aa]
            n_prot["mesophile"] += 1
            n_aa["mesophile"] += local_total
        if verbose and seen % 200_000 == 0:
            print(f"[bias_aa]   seen={seen:,} matched={matched:,} unmatched={unmatched:,}", flush=True)

    if verbose:
        print(f"[bias_aa] done streaming: seen={seen:,} matched={matched:,} unmatched={unmatched:,}", flush=True)
        for p in list(counters):
            print(f"[bias_aa]   {p:20s} n_prot={n_prot[p]:>9,} n_aa={n_aa[p]:>13,}", flush=True)

    # Frequencies (with pseudocount) and log-ratios.
    def freqs(pool: str) -> dict[str, float]:
        total = sum(counters[pool].values()) + pseudocount * len(AA20)
        return {aa: (counters[pool][aa] + pseudocount) / total for aa in AA20}

    freq_meso = freqs("mesophile")
    bias: dict[str, dict[str, float]] = {}
    for p in PHENOTYPES:
        if n_aa[p] == 0:
            print(f"[bias_aa]   WARN: {p} has zero counted AAs; emitting zeros", flush=True)
            bias[p] = {aa: 0.0 for aa in AA20}
            continue
        f_p = freqs(p)
        bias[p] = {aa: math.log(f_p[aa] / freq_meso[aa]) for aa in AA20}

    out = {
        "source_parquet": os.path.abspath(parquet),
        "source_faa": os.path.abspath(faa),
        "computed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reference": "extremophile_vs_all_mesophile",
        "pseudocount": pseudocount,
        "alphabet": AA20,
        "n_proteins": n_prot,
        "n_amino_acids": n_aa,
        "faa_seen": seen,
        "faa_matched": matched,
        "faa_unmatched": unmatched,
        "bias_AA": bias,
    }
    return out


def format_ranks(out: dict) -> str:
    """Return a compact per-phenotype top/bottom-5 report for eyeball QC."""
    lines = []
    for p in PHENOTYPES:
        vec = out["bias_AA"][p]
        ranked = sorted(vec.items(), key=lambda kv: kv[1])
        bottom = ranked[:5]
        top = ranked[-5:][::-1]
        top_s = ", ".join(f"{aa}{v:+.2f}" for aa, v in top)
        bot_s = ", ".join(f"{aa}{v:+.2f}" for aa, v in bottom)
        lines.append(f"  {p:18s}  ↑ {top_s}   ↓ {bot_s}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--parquet", required=True, help="labeled_dataset_r232_clustered.parquet")
    ap.add_argument("--faa", required=True, help="secretome FASTA (plain or .gz). Headers must match `tagged_id` in the parquet.")
    ap.add_argument("--out", required=True, help="output JSON path (typically data/bias_aa_by_phenotype.json)")
    ap.add_argument("--pseudocount", type=float, default=1.0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    out = compute_bias(
        args.parquet, args.faa, pseudocount=args.pseudocount, verbose=not args.quiet
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[bias_aa] wrote {args.out}")
    print("[bias_aa] per-phenotype top-5/bottom-5 log(freq_ext/freq_meso):")
    print(format_ranks(out))


if __name__ == "__main__":
    sys.exit(main())
