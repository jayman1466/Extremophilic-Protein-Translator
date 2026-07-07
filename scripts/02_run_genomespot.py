#!/usr/bin/env python3
"""Stage 02 - run GenomeSPOT on a set of genomes and collect predictions.

Given a list of genome accessions (resolved to genome/proteome files via the
GenomeIndex accessors / config'd paths), runs GenomeSPOT on each and writes a
tidy per-genome predictions parquet + TSV.

On biotite this is dispatched as a SLURM array; locally it runs serially. This
script is the serial/local driver; the SLURM wrapper calls the same per-genome
function.

Usage
-----
    python scripts/02_run_genomespot.py \
        --accessions data/pilot_accessions.txt \
        --models /path/to/GenomeSPOT/models \
        --out results/genomespot_predictions.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from eptrans.config import load_config
from eptrans.gtdb import GenomeIndex, bare_accession
from eptrans.genomespot import run_genomespot, results_to_frame


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--accessions", required=True, help="text file, one accession per line")
    ap.add_argument("--models", required=True, help="GenomeSPOT models directory")
    ap.add_argument("--genome-index", default=cfg.get_path("biotite.genome_index"),
                    help="genome_index.tsv (bare_acc -> fna path)")
    ap.add_argument("--workdir", default="results/genomespot_work")
    ap.add_argument("--out", default="results/genomespot_predictions.parquet")
    ap.add_argument("--python-exe", default=None, help="python with genome_spot installed")
    args = ap.parse_args()

    accs = [l.strip() for l in open(args.accessions) if l.strip()]
    idx = GenomeIndex(args.genome_index)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    results = []
    for acc in tqdm(accs, desc="GenomeSPOT"):
        bare = bare_accession(acc)
        try:
            fna = idx.genome_fna(bare)
        except KeyError:
            from eptrans.genomespot import _empty_result
            results.append(_empty_result(bare, ok=False, msg="genome fna not in index"))
            continue
        faa = idx.proteome_faa(acc)
        res = run_genomespot(
            contigs=fna, proteins=faa, models_dir=args.models,
            output_prefix=workdir / bare, genome=bare, python_exe=args.python_exe,
        )
        results.append(res)

    df = results_to_frame(results)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    df.to_csv(out.with_suffix(".tsv"), sep="\t", index=False)
    n_ok = int(df["genomespot_ok"].sum())
    print(f"[02] genomes: {len(df)}  ok: {n_ok}  failed: {len(df)-n_ok}")
    print(f"[02] wrote {out} and {out.with_suffix('.tsv')}")


if __name__ == "__main__":
    main()
