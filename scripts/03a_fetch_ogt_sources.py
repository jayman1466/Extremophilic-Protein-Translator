#!/usr/bin/env python3
"""Stage 03a: download the four measured-OGT source databases.

Separated from 03b so the network fetch and the merge can fail independently,
and so 03b is a pure function of files on disk.

Only `pooled_measured_ogt.tsv` (the 03b output) is committed to the repo. The
raw sources are re-fetchable and one of them is 46 MB, so they are left
untracked under data/ogt/ and this script rebuilds them.

    python scripts/03a_fetch_ogt_sources.py
    python scripts/03a_fetch_ogt_sources.py --check   # report presence, fetch nothing

PROVENANCE. All four are cited by Toki et al. 2026 (mSystems) or assembled by
OGTFinder; see the 03b docstring for how they combine and what each contributes
to the cold tail.

    TEMPURA    Sato et al. 2020, Microbes Environ 35:ME20074
               doi:10.1264/jsme2.ME20074
    Madin      Madin et al. 2020, Sci Data 7:170
               doi:10.1038/s41597-020-0497-4
    Toki       Toki et al. 2026, mSystems
               doi:10.1128/msystems.00062-26
               NB: 3,131 species, NOT the 8,972 quoted in the paper's methods.
               8,972 is the pre-QC merge; the repo CSV is the post-QC modelling
               set (2,869 bacteria + 262 archaea).
    OGTFinder  Colette et al. 2025, bioRxiv 2025.03.03.640802 (MIT licence)
               Pools BacDive, ThermoBase, aciDB, MediaDB, Lyubetsky et al. 2020.
               Ships Type=='growth' AND Type=='optimum'; only optima are OGT.

NOT AVAILABLE. Sauer & Wang 2015 (Biophys J 109:1420, doi
10.1016/j.bpj.2015.07.026), Toki's dataset I at 11,004 species. PMC4601007 is
not open access and efetch exposes no supplementary material. If it is obtained,
drop it at data/ogt/sauer_wang_ogt.csv and add a loader to 03b -- its value is
corroboration, not coverage, since 57.9% of pooled species currently rest on a
single measurement.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

SOURCES = {
    "tempura.csv": "http://togodb.org/release/tempura.csv",
    "madin_condensed_traits.csv": (
        "https://raw.githubusercontent.com/bacteria-archaea-traits/"
        "bacteria-archaea-traits/master/output/condensed_traits_NCBI.csv"
    ),
    "toki_OGT.csv": (
        "https://raw.githubusercontent.com/tsuchimatsu/OGT_prediction/"
        "main/data/csv/OGT.csv"
    ),
    "ogtfinder_growth_temp.tsv": (
        "https://raw.githubusercontent.com/SC-Git1/OGTFinder/"
        "main/Data/growth_temp_dataset.tsv"
    ),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ogt-dir", default="data/ogt")
    ap.add_argument("--check", action="store_true",
                    help="report which sources are present, download nothing")
    ap.add_argument("--force", action="store_true",
                    help="re-download even if the file already exists")
    args = ap.parse_args(argv)

    out_dir = Path(args.ogt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    for name, url in SOURCES.items():
        dest = out_dir / name
        if args.check:
            size = f"{dest.stat().st_size:,} B" if dest.exists() else "MISSING"
            print(f"{name:34s} {size}")
            continue
        if dest.exists() and not args.force:
            print(f"{name:34s} present ({dest.stat().st_size:,} B), skipping")
            continue
        try:
            with urllib.request.urlopen(url, timeout=300) as fh:
                payload = fh.read()
            dest.write_bytes(payload)
            print(f"{name:34s} fetched {len(payload):,} B")
        except Exception as exc:                      # noqa: BLE001 - report and continue
            print(f"{name:34s} FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
            failures.append(name)

    if failures:
        print(f"\n{len(failures)} source(s) failed: {', '.join(failures)}", file=sys.stderr)
        print("03b will still run on the remainder, but corroboration tiers "
              "(ogt_n_sources) and therefore the `high` tier will differ.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
