#!/usr/bin/env python
"""Constrain a phenotype's training set + outgroup to proteins with an EC designation.

Motivation. Whole-proteome scope was adopted for psychrophile/hyperthermophile to
gain data volume, but most proteins in a proteome carry no cold-adaptation signal:
cold adaptation acts on ENZYMES whose catalytic rate is temperature-limited, via
increased local flexibility around the active site. Structural proteins,
transporters and small ORFs inherit the genome-level label while expressing none
of the phenotype -- protein-level LABEL NOISE that caps achievable AUC no matter
the model. Restricting to EC-designated proteins tests that directly, and has the
side benefit of comparing like with like: an EC number is a function identity, so
an ext protein and its outgroup partner sharing an EC are near-orthologous by
function, not merely by sequence cluster.

Method. KofamScan (exec_annotation) assigns KEGG Orthology numbers by HMM against
the KOfam profile set; KOfam's ko_list maps 10,736 KOs to explicit [EC:...] terms.
So KO assignment -> EC designation. This route was chosen after checking the
alternative: the Swiss-Prot FASTA on this host carries NO EC in its headers (0 of
572,970), because EC lives in the .dat records, which are not present.

Emits a per-protein table (tagged_id, ko, ec, ec_class) plus a filtered pair table
keeping only pairs where BOTH members are EC-designated, and -- with
--require-same-ec -- only pairs whose members share an EC at the requested depth
(default 3 = sub-subclass, e.g. 1.1.1, which is the level at which chemistry is
shared but substrate specificity may differ).

Two stages, because the HMM search is the expensive part:
  --mode annotate  : write FASTA shards to scan, or parse exec_annotation output
  --mode filter    : join KO->EC, filter pairs, emit stats
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_EC_RE = re.compile(r"\[EC:([^\]]+)\]")


def load_ko2ec(ko_list_path: str) -> dict:
    """Parse KOfam ko_list -> {KO: [ec, ...]}. Only KOs with an explicit [EC:...]."""
    ko2ec: dict = {}
    with open(ko_list_path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        try:
            i_ko = header.index("knum")
            i_def = header.index("definition")
        except ValueError:
            i_ko, i_def = 0, len(header) - 1
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) <= max(i_ko, i_def):
                continue
            m = _EC_RE.search(f[i_def])
            if m:
                ko2ec[f[i_ko]] = m.group(1).split()
    return ko2ec


def ec_prefix(ec: str, depth: int) -> str:
    parts = [p for p in ec.split(".") if p != ""]
    return ".".join(parts[:depth])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["annotate", "filter"], required=True)
    ap.add_argument("--labeled")
    ap.add_argument("--pairs")
    ap.add_argument("--fasta", help="corpus FASTA (annotate mode: source of seqs)")
    ap.add_argument("--phenotypes", nargs="+", default=["psychrophile"])
    ap.add_argument("--ko-list",
                    default="/shared/db/kegg/kofam/latest/metadata/ko_list")
    ap.add_argument("--kofam-out", help="filter mode: exec_annotation detail-tsv output(s)",
                    nargs="*")
    ap.add_argument("--ec-depth", type=int, default=3)
    ap.add_argument("--require-same-ec", action="store_true",
                    help="keep only pairs whose members share an EC prefix at --ec-depth")
    ap.add_argument("--nshards", type=int, default=8)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- annotate
    if args.mode == "annotate":
        # Only the proteins that can possibly matter: both members of the target
        # phenotypes' pairs. Scanning the whole 4.16M-protein psychrophile corpus
        # against 26,899 profiles is not affordable; the pair set is ~thousands.
        pr = pd.read_csv(args.pairs, sep="\t", dtype=str)
        pr = pr[pr["class"].isin(args.phenotypes)]
        want = sorted(set(pr["ext_id"].dropna()) | set(pr["outgroup_id"].dropna()))
        wset = set(want)
        print(f"[16] {len(pr):,} pairs -> {len(want):,} distinct proteins to annotate",
              flush=True)
        (out / "targets.txt").write_text("\n".join(want) + "\n")

        # stream the corpus FASTA once, emit only wanted records, round-robin sharded
        handles = [open(out / f"ec_shard{i}.faa", "w") for i in range(args.nshards)]
        kept = 0; keep = False; h = None
        with open(args.fasta) as fh:
            for line in fh:
                if line.startswith(">"):
                    tid = line[1:].split()[0]
                    keep = tid in wset
                    if keep:
                        h = handles[kept % args.nshards]
                        kept += 1
                        h.write(line)
                elif keep:
                    h.write(line)
        for x in handles:
            x.close()
        print(f"[16] wrote {kept:,} sequences across {args.nshards} shards "
              f"(missing {len(want)-kept:,} not found in FASTA)", flush=True)
        (out / "annotate_stats.json").write_text(json.dumps(
            {"n_pairs": int(len(pr)), "n_targets": len(want),
             "n_written": kept, "n_missing": len(want) - kept}, indent=1))
        return

    # ------------------------------------------------------------------ filter
    ko2ec = load_ko2ec(args.ko_list)
    print(f"[16] ko_list: {len(ko2ec):,} KOs carry an explicit [EC:...]", flush=True)

    # exec_annotation detail-tsv: '*' marks above-threshold; cols
    # (mark, gene name, KO, thrshld, score, E-value, "KO definition")
    rows = []
    for p in (args.kofam_out or []):
        with open(p) as fh:
            for line in fh:
                if not line.startswith("*"):
                    continue          # keep only above-threshold assignments
                f = [x.strip() for x in line.rstrip("\n").split("\t")]
                if len(f) < 3:
                    continue
                rows.append({"tagged_id": f[1], "ko": f[2]})
    hits = pd.DataFrame(rows).drop_duplicates()
    if hits.empty:
        raise SystemExit("[16] FATAL: no above-threshold KO assignments parsed")
    hits["ec"] = hits["ko"].map(lambda k: ";".join(ko2ec.get(k, [])))
    ann = hits[hits["ec"] != ""].copy()
    ann["ec_class"] = ann["ec"].map(
        lambda s: ";".join(sorted({ec_prefix(e, args.ec_depth)
                                   for e in s.split(";")})))
    print(f"[16] KO hits {len(hits):,} -> EC-designated {len(ann):,} "
          f"({len(ann)/max(len(hits),1):.1%} of hits)", flush=True)
    ann.to_csv(out / "protein_ec.tsv", sep="\t", index=False)

    ec_of = dict(zip(ann["tagged_id"], ann["ec_class"]))
    pr = pd.read_csv(args.pairs, sep="\t", dtype=str)
    stats = {"ec_depth": args.ec_depth,
             "require_same_ec": bool(args.require_same_ec),
             "n_ko_hits": int(len(hits)), "n_ec_designated": int(len(ann))}
    out_rows = []
    for cls, g in pr.groupby("class"):
        both = g[g["ext_id"].isin(ec_of) & g["outgroup_id"].isin(ec_of)].copy()
        if args.require_same_ec and len(both):
            same = both.apply(
                lambda r: bool(set(ec_of[r["ext_id"]].split(";"))
                               & set(ec_of[r["outgroup_id"]].split(";"))), axis=1)
            both = both[same]
        stats[f"{cls}_pairs_in"] = int(len(g))
        stats[f"{cls}_pairs_ec"] = int(len(both))
        if len(g):
            print(f"[16] {cls:16s} {len(g):>7,} pairs -> {len(both):>7,} EC-constrained "
                  f"({len(both)/len(g):.1%})", flush=True)
        out_rows.append(both)
    kept = pd.concat(out_rows) if out_rows else pr.head(0)
    kept.to_csv(out / "pairs_ec.tsv", sep="\t", index=False)
    (out / "ec_filter_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"[16] wrote {out/'pairs_ec.tsv'} ({len(kept):,} pairs) and stats", flush=True)


if __name__ == "__main__":
    main()
