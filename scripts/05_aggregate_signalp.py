#!/usr/bin/env python3
"""Stage 05agg - aggregate SignalP chunk outputs into one per-protein table.

Consumes every ``prediction_results.txt`` under the given chunk directories plus,
optionally, a legacy already-aggregated TSV, and emits:

    <out>                 per-protein table: tagged_id, genome, protein_id,
                          prediction, cs_prob, is_secreted, anchoring
    <out>.stats.json      counts by prediction class, per source
    --faa-secreted        FASTA of secreted mature chains  (50% clustering input)
    --faa-whole           FASTA of ALL proteins for whole-proteome-scope genomes
                          (40% clustering input)

`is_secreted` is `prediction != "OTHER"`, i.e. any signal-peptide class (SP, LIPO,
TAT, TATLIPO, PILIN) -- the definition already implemented by
`SignalPrediction.is_secreted` and used to build the r232 production table. Note
this includes the membrane-anchored classes (LIPO/TATLIPO/PILIN): their mature
chain is extracellular-facing but remains membrane-tethered (see ANCHORING). The
`anchoring` column is carried through so a downstream stage can restrict to
`soluble` without re-running SignalP.

Usage
-----
    python scripts/05_aggregate_signalp.py \
        --pred-dirs '/path/signalp_targeted/chunk_*' \
        --legacy /path/secreted_proteins_r232.tsv \
        --faa-secreted secretome.faa --faa-whole wholeproteome.faa \
        --whole-scope-accessions whole.txt \
        --out secreted_all.tsv
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eptrans.signalp import ANCHORING, parse_prediction_results  # noqa: E402


def _iter_pred_files(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        for d in sorted(glob.glob(pat)):
            f = Path(d) / "prediction_results.txt"
            if f.is_file() and f.stat().st_size > 0:
                out.append(f)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dirs", nargs="+", required=True,
                    help="glob(s) matching chunk dirs holding prediction_results.txt")
    ap.add_argument("--legacy", default=None,
                    help="previously aggregated per-protein TSV to union in")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", default=None)
    ap.add_argument("--faa-secreted", default=None,
                    help="write secreted mature-chain FASTA here (needs --proteome-root)")
    ap.add_argument("--faa-whole", default=None,
                    help="write all-protein FASTA for whole-scope genomes here")
    ap.add_argument("--whole-scope-accessions", default=None,
                    help="file of accessions whose FULL proteome is in scope")
    ap.add_argument("--proteome-root", default=None,
                    help="root holding <domain>/<ACC>_protein.faa.gz")
    ap.add_argument("--extra-proteome-root", action="append", default=None,
                    help="additional proteome root, repeatable. Needed because the "
                         "ingested MAGs (CU_CUST_*) live in custom_genomes/, a parallel "
                         "tree to the GTDB protein_faa_reps/.")
    args = ap.parse_args()

    files = _iter_pred_files(args.pred_dirs)
    if not files:
        raise SystemExit(f"[05agg] no prediction_results.txt matched {args.pred_dirs}")
    print(f"[05agg] {len(files)} prediction files")

    rows = []
    per_source = {}
    for f in files:
        preds = parse_prediction_results(f)
        per_source[str(f.parent.name)] = len(preds)
        for p in preds:
            gen, _, pid = p.protein_id.partition("~")
            rows.append((p.protein_id, gen or None, pid or p.protein_id,
                         p.prediction, p.cs_prob, p.prediction != "OTHER",
                         ANCHORING.get(p.prediction, "none")))
        print(f"  {f.parent.name}: {len(preds):,}")

    df = pd.DataFrame(rows, columns=["tagged_id", "genome", "protein_id",
                                     "prediction", "cs_prob", "is_secreted",
                                     "anchoring"])
    n_new = len(df)

    if args.legacy and Path(args.legacy).is_file():
        leg = pd.read_csv(args.legacy, sep="\t", low_memory=False)
        # the legacy table holds SECRETED rows only; keep its columns compatible
        if "tagged_id" not in leg.columns and {"genome", "protein_id"} <= set(leg.columns):
            leg["tagged_id"] = leg["genome"].astype(str) + "~" + leg["protein_id"].astype(str)
        for c, d in [("prediction", "SP"), ("is_secreted", True), ("anchoring", "soluble")]:
            if c not in leg.columns:
                leg[c] = d
        keep = [c for c in df.columns if c in leg.columns]
        df = pd.concat([df, leg[keep]], ignore_index=True)
        print(f"[05agg] legacy rows {len(leg):,}")

    before = len(df)
    df = df.drop_duplicates(subset=["tagged_id"], keep="first").reset_index(drop=True)
    print(f"[05agg] {before:,} rows -> {len(df):,} unique tagged_id "
          f"({before - len(df):,} duplicates dropped, new-first)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep="\t", index=False)
    n_sec = int(df["is_secreted"].sum())
    stats = {
        "n_proteins": int(len(df)),
        "n_secreted": n_sec,
        "secreted_fraction": round(n_sec / max(1, len(df)), 6),
        "n_from_chunks": n_new,
        "by_prediction": {k: int(v) for k, v in df["prediction"].value_counts().items()},
        "by_anchoring": {k: int(v) for k, v in df["anchoring"].value_counts().items()},
        "per_source": per_source,
        "n_genomes": int(df["genome"].nunique()),
    }
    print(f"[05agg] proteins {len(df):,} | secreted {n_sec:,} "
          f"({100*n_sec/max(1,len(df)):.2f}%) | genomes {stats['n_genomes']:,}")
    if args.stats:
        Path(args.stats).write_text(json.dumps(stats, indent=1, sort_keys=True))

    # ---- optional FASTA emission for the two clustering inputs ----
    if args.faa_secreted or args.faa_whole:
        if not args.proteome_root:
            raise SystemExit("--faa-* requires --proteome-root")
        whole = set()
        if args.whole_scope_accessions and Path(args.whole_scope_accessions).is_file():
            whole = {l.strip() for l in open(args.whole_scope_accessions) if l.strip()}
            print(f"[05agg] whole-scope genomes: {len(whole):,}")
        sec_ids = set(df.loc[df["is_secreted"], "tagged_id"])
        roots = [Path(args.proteome_root)] + [Path(r) for r in (args.extra_proteome_root or [])]
        print(f"[05agg] proteome roots: {[str(r) for r in roots]}")
        fh_s = open(args.faa_secreted, "w") if args.faa_secreted else None
        fh_w = open(args.faa_whole, "w") if args.faa_whole else None
        n_s = n_w = missing = 0
        # Iterate the UNION of prediction-table genomes and whole-scope genomes.
        #
        # THE BUG THIS FIXES: the loop used to walk df["genome"] only. Whole-proteome
        # scope does NOT require SignalP -- every protein is wanted regardless of
        # secretion -- so a whole-scope genome with no SignalP row was never opened
        # and contributed ZERO sequences, silently. Measured on the real run: of 7,320
        # whole-scope genomes, only 685 appear in the prediction table (630 via the
        # legacy secreted-only r232 table, 55 via fresh targeted chunks), so 6,635
        # (90.6%) of the hyperthermophile+psychrophile whole-proteome corpus would
        # have been missing -- and the emptiness guard would NOT have fired, because
        # 685 genomes still yield a non-empty file.
        for gen in sorted(set(df["genome"].dropna().unique()) | whole):
            hits = []
            for root in roots:
                hits = list(root.glob(f"*/{gen}_protein.faa.gz")) or \
                       list(root.glob(f"{gen}_protein.faa.gz"))
                if hits:
                    break
            if not hits:
                missing += 1
                continue
            with gzip.open(hits[0], "rt") as fh:
                pid = None
                buf: list[str] = []

                def flush():
                    nonlocal n_s, n_w
                    if pid is None:
                        return
                    tid = f"{gen}~{pid}"
                    seq = "".join(buf)
                    if fh_s and tid in sec_ids:
                        fh_s.write(f">{tid}\n{seq}\n"); n_s += 1
                    if fh_w and gen in whole:
                        fh_w.write(f">{tid}\n{seq}\n"); n_w += 1

                for line in fh:
                    if line.startswith(">"):
                        flush()
                        pid = line[1:].split()[0]
                        buf = []
                    else:
                        buf.append(line.strip())
                flush()
        for fh in (fh_s, fh_w):
            if fh:
                fh.close()
        print(f"[05agg] wrote secreted FASTA {n_s:,} seqs | whole FASTA {n_w:,} seqs")
        if missing:
            print(f"[05agg] WARNING: {missing:,} genomes had no proteome file under "
                  f"{[str(r) for r in roots]}")
        if whole:
            n_ws_emitted = len(whole & set(df["genome"].dropna().unique()))
            print(f"[05agg] whole-scope genomes requested {len(whole):,} | "
                  f"in prediction table {n_ws_emitted:,} | "
                  f"read from proteome regardless of SignalP {len(whole):,}")
        if n_s == 0 and args.faa_secreted:
            raise SystemExit("[05agg] secreted FASTA is EMPTY -- proteome roots wrong?")
        if args.faa_whole and whole and n_w == 0:
            raise SystemExit("[05agg] whole FASTA is EMPTY but whole-scope genomes were "
                             "requested -- accession space mismatch?")


if __name__ == "__main__":
    main()
