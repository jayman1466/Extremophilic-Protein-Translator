#!/usr/bin/env python3
"""Standalone (stdlib-only) merge of chunked SignalP-6.0 outputs into one
secreted-protein table + mature-chain FASTA. Mirrors src/eptrans/signalp.py
logic (SP_CLASSES, ANCHORING, CS parsing, mature = seq[cs_after:]) but has no
repo/pandas dependency so it runs under any python3 on the cluster.

Streams chunk-by-chunk, writing output incrementally (low memory). Sequences are
taken from each chunk's processed_entries.fasta (headers match prediction ids).

Usage:
  python3 merge_signalp_chunks.py --root SIGNALP_R232_DIR \
      --out-table secreted_proteins_r232.tsv \
      --out-fasta  secreted_proteins_r232.faa \
      --out-summary secreted_proteins_r232.summary.json [--mature]
"""
import argparse, glob, gzip, json, os, re
from collections import Counter

SP_CLASSES = ["SP", "LIPO", "TAT", "TATLIPO", "PILIN"]
ALL_CLASSES = ["OTHER"] + SP_CLASSES
ANCHORING = {"SP": "soluble", "TAT": "soluble", "LIPO": "membrane_anchored",
             "TATLIPO": "membrane_anchored", "PILIN": "membrane_anchored", "OTHER": "none"}
PROB_COLS = ["OTHER", "SP(Sec/SPI)", "LIPO(Sec/SPII)", "TAT(Tat/SPI)",
             "TATLIPO(Tat/SPII)", "PILIN(Sec/SPIII)"]
PROB_KEY = {"OTHER": "p_OTHER", "SP(Sec/SPI)": "p_SP", "LIPO(Sec/SPII)": "p_LIPO",
            "TAT(Tat/SPI)": "p_TAT", "TATLIPO(Tat/SPII)": "p_TATLIPO",
            "PILIN(Sec/SPIII)": "p_PILIN"}
CS_RE = re.compile(r"CS pos:\s*(\d+)-(\d+)\.\s*Pr:\s*([\d.]+)")
TSV_COLS = ["genome", "protein_id", "signalp_class", "anchoring", "cs_after", "cs_prob",
            "p_OTHER", "p_SP", "p_LIPO", "p_TAT", "p_TATLIPO", "p_PILIN"]


def parse_chunk_predictions(pred_path):
    """Yield secreted-only dict rows keyed by first-token id (GENOME~PROTID)."""
    header_idx = None
    kept = {}
    with open(pred_path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                cols = [c.strip() for c in line.lstrip("#").strip().split("\t")]
                if cols and cols[0] == "ID":
                    header_idx = {c: i for i, c in enumerate(cols)}
                continue
            fields = line.split("\t")
            first_token = fields[0].split()[0]
            prediction = fields[1] if len(fields) > 1 else "OTHER"
            if prediction not in SP_CLASSES:
                continue
            genome, _, protid = first_token.partition("~")
            probs = {}
            if header_idx:
                for cname in PROB_COLS:
                    j = header_idx.get(cname)
                    if j is not None and j < len(fields):
                        try:
                            probs[PROB_KEY[cname]] = float(fields[j])
                        except ValueError:
                            pass
            cs_after = cs_prob = ""
            m = CS_RE.search(line)
            if m:
                cs_after = int(m.group(1)); cs_prob = float(m.group(3))
            row = {"genome": genome, "protein_id": protid, "signalp_class": prediction,
                   "anchoring": ANCHORING.get(prediction, "none"),
                   "cs_after": cs_after, "cs_prob": cs_prob}
            row.update({k: probs.get(k, "") for k in
                        ["p_OTHER", "p_SP", "p_LIPO", "p_TAT", "p_TATLIPO", "p_PILIN"]})
            kept[first_token] = row
    return kept


def iter_fasta(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    hid = None; seq = []
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if hid is not None:
                    yield hid, "".join(seq)
                hid = line[1:].split()[0] if len(line) > 1 else ""
                seq = []
            else:
                seq.append(line.strip())
    if hid is not None:
        yield hid, "".join(seq)


def class_counts_all(pred_path):
    """Total class counts for the chunk (for summary secreted-fraction)."""
    c = Counter()
    with open(pred_path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.split("\t")
            c[f[1] if len(f) > 1 else "OTHER"] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-table", required=True)
    ap.add_argument("--out-fasta", required=True)
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--mature", action="store_true",
                    help="write mature chain (seq[cs_after:]) instead of precursor")
    args = ap.parse_args()

    chunk_dirs = sorted(glob.glob(os.path.join(args.root, "chunk_*")))
    if not chunk_dirs:
        raise SystemExit(f"no chunk_* dirs under {args.root}")

    total = Counter()
    n_secreted_written = 0
    genomes_seen = set()
    tf = open(args.out_table, "w")
    ff = open(args.out_fasta, "w")
    tf.write("\t".join(TSV_COLS) + "\n")
    for cd in chunk_dirs:
        pred = os.path.join(cd, "prediction_results.txt")
        fasta = os.path.join(cd, "processed_entries.fasta")
        if not (os.path.exists(pred) and os.path.exists(fasta)):
            print(f"[merge] SKIP {cd} (missing pred/fasta)", flush=True)
            continue
        total.update(class_counts_all(pred))
        kept = parse_chunk_predictions(pred)
        # stream sequences, match by first-token id, extract mature chain
        n_chunk = 0
        for hid, seq in iter_fasta(fasta):
            row = kept.get(hid)
            if row is None:
                continue
            genomes_seen.add(row["genome"])
            tf.write("\t".join(str(row[c]) for c in TSV_COLS) + "\n")
            s = seq
            if args.mature and row["cs_after"] and 0 < int(row["cs_after"]) < len(seq):
                s = seq[int(row["cs_after"]):]
            ff.write(f">{hid} class={row['signalp_class']} anchoring={row['anchoring']}\n")
            for i in range(0, len(s), 60):
                ff.write(s[i:i+60] + "\n")
            n_chunk += 1; n_secreted_written += 1
        print(f"[merge] {os.path.basename(cd)}: kept={len(kept)} written={n_chunk}", flush=True)
    tf.close(); ff.close()

    n_total = sum(total.values())
    n_sec = sum(v for k, v in total.items() if k != "OTHER")
    summary = {
        "n_proteins": n_total, "n_secreted": n_sec,
        "secreted_fraction": round(n_sec / n_total, 4) if n_total else 0.0,
        "by_class": {k: int(total.get(k, 0)) for k in ALL_CLASSES},
        "n_secreted_written": n_secreted_written,
        "n_genomes": len(genomes_seen), "mature": args.mature,
    }
    json.dump(summary, open(args.out_summary, "w"), indent=2)
    print(f"[merge] DONE proteins={n_total:,} secreted={n_sec:,} "
          f"({summary['secreted_fraction']*100:.1f}%) written={n_secreted_written:,} "
          f"genomes={len(genomes_seen):,}", flush=True)


if __name__ == "__main__":
    main()
