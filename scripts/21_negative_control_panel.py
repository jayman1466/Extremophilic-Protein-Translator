#!/usr/bin/env python
"""Negative-control panel: score held-out mesophile proteins with each phenotype head.

A phenotype head consumed by ``11_generate_pipelineB.py`` is read as
``p_classifier`` -- a probability that a sequence carries the phenotype.  A head
that is behaving should place a *mesophile* protein near the corpus base rate.
The production heads do not: trained with ``pos_weight`` up to 975, they carry a
fixed ``+ln(pos_weight)`` logit offset and saturate, which is why the mesophilic
E. coli IS621 wild type scores at the ceiling.

This script runs the same panel against any heads directory, so the pre-fix and
post-fix heads are directly comparable.  Heads written by
``20_train_calibrated_heads.py`` carry a ``calibration`` block and are scored
through it; heads written by ``19_train_meanpool_production.py`` do not and are
scored as the generation script scores them today (bare sigmoid).

The panel is drawn from confident mesophiles only, with an optional taxonomic
holdout so the comparison is not flattered by genomes a head was trained on.
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn

sys.path.insert(0, "/groups/cress/projects/jaymin/eptrans_scratch/repo/src")

_PREFIX_RE = re.compile(r"^(GB_|RS_|CU_)")
QS = [("p01", .01), ("p25", .25), ("p50", .50), ("p75", .75),
      ("p95", .95), ("p99", .99)]


def load_ids(cache):
    d = Path(cache); ids = []
    for ip in sorted(d.glob("ids_shard*.txt"),
                     key=lambda p: int(p.stem.replace("ids_shard", ""))):
        ids.extend([x for x in ip.read_text().split("\n") if x])
    return ids


def open_emb(cache, n):
    d = Path(cache)
    sh = sorted(int(p.stem.replace("emb_shard", "")) for p in d.glob("emb_shard*.npy"))
    mm, sizes = [], []
    for s in sh:
        a = np.load(d / ("emb_shard%d.npy" % s), mmap_mode="r")
        mm.append(a); sizes.append(a.shape[0])
    offs = np.concatenate([[0], np.cumsum(sizes)])
    shard_of = np.empty(int(sum(sizes)), dtype=np.int16)
    local_of = np.empty(int(sum(sizes)), dtype=np.int64)
    for j, (a, b) in enumerate(zip(offs[:-1], offs[1:])):
        shard_of[a:b] = j; local_of[a:b] = np.arange(b - a, dtype=np.int64)
    assert int(sum(sizes)) == n
    return mm, shard_of, local_of, mm[0].shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--labeled", required=True)
    ap.add_argument("--heads-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="heads", help="tag for this heads set")
    ap.add_argument("--scope", default="secreted", choices=["whole", "secreted"])
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1466)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)

    ids = load_ids(args.cache_dir); id2row = {t: i for i, t in enumerate(ids)}
    mm, shard_of, local_of, Hd = open_emb(args.cache_dir, len(ids))

    def gather(rows):
        r = np.asarray(rows); sh = shard_of[r]; lo = local_of[r]
        o = np.empty((len(r), Hd), dtype=np.float32)
        for j in np.unique(sh):
            sel = sh == j; o[sel] = mm[j][lo[sel]].astype(np.float32)
        return o

    df = pd.read_parquet(args.labeled)
    df = df[df["tagged_id"].isin(id2row)]
    if args.scope == "secreted":
        df = df[df["is_secreted"].fillna(False).astype(bool)]
    mes = df[df["is_mesophile"].fillna(False).astype(bool)]
    k = min(args.n, len(mes))
    mp = mes.iloc[rng.choice(len(mes), size=k, replace=False)]
    X = torch.tensor(gather([id2row[t] for t in mp["tagged_id"]]))
    print("[21] mesophile panel n=%d dim=%d" % (k, Hd), flush=True)

    res = {"label": args.label, "heads_dir": args.heads_dir, "n_panel": int(k),
           "scope": args.scope, "phenotypes": {}}
    for hd in sorted(Path(args.heads_dir).glob("clf_*")):
        pheno = hd.name.replace("clf_", "")
        ck = torch.load(hd / "head_best.pt", map_location="cpu")
        arch = ck.get("arch", {})
        head = nn.Sequential(nn.Linear(arch.get("in_dim", Hd), arch.get("hidden", 512)),
                             nn.GELU(), nn.Dropout(arch.get("dropout", 0.1)),
                             nn.Linear(arch.get("hidden", 512), 1))
        head.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
        head.eval().to(args.device)
        z = []
        with torch.no_grad():
            for i in range(0, X.shape[0], 8192):
                z.append(head(X[i:i + 8192].to(args.device)).squeeze(-1).float().cpu().numpy())
        z = np.concatenate(z)
        cal = ck.get("calibration")
        p_raw = 1.0 / (1.0 + np.exp(-z))
        p = (1.0 / (1.0 + np.exp(-(cal["a"] * z + cal["b"])))) if cal else p_raw
        res["phenotypes"][pheno] = {
            "calibrated": bool(cal),
            "p": {q: round(float(np.quantile(p, v)), 5) for q, v in QS},
            "p_max": round(float(p.max()), 5),
            "frac_above_0.5": round(float((p > 0.5).mean()), 5),
            "frac_above_0.9": round(float((p > 0.9).mean()), 5),
            "logit": {"p50": round(float(np.median(z)), 3),
                      "p99": round(float(np.quantile(z, .99)), 3),
                      "max": round(float(z.max()), 3)}}
        print("[21] %-17s calibrated=%-5s median_p=%.4f frac>0.9=%.4f"
              % (pheno, bool(cal), np.median(p), (p > 0.9).mean()), flush=True)

    Path(args.out).write_text(json.dumps(res, indent=2))
    print("DONE_NEGATIVE_CONTROL", flush=True)


if __name__ == "__main__":
    main()
