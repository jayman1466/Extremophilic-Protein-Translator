#!/usr/bin/env python
"""Phase-3 psychrophile scope x tier 2x2 on ONE fixed clean eval set.

Deployment compartment = secreted. To make scope and tier directly comparable
(AUROC), ALL four heads are scored on the SAME clean eval set:
    secreted val positives with tier in {high,medium}  +  ALL secreted val negatives.

Only the TRAIN set varies:
  train pointwise scope in {whole, secreted}   (whole = all is_secreted values;
                                                secreted = is_secreted==True only)
  train tier            in {all (H+M+L), hm (H+M)}
All heads: lam=1, margin=1, weighted BCE (rubric confidence weights + effective
pos_weight) + matched-pair margin. Margin pairs are aligned to the train pointwise
scope (INV-SCOPE-E): for secreted-scope training only secreted ext/outgroup pairs
enter the margin term.

Reuses the exact head/loss/gather logic from scope_tier_measure.py Part B.
Writes <out>/psy_scope_tier_2x2.json.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/groups/cress/projects/jaymin/eptrans_scratch/repo/src")
from eptrans.modeling.data import phenotype_binary_labels
from eptrans.modeling.losses import confidence_to_weight

PHENO = "psychrophile"


def load_ids(cache):
    d = Path(cache); ids = []
    sh = sorted(d.glob("ids_shard*.txt"), key=lambda p: int(p.stem.replace("ids_shard", "")))
    for ip in sh:
        ids.extend([x for x in ip.read_text().split("\n") if x])
    return ids


def open_emb(cache, n):
    d = Path(cache)
    sh = sorted(int(p.stem.replace("emb_shard", "")) for p in d.glob("emb_shard*.npy"))
    mm = []; sizes = []
    for s in sh:
        a = np.load(d / ("emb_shard%d.npy" % s), mmap_mode="r"); mm.append(a); sizes.append(a.shape[0])
    offs = np.concatenate([[0], np.cumsum(sizes)])
    shard_of = np.empty(int(sum(sizes)), dtype=np.int16); local_of = np.empty(int(sum(sizes)), dtype=np.int64)
    for j, (a, b) in enumerate(zip(offs[:-1], offs[1:])):
        shard_of[a:b] = j; local_of[a:b] = np.arange(b - a, dtype=np.int64)
    assert int(sum(sizes)) == n, "emb rows %d != ids %d" % (sum(sizes), n)
    return mm, shard_of, local_of, mm[0].shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--labeled", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--pair-batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1466)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    import torch, torch.nn as nn
    from sklearn.metrics import average_precision_score, roc_auc_score
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("[load] ids", flush=True)
    ids = load_ids(args.cache_dir); n = len(ids)
    id2row = {t: i for i, t in enumerate(ids)}
    print("[load] %d cache ids" % n, flush=True)

    cols = ["tagged_id", "genome", "label", "is_mesophile", "label_confidence",
            "is_secreted", "split", "cluster_id50", "cluster_id40"]
    df = pd.read_parquet(args.labeled, columns=cols)
    df["tagged_id"] = df["tagged_id"].astype(str)
    incache = df["tagged_id"].isin(id2row)
    df = df[incache].reset_index(drop=True)
    print("[load] in-cache rows %d" % len(df), flush=True)

    mm, shard_of, local_of, H = open_emb(args.cache_dir, n)

    def gather(rows):
        r = np.asarray(rows); sh = shard_of[r]; lo = local_of[r]
        outa = np.empty((len(r), H), dtype=np.float32)
        for j in np.unique(sh):
            sel = sh == j; outa[sel] = mm[j][lo[sel]].astype(np.float32)
        return outa

    dev = args.device if torch.cuda.is_available() else "cpu"
    print("[B] device=%s H=%d" % (dev, H), flush=True)

    y = phenotype_binary_labels(df, PHENO)
    sub_all = df.assign(_y=y); sub_all = sub_all[sub_all["_y"].notna()]
    sub_secr = sub_all[sub_all["is_secreted"].fillna(False).astype(bool)]

    # ---- FIXED clean eval set: secreted val, H+M positives + all secreted val negatives ----
    va = sub_secr[sub_secr["split"] == "val"]
    va_pos = va[(va["_y"] == 1) & (va["label_confidence"].isin(["high", "medium"]))]
    va_neg = va[va["_y"] == 0]
    va_use = pd.concat([va_pos, va_neg])
    va_rows = np.array([id2row[t] for t in va_use["tagged_id"]])
    va_y = va_use["_y"].values.astype(float)
    Xva_t = torch.tensor(gather(va_rows), device=dev)
    base_rate = float((va_y == 1).mean())
    print("[eval] FIXED secreted clean val: pos=%d neg=%d base=%.4f"
          % (len(va_pos), len(va_neg), base_rate), flush=True)

    pairs = pd.read_csv(args.pairs, sep="\t", dtype=str)
    split_of = dict(zip(df["tagged_id"], df["split"]))
    secr_of = dict(zip(df["tagged_id"], df["is_secreted"].fillna(False).astype(bool)))
    bce_none = nn.BCEWithLogitsLoss

    def train_head(tr_df, tpp):
        tr_rows = np.array([id2row[t] for t in tr_df["tagged_id"]])
        tr_y = torch.tensor(tr_df["_y"].values.astype(float), device=dev)
        tr_w = torch.tensor(tr_df["label_confidence"].map(confidence_to_weight).astype(float).values, device=dev)
        Xtr = torch.tensor(gather(tr_rows), device=dev)
        n_pos = int((tr_y == 1).sum())
        eff_pos = float(tr_w[tr_y == 1].sum()); eff_neg = float(tr_w[tr_y == 0].sum())
        pw = torch.tensor(eff_neg / max(eff_pos, 1.0), device=dev)
        head = nn.Sequential(nn.Linear(H, args.hidden), nn.GELU(),
                             nn.Dropout(args.dropout), nn.Linear(args.hidden, 1)).to(dev)
        opt = torch.optim.Adam(head.parameters(), lr=args.lr)
        te = to = None
        if len(tpp):
            te = torch.tensor(gather(np.array([id2row[t] for t in tpp["ext_id"]])), device=dev)
            to = torch.tensor(gather(np.array([id2row[t] for t in tpp["outgroup_id"]])), device=dev)
        n_tp = 0 if te is None else te.shape[0]
        bce = bce_none(pos_weight=pw, reduction="none")
        ntr = Xtr.shape[0]; best_ap = -1.0; best_auc = None
        for ep in range(args.epochs):
            head.train(); perm = torch.randperm(ntr, device=dev)
            pp_cur = 0; pair_perm = torch.randperm(n_tp, device=dev) if n_tp else None
            for i in range(0, ntr, args.batch_size):
                idx = perm[i:i + args.batch_size]
                s = head(Xtr[idx]).squeeze(-1)
                loss = (bce(s, tr_y[idx]) * tr_w[idx]).mean()
                if n_tp:
                    if pp_cur + args.pair_batch_size > n_tp:
                        pair_perm = torch.randperm(n_tp, device=dev); pp_cur = 0
                    pj = pair_perm[pp_cur:pp_cur + args.pair_batch_size]; pp_cur += args.pair_batch_size
                    se = head(te[pj]).squeeze(-1); so = head(to[pj]).squeeze(-1)
                    loss = loss + args.lam * torch.clamp(args.margin - (se - so), min=0).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            head.eval()
            with torch.no_grad():
                sv = head(Xva_t).squeeze(-1).float().cpu().numpy()
            ap = average_precision_score(va_y, sv); auc = roc_auc_score(va_y, sv)
            if ap > best_ap:
                best_ap = ap; best_auc = auc
        return {"clean_val_auprc": round(float(best_ap), 4), "clean_val_auroc": round(float(best_auc), 4),
                "train_n": int(ntr), "train_pos": n_pos, "pos_weight": round(float(pw), 2),
                "train_pairs": int(n_tp)}

    res = {"pheno": PHENO, "eval": "fixed_secreted_clean_HM", "lam": args.lam,
           "clean_val_pos": int(len(va_pos)), "clean_val_neg": int(len(va_neg)),
           "base_rate": round(base_rate, 4),
           "confidence_weights": {t: confidence_to_weight(t) for t in ["high", "medium", "low", "none"]},
           "cells": {}}

    for train_scope in ["whole", "secreted"]:
        pool = sub_all if train_scope == "whole" else sub_secr
        tr = pool[pool["split"] == "train"]
        # margin pairs aligned to train pointwise scope
        pp = pairs[pairs["class"] == PHENO]
        pp = pp[pp["ext_id"].isin(id2row) & pp["outgroup_id"].isin(id2row)]
        pp = pp[(pp["ext_id"].map(split_of) == "train") & (pp["outgroup_id"].map(split_of) == "train")]
        if train_scope == "secreted":
            pp = pp[pp["ext_id"].map(secr_of).fillna(False) & pp["outgroup_id"].map(secr_of).fillna(False)]
        all_r = train_head(tr, pp)
        hm_r = train_head(tr[tr["label_confidence"].isin(["high", "medium"])], pp)
        res["cells"][train_scope] = {
            "train_pointwise_scope": train_scope,
            "all_tier": all_r, "hm_only": hm_r,
            "d_auroc_hm_minus_all": round(hm_r["clean_val_auroc"] - all_r["clean_val_auroc"], 4),
            "d_auprc_hm_minus_all": round(hm_r["clean_val_auprc"] - all_r["clean_val_auprc"], 4)}
        print("[2x2] scope=%s all AUROC %.4f / H+M AUROC %.4f (d=%+.4f)"
              % (train_scope, all_r["clean_val_auroc"], hm_r["clean_val_auroc"],
                 res["cells"][train_scope]["d_auroc_hm_minus_all"]), flush=True)

    # scope deltas at fixed tier
    cw, cs = res["cells"]["whole"], res["cells"]["secreted"]
    res["d_auroc_secreted_minus_whole"] = {
        "all_tier": round(cs["all_tier"]["clean_val_auroc"] - cw["all_tier"]["clean_val_auroc"], 4),
        "hm_only": round(cs["hm_only"]["clean_val_auroc"] - cw["hm_only"]["clean_val_auroc"], 4)}
    (out / "psy_scope_tier_2x2.json").write_text(json.dumps(res, indent=2))
    print("[2x2] written %s" % (out / "psy_scope_tier_2x2.json"), flush=True)
    print("DONE_PSY_2X2", flush=True)


if __name__ == "__main__":
    main()
