#!/usr/bin/env python
"""Phase-3b per-phenotype tier 1x2 (H+M+L vs H+M) at a LOCKED scope.

Scope is fixed (decided by the psychrophile 2x2), so this runs one scope per
phenotype and only varies the TRAIN tier:
  tier in {all (H+M+L), hm (H+M-positives-only, all negatives kept)}
Each phenotype is scored on ITS OWN fixed clean eval set at the locked scope:
  <scope> val positives with tier in {high,medium}  +  ALL <scope> val negatives.
Both tier heads for a phenotype share that eval set, so the H+M-vs-all delta is
apples-to-apples. AUROC is the primary metric (scope/signal decision).

Head/loss/gather logic is copied verbatim from psy_scope_tier_2x2.py (Part B):
weighted BCE (rubric confidence weights + effective pos_weight) + matched-pair
margin at lam. Margin pairs aligned to the locked pointwise scope (INV-SCOPE-E).

--train-device cpu (default, safe for whole scope ~107 GB) or cuda (secreted
~90 GB fits an H200 alongside the ~11 GB val tensor -> ~4x faster).

Writes <out>/phenotype_tier_1x2.json.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/groups/cress/projects/jaymin/eptrans_scratch/repo/src")
from eptrans.modeling.data import phenotype_binary_labels
from eptrans.modeling.losses import confidence_to_weight

DEFAULT_PHENOS = ["thermophile", "hyperthermophile", "acidophile", "alkaliphile", "halophile"]


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
    ap.add_argument("--scope", required=True, choices=["whole", "secreted"])
    ap.add_argument("--phenotypes", default=",".join(DEFAULT_PHENOS))
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
    ap.add_argument("--train-device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    phenos = [p.strip() for p in args.phenotypes.split(",") if p.strip()]
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
    df = df[df["tagged_id"].isin(id2row)].reset_index(drop=True)
    print("[load] in-cache rows %d" % len(df), flush=True)

    mm, shard_of, local_of, H = open_emb(args.cache_dir, n)

    def gather(rows):
        r = np.asarray(rows); sh = shard_of[r]; lo = local_of[r]
        outa = np.empty((len(r), H), dtype=np.float32)
        for j in np.unique(sh):
            sel = sh == j; outa[sel] = mm[j][lo[sel]].astype(np.float32)
        return outa

    dev = args.device if torch.cuda.is_available() else "cpu"
    tdev = torch.device(args.train_device if torch.cuda.is_available() else "cpu")
    print("[B] eval-device=%s train-device=%s H=%d scope=%s" % (dev, tdev, H, args.scope), flush=True)

    pairs = pd.read_csv(args.pairs, sep="\t", dtype=str)
    split_of = dict(zip(df["tagged_id"], df["split"]))
    secr_of = dict(zip(df["tagged_id"], df["is_secreted"].fillna(False).astype(bool)))
    bce_none = nn.BCEWithLogitsLoss

    def train_head(tr_df, tpp, Xva_t, va_y, Xve_t=None, Xvo_t=None):
        tr_rows = np.array([id2row[t] for t in tr_df["tagged_id"]])
        tr_y = torch.tensor(tr_df["_y"].values.astype(float), device=dev)
        tr_w = torch.tensor(tr_df["label_confidence"].map(confidence_to_weight).astype(float).values, device=dev)
        Xtr = torch.tensor(gather(tr_rows), device=tdev)
        n_pos = int((tr_y == 1).sum())
        eff_pos = float(tr_w[tr_y == 1].sum()); eff_neg = float(tr_w[tr_y == 0].sum())
        pw = torch.tensor(eff_neg / max(eff_pos, 1.0), device=dev)
        head = nn.Sequential(nn.Linear(H, args.hidden), nn.GELU(),
                             nn.Dropout(args.dropout), nn.Linear(args.hidden, 1)).to(dev)
        opt = torch.optim.Adam(head.parameters(), lr=args.lr)
        te = to = None
        if len(tpp):
            te = torch.tensor(gather(np.array([id2row[t] for t in tpp["ext_id"]])), device=tdev)
            to = torch.tensor(gather(np.array([id2row[t] for t in tpp["outgroup_id"]])), device=tdev)
        n_tp = 0 if te is None else te.shape[0]
        bce = bce_none(pos_weight=pw, reduction="none")
        ntr = Xtr.shape[0]; best_ap = -1.0; best_auc = None; best_pair_auc = None
        for ep in range(args.epochs):
            head.train(); perm = torch.randperm(ntr, device=tdev)
            pp_cur = 0; pair_perm = torch.randperm(n_tp, device=tdev) if n_tp else None
            for i in range(0, ntr, args.batch_size):
                idx = perm[i:i + args.batch_size]; idx_d = idx.to(dev)
                s = head(Xtr[idx].to(dev, non_blocking=True)).squeeze(-1)
                loss = (bce(s, tr_y[idx_d]) * tr_w[idx_d]).mean()
                if n_tp:
                    if pp_cur + args.pair_batch_size > n_tp:
                        pair_perm = torch.randperm(n_tp, device=tdev); pp_cur = 0
                    pj = pair_perm[pp_cur:pp_cur + args.pair_batch_size]; pp_cur += args.pair_batch_size
                    se = head(te[pj].to(dev, non_blocking=True)).squeeze(-1)
                    so = head(to[pj].to(dev, non_blocking=True)).squeeze(-1)
                    loss = loss + args.lam * torch.clamp(args.margin - (se - so), min=0).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            head.eval()
            with torch.no_grad():
                sv = head(Xva_t).squeeze(-1).float().cpu().numpy()
                pair_auc_ep = None
                if Xve_t is not None and Xve_t.shape[0] > 0:
                    sve = head(Xve_t).squeeze(-1).float().cpu().numpy()
                    svo = head(Xvo_t).squeeze(-1).float().cpu().numpy()
                    pair_lab = np.concatenate([np.ones_like(sve), np.zeros_like(svo)])
                    pair_scr = np.concatenate([sve, svo])
                    pair_auc_ep = float(roc_auc_score(pair_lab, pair_scr))
            ap = average_precision_score(va_y, sv); auc = roc_auc_score(va_y, sv)
            if ap > best_ap:
                best_ap = ap; best_auc = auc; best_pair_auc = pair_auc_ep
        out_r = {"clean_val_auprc": round(float(best_ap), 4), "clean_val_auroc": round(float(best_auc), 4),
                "val_pair_auc": (round(float(best_pair_auc), 4) if best_pair_auc is not None else None),
                "train_n": int(ntr), "train_pos": n_pos, "pos_weight": round(float(pw), 2),
                "train_pairs": int(n_tp)}
        del Xtr
        if te is not None: del te, to
        import gc as _gc; _gc.collect(); torch.cuda.empty_cache()
        return out_r

    result = {"scope": args.scope, "eval": "fixed_clean_HM_at_locked_scope", "lam": args.lam,
              "confidence_weights": {t: confidence_to_weight(t) for t in ["high", "medium", "low", "none"]},
              "phenotypes": {}}

    for PHENO in phenos:
        y = phenotype_binary_labels(df, PHENO)
        sub_all = df.assign(_y=y); sub_all = sub_all[sub_all["_y"].notna()]
        pool = sub_all if args.scope == "whole" else sub_all[sub_all["is_secreted"].fillna(False).astype(bool)]

        va = pool[pool["split"] == "val"]
        va_pos = va[(va["_y"] == 1) & (va["label_confidence"].isin(["high", "medium"]))]
        va_neg = va[va["_y"] == 0]
        va_use = pd.concat([va_pos, va_neg])
        va_rows = np.array([id2row[t] for t in va_use["tagged_id"]])
        va_y = va_use["_y"].values.astype(float)
        Xva_t = torch.tensor(gather(va_rows), device=dev)
        base_rate = float((va_y == 1).mean())
        print("[eval] %s clean val: pos=%d neg=%d base=%.4f"
              % (PHENO, len(va_pos), len(va_neg), base_rate), flush=True)

        tr = pool[pool["split"] == "train"]
        pp = pairs[pairs["class"] == PHENO]
        pp = pp[pp["ext_id"].isin(id2row) & pp["outgroup_id"].isin(id2row)]
        pp = pp[(pp["ext_id"].map(split_of) == "train") & (pp["outgroup_id"].map(split_of) == "train")]
        if args.scope == "secreted":
            pp = pp[pp["ext_id"].map(secr_of).fillna(False) & pp["outgroup_id"].map(secr_of).fillna(False)]

        # val-both-endpoints pairs at locked scope (same construction as training pairs, split==val)
        vpp = pairs[pairs["class"] == PHENO]
        vpp = vpp[vpp["ext_id"].isin(id2row) & vpp["outgroup_id"].isin(id2row)]
        vpp = vpp[(vpp["ext_id"].map(split_of) == "val") & (vpp["outgroup_id"].map(split_of) == "val")]
        if args.scope == "secreted":
            vpp = vpp[vpp["ext_id"].map(secr_of).fillna(False) & vpp["outgroup_id"].map(secr_of).fillna(False)]
        Xve_t = Xvo_t = None
        if len(vpp):
            Xve_t = torch.tensor(gather(np.array([id2row[t] for t in vpp["ext_id"]])), device=dev)
            Xvo_t = torch.tensor(gather(np.array([id2row[t] for t in vpp["outgroup_id"]])), device=dev)
        print("[eval] %s val pairs: %d" % (PHENO, len(vpp)), flush=True)

        all_r = train_head(tr, pp, Xva_t, va_y, Xve_t, Xvo_t)
        hm_mask = (tr["_y"] == 0) | (tr["label_confidence"].isin(["high", "medium"]))
        hm_r = train_head(tr[hm_mask], pp, Xva_t, va_y, Xve_t, Xvo_t)
        del Xva_t
        if Xve_t is not None: del Xve_t, Xvo_t
        import gc as _gc; _gc.collect(); torch.cuda.empty_cache()

        result["phenotypes"][PHENO] = {
            "clean_val_pos": int(len(va_pos)), "clean_val_neg": int(len(va_neg)),
            "base_rate": round(base_rate, 4),
            "all_tier": all_r, "hm_only": hm_r,
            "d_auroc_hm_minus_all": round(hm_r["clean_val_auroc"] - all_r["clean_val_auroc"], 4),
            "d_auprc_hm_minus_all": round(hm_r["clean_val_auprc"] - all_r["clean_val_auprc"], 4)}
        print("[1x2] %s all AUROC %.4f (pair %s) / H+M AUROC %.4f (pair %s) (d=%+.4f)"
              % (PHENO, all_r["clean_val_auroc"], str(all_r["val_pair_auc"]),
                 hm_r["clean_val_auroc"], str(hm_r["val_pair_auc"]),
                 result["phenotypes"][PHENO]["d_auroc_hm_minus_all"]), flush=True)
        (out / "phenotype_tier_1x2.json").write_text(json.dumps(result, indent=2))

    print("[1x2] written %s" % (out / "phenotype_tier_1x2.json"), flush=True)
    print("DONE_PHENO_1X2", flush=True)


if __name__ == "__main__":
    main()
