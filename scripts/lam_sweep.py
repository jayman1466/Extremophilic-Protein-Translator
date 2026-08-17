#!/usr/bin/env python
"""Phase-4 lambda sweep at LOCKED scope+tier, per phenotype.

Scope is locked = secreted (psychrophile 2x2). Tier is locked PER PHENOTYPE
(phase-3b 1x2): H+M for all except halophile (H+M+L). This sweeps the pair-loss
weight lambda over a grid and, for each (lam, phenotype), trains one cached-probe
mean-pooling head at that phenotype's locked tier, scoring on its OWN fixed clean
eval set (that-scope val H+M positives + all that-scope val negatives).

Head/loss/gather logic is copied VERBATIM from phenotype_tier_1x2.py (which is
byte-identical to psy_scope_tier_2x2.py Part B): weighted BCE (rubric confidence
weights + effective pos_weight) + matched-pair margin at lam, margin pairs aligned
to the locked pointwise scope (INV-SCOPE-E). The ONLY change vs 1x2: the train
tensor for a phenotype is gathered ONCE and reused across the lam grid (gather I/O
is the bottleneck); heads are otherwise independent per lam (fresh init, fresh opt).

Each head tracks its best-epoch value SEPARATELY for AUROC and AUPRC (so each
metric is self-consistent for across-lam selection), plus the pair-AUC at the
best-AUROC epoch as a tie-break. Writes, for each lam:
  <out>/cached_probes_lam<LAM>/cached_probe_summary.json = {pheno: {val_auroc,
  val_auprc, val_pair_auc, tier, lam, train_n, train_pos, pos_weight, train_pairs}}
so scripts/select_best_lam.py --metric auroc reads it directly.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/groups/cress/projects/jaymin/eptrans_scratch/repo/src")
from eptrans.modeling.data import phenotype_binary_labels
from eptrans.modeling.losses import confidence_to_weight

DEFAULT_PHENOS = ["psychrophile", "thermophile", "hyperthermophile",
                  "acidophile", "alkaliphile", "halophile"]
DEFAULT_TIERS = "psychrophile:hm,thermophile:hm,hyperthermophile:hm,acidophile:hm,alkaliphile:hm,halophile:hm"


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
    ap.add_argument("--tiers", default=DEFAULT_TIERS,
                    help="comma sep pheno:tier, tier in {hm,all}")
    ap.add_argument("--lams", default="0,0.5,1,2,4",
                    help="comma-separated lambda grid")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--pair-batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1466)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train-device", default="cuda", choices=["cpu", "cuda"])
    args = ap.parse_args()

    phenos = [p.strip() for p in args.phenotypes.split(",") if p.strip()]
    tiers = {}
    for kv in args.tiers.split(","):
        k, v = kv.split(":"); tiers[k.strip()] = v.strip()
    lams = [float(x) for x in args.lams.split(",")]
    lam_strs = [x.strip() for x in args.lams.split(",")]
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
    print("[B] eval-device=%s train-device=%s H=%d scope=%s lams=%s"
          % (dev, tdev, H, args.scope, lams), flush=True)

    pairs = pd.read_csv(args.pairs, sep="\t", dtype=str)
    split_of = dict(zip(df["tagged_id"], df["split"]))
    secr_of = dict(zip(df["tagged_id"], df["is_secreted"].fillna(False).astype(bool)))
    bce_none = nn.BCEWithLogitsLoss

    # results[lam_str] -> {pheno: {...}}
    results = {ls: {"scope": args.scope, "eval": "fixed_clean_HM_at_locked_scope",
                    "lam": float(ls), "phenotypes": {}} for ls in lam_strs}

    def train_head(Xtr, tr_y, tr_w, te, to, Xva_t, va_y, Xve, Xvo, lam, pw):
        n_pos = int((tr_y == 1).sum())
        head = nn.Sequential(nn.Linear(H, args.hidden), nn.GELU(),
                             nn.Dropout(args.dropout), nn.Linear(args.hidden, 1)).to(dev)
        opt = torch.optim.Adam(head.parameters(), lr=args.lr)
        n_tp = 0 if te is None else te.shape[0]
        bce = bce_none(pos_weight=pw, reduction="none")
        ntr = Xtr.shape[0]
        best_ap = -1.0; best_ap_auc = None
        best_auc = -1.0; best_auc_ap = None; best_auc_pair = None
        for ep in range(args.epochs):
            head.train(); perm = torch.randperm(ntr, device=tdev)
            pp_cur = 0; pair_perm = torch.randperm(n_tp, device=tdev) if n_tp else None
            for i in range(0, ntr, args.batch_size):
                idx = perm[i:i + args.batch_size]; idx_d = idx.to(dev)
                s = head(Xtr[idx].to(dev, non_blocking=True)).squeeze(-1)
                loss = (bce(s, tr_y[idx_d]) * tr_w[idx_d]).mean()
                if n_tp and lam > 0:
                    if pp_cur + args.pair_batch_size > n_tp:
                        pair_perm = torch.randperm(n_tp, device=tdev); pp_cur = 0
                    pj = pair_perm[pp_cur:pp_cur + args.pair_batch_size]; pp_cur += args.pair_batch_size
                    se = head(te[pj].to(dev, non_blocking=True)).squeeze(-1)
                    so = head(to[pj].to(dev, non_blocking=True)).squeeze(-1)
                    loss = loss + lam * torch.clamp(args.margin - (se - so), min=0).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            head.eval()
            with torch.no_grad():
                sv = head(Xva_t).squeeze(-1).float().cpu().numpy()
                pair_auc = float("nan")
                if Xve is not None and Xve.shape[0] > 0:
                    sve = head(Xve).squeeze(-1).float().cpu().numpy()
                    svo = head(Xvo).squeeze(-1).float().cpu().numpy()
                    yp = np.concatenate([np.ones(len(sve)), np.zeros(len(svo))])
                    sp = np.concatenate([sve, svo])
                    if len(np.unique(yp)) == 2:
                        pair_auc = float(roc_auc_score(yp, sp))
            ap = average_precision_score(va_y, sv); auc = roc_auc_score(va_y, sv)
            if ap > best_ap:
                best_ap = ap; best_ap_auc = auc
            if auc > best_auc:
                best_auc = auc; best_auc_ap = ap; best_auc_pair = pair_auc
        return {"val_auroc": round(float(best_auc), 4),
                "val_auprc": round(float(best_ap), 4),
                "val_pair_auc": (round(float(best_auc_pair), 4)
                                 if best_auc_pair == best_auc_pair else None),
                "auprc_at_best_auroc": round(float(best_auc_ap), 4),
                "auroc_at_best_auprc": round(float(best_ap_auc), 4),
                "train_n": int(ntr), "train_pos": n_pos,
                "pos_weight": round(float(pw), 2), "train_pairs": int(n_tp)}

    for PHENO in phenos:
        tier = tiers.get(PHENO, "hm")
        y = phenotype_binary_labels(df, PHENO)
        sub_all = df.assign(_y=y); sub_all = sub_all[sub_all["_y"].notna()]
        pool = sub_all if args.scope == "whole" else sub_all[sub_all["is_secreted"].fillna(False).astype(bool)]

        # clean eval set (pointwise): val H+M positives + all val negatives
        va = pool[pool["split"] == "val"]
        va_pos = va[(va["_y"] == 1) & (va["label_confidence"].isin(["high", "medium"]))]
        va_neg = va[va["_y"] == 0]
        va_use = pd.concat([va_pos, va_neg])
        va_rows = np.array([id2row[t] for t in va_use["tagged_id"]])
        va_y = va_use["_y"].values.astype(float)
        Xva_t = torch.tensor(gather(va_rows), device=dev)
        base_rate = float((va_y == 1).mean())

        # val pairs (both endpoints val split, at locked scope) for pair-AUC tie-break
        vp = pairs[pairs["class"] == PHENO]
        vp = vp[vp["ext_id"].isin(id2row) & vp["outgroup_id"].isin(id2row)]
        vp = vp[(vp["ext_id"].map(split_of) == "val") & (vp["outgroup_id"].map(split_of) == "val")]
        if args.scope == "secreted":
            vp = vp[vp["ext_id"].map(secr_of).fillna(False) & vp["outgroup_id"].map(secr_of).fillna(False)]
        if len(vp):
            Xve = torch.tensor(gather(np.array([id2row[t] for t in vp["ext_id"]])), device=dev)
            Xvo = torch.tensor(gather(np.array([id2row[t] for t in vp["outgroup_id"]])), device=dev)
        else:
            Xve = Xvo = None

        # train subset at locked tier
        tr = pool[pool["split"] == "train"]
        if tier == "hm":
            hm_mask = (tr["_y"] == 0) | (tr["label_confidence"].isin(["high", "medium"]))
            tr = tr[hm_mask]
        # train pairs (both endpoints train split, at locked scope)
        pp = pairs[pairs["class"] == PHENO]
        pp = pp[pp["ext_id"].isin(id2row) & pp["outgroup_id"].isin(id2row)]
        pp = pp[(pp["ext_id"].map(split_of) == "train") & (pp["outgroup_id"].map(split_of) == "train")]
        if args.scope == "secreted":
            pp = pp[pp["ext_id"].map(secr_of).fillna(False) & pp["outgroup_id"].map(secr_of).fillna(False)]

        # gather train tensor ONCE, reuse across lam grid
        tr_rows = np.array([id2row[t] for t in tr["tagged_id"]])
        tr_y = torch.tensor(tr["_y"].values.astype(float), device=dev)
        tr_w = torch.tensor(tr["label_confidence"].map(confidence_to_weight).astype(float).values, device=dev)
        Xtr = torch.tensor(gather(tr_rows), device=tdev)
        eff_pos = float(tr_w[tr_y == 1].sum()); eff_neg = float(tr_w[tr_y == 0].sum())
        pw = torch.tensor(eff_neg / max(eff_pos, 1.0), device=dev)
        te = to = None
        if len(pp):
            te = torch.tensor(gather(np.array([id2row[t] for t in pp["ext_id"]])), device=tdev)
            to = torch.tensor(gather(np.array([id2row[t] for t in pp["outgroup_id"]])), device=tdev)
        print("[sweep] %s tier=%s clean val pos=%d neg=%d base=%.4f train_n=%d train_pairs=%d"
              % (PHENO, tier, len(va_pos), len(va_neg), base_rate, Xtr.shape[0],
                 0 if te is None else te.shape[0]), flush=True)

        for lam, ls in zip(lams, lam_strs):
            m = train_head(Xtr, tr_y, tr_w, te, to, Xva_t, va_y, Xve, Xvo, lam, pw)
            m["tier"] = tier; m["base_rate"] = round(base_rate, 4); m["lam"] = lam
            results[ls]["phenotypes"][PHENO] = m
            d = out / ("cached_probes_lam%s" % ls); d.mkdir(parents=True, exist_ok=True)
            (d / "cached_probe_summary.json").write_text(json.dumps(results[ls]["phenotypes"], indent=2))
            print("[sweep] %s lam=%s AUROC %.4f AUPRC %.4f pair_auc %s"
                  % (PHENO, ls, m["val_auroc"], m["val_auprc"], m["val_pair_auc"]), flush=True)

        del Xtr, Xva_t
        if te is not None: del te, to
        if Xve is not None: del Xve, Xvo
        import gc as _gc; _gc.collect(); torch.cuda.empty_cache()

    (out / "lam_sweep_all.json").write_text(json.dumps(results, indent=2))
    print("[sweep] written %s" % (out / "lam_sweep_all.json"), flush=True)
    print("DONE_LAM_SWEEP", flush=True)


if __name__ == "__main__":
    main()
