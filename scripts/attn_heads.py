#!/usr/bin/env python
"""Best-lambda gated-attention pooling heads at LOCKED scope=secreted, tier=H+M.

Reads the mhk32 top-32 per-residue cache written by 09_embed_secretome.py
--emit-topk:
  topk_shard{i}.npy  (n_i, K=32, H=2560) fp16   per-residue block (saliency top-K)
  lens_shard{i}.npy  (n_i,) int32               true mature length (for the mask)
  ids_shard{i}.txt   tagged_id per row
The mean cache (emb_shard) is NOT needed for attention pooling.

Head: gated attention pooling (Ilse et al. 2018), eptrans.modeling.pooling
      .build_pooling_head('attention'): e_i = w^T(tanh(V h_i) * sigmoid(U h_i)),
      alpha = softmax(e) over the K residues, z = sum_i alpha_i h_i, then the
      SAME 2-layer MLP readout as the mean-pool heads.

Loss is byte-identical to lam_sweep.py: rubric-confidence-weighted BCE with an
EFFECTIVE (rubric-weighted) pos_weight + matched-pair hinge margin at the
per-phenotype best lambda. Pairs aligned to the locked scope.

TRAIN set (user decision 2026-08-17): class-balanced subsample -- ALL H+M
positives + neg_per_pos x negatives -- gathered into a CPU fp16 tensor for fast
shuffled minibatching. Effective pos_weight corrects the residual imbalance.

EVAL set: the FULL clean set (val H+M positives + ALL val negatives), gathered
ONCE into a CPU fp16 tensor, so val_auroc / val_auprc / val_pair_auc are the
SAME eval as the lambda sweep / mean-pool 1x2 -> directly comparable headline.
Best epoch selected by val_auroc (locked policy); auprc + pair_auc reported at
that epoch. Attention alpha at the best epoch is dumped for interpretability.

Per phenotype writes <out>/attn_<pheno>/{metrics.json,history.json,head_best.pt,
alpha_ext_best.npy}; aggregate <out>/attn_heads_summary.json; prints
DONE_ATTN_HEADS.
"""
import argparse, json, sys, time, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/groups/cress/projects/jaymin/eptrans_scratch/repo/src")
from eptrans.modeling.data import phenotype_binary_labels
from eptrans.modeling.losses import confidence_to_weight

DEFAULT_PHENOS = ["psychrophile", "thermophile", "hyperthermophile",
                  "acidophile", "alkaliphile", "halophile"]


def load_ids(cache):
    d = Path(cache); ids = []
    sh = sorted(d.glob("ids_shard*.txt"), key=lambda p: int(p.stem.replace("ids_shard", "")))
    for ip in sh:
        ids.extend([x for x in ip.read_text().split("\n") if x])
    return ids


def open_topk(cache, n):
    """mmap topk_shard + lens_shard; return (mm_list, shard_of, local_of, lens_all, K, H)."""
    d = Path(cache)
    shs = sorted(int(p.stem.replace("topk_shard", "")) for p in d.glob("topk_shard*.npy"))
    mm = []; sizes = []; lens_parts = []
    for s in shs:
        a = np.load(d / ("topk_shard%d.npy" % s), mmap_mode="r"); mm.append(a); sizes.append(a.shape[0])
        lens_parts.append(np.load(d / ("lens_shard%d.npy" % s)))
    total = int(sum(sizes))
    offs = np.concatenate([[0], np.cumsum(sizes)])
    shard_of = np.empty(total, dtype=np.int16); local_of = np.empty(total, dtype=np.int64)
    for j, (a, b) in enumerate(zip(offs[:-1], offs[1:])):
        shard_of[a:b] = j; local_of[a:b] = np.arange(b - a, dtype=np.int64)
    lens_all = np.concatenate(lens_parts).astype(np.int64)
    assert total == n, "topk rows %d != ids %d" % (total, n)
    assert len(lens_all) == n, "lens rows %d != ids %d" % (len(lens_all), n)
    K = mm[0].shape[1]; H = mm[0].shape[2]
    return mm, shard_of, local_of, lens_all, K, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--labeled", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scope", default="secreted", choices=["whole", "secreted"])
    ap.add_argument("--phenotypes", default=",".join(DEFAULT_PHENOS))
    ap.add_argument("--lam-json", default="", help="json {pheno: lam}; overrides --default-lam")
    ap.add_argument("--default-lam", type=float, default=1.0)
    ap.add_argument("--neg-per-pos", type=float, default=3.0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--eval-batch-size", type=int, default=8192)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--attn-dim", type=int, default=128)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--pair-batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1466)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    phenos = [p.strip() for p in args.phenotypes.split(",") if p.strip()]
    lam_map = {}
    if args.lam_json:
        lam_map = {k: float(v) for k, v in json.loads(Path(args.lam_json).read_text()).items()}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    import torch, torch.nn as nn
    from sklearn.metrics import average_precision_score, roc_auc_score
    from eptrans.modeling.pooling import build_pooling_head
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = args.device if torch.cuda.is_available() else "cpu"

    print("[load] ids", flush=True)
    ids = load_ids(args.cache_dir); n = len(ids)
    id2row = {t: i for i, t in enumerate(ids)}
    print("[load] %d cache ids" % n, flush=True)
    mm, shard_of, local_of, lens_all, K, H = open_topk(args.cache_dir, n)
    print("[load] topk cache mmap: n=%d K=%d H=%d" % (n, K, H), flush=True)

    def gather_topk(rows):
        """(len(rows), K, H) fp16 CPU tensor + (len(rows), K) fp16 CPU mask."""
        r = np.asarray(rows); sh = shard_of[r]; lo = local_of[r]
        outa = np.empty((len(r), K, H), dtype=np.float16)
        for j in np.unique(sh):
            sel = sh == j
            loj = lo[sel]
            order = np.argsort(loj, kind="stable")   # sort into monotonic offset order
            block = mm[j][loj[order]]                 # sequential Lustre read (no seek amplification)
            dest = np.empty_like(block)
            dest[order] = block                        # unscatter back to caller's row order
            outa[sel] = dest
        kv = np.minimum(lens_all[r], K)
        mask = (np.arange(K)[None, :] < kv[:, None]).astype(np.float16)
        return torch.from_numpy(outa), torch.from_numpy(mask)

    cols = ["tagged_id", "label", "label_confidence", "is_secreted", "split"]
    df = pd.read_parquet(args.labeled, columns=cols)
    df["tagged_id"] = df["tagged_id"].astype(str)
    df = df[df["tagged_id"].isin(id2row)].reset_index(drop=True)
    print("[load] in-cache rows %d" % len(df), flush=True)

    pairs = pd.read_csv(args.pairs, sep="\t", dtype=str)
    split_of = dict(zip(df["tagged_id"], df["split"]))
    secr_of = dict(zip(df["tagged_id"], df["is_secreted"].fillna(False).astype(bool)))

    result = {"scope": args.scope, "tier": "hm", "pooling": "attention",
              "eval": "fixed_clean_HM_at_locked_scope", "neg_per_pos": args.neg_per_pos,
              "confidence_weights": {t: confidence_to_weight(t) for t in ["high", "medium", "low", "none"]},
              "phenotypes": {}}

    for PHENO in phenos:
        lam = lam_map.get(PHENO, args.default_lam)
        t0 = time.time()
        y = phenotype_binary_labels(df, PHENO)
        sub_all = df.assign(_y=y); sub_all = sub_all[sub_all["_y"].notna()]
        pool = sub_all if args.scope == "whole" else sub_all[sub_all["is_secreted"].fillna(False).astype(bool)]

        # ---- clean eval set (pointwise): val H+M positives + ALL val negatives
        va = pool[pool["split"] == "val"]
        va_pos = va[(va["_y"] == 1) & (va["label_confidence"].isin(["high", "medium"]))]
        va_neg = va[va["_y"] == 0]
        va_use = pd.concat([va_pos, va_neg])
        va_rows = np.array([id2row[t] for t in va_use["tagged_id"]])
        va_y = va_use["_y"].values.astype(float)
        base_rate = float((va_y == 1).mean())
        Xva, Mva = gather_topk(va_rows)   # CPU fp16, streamed to GPU in eval batches

        # ---- val pairs (both endpoints val, locked scope) for pair-AUC
        vp = pairs[pairs["class"] == PHENO]
        vp = vp[vp["ext_id"].isin(id2row) & vp["outgroup_id"].isin(id2row)]
        vp = vp[(vp["ext_id"].map(split_of) == "val") & (vp["outgroup_id"].map(split_of) == "val")]
        if args.scope == "secreted":
            vp = vp[vp["ext_id"].map(secr_of).fillna(False) & vp["outgroup_id"].map(secr_of).fillna(False)]
        Xve = Xvo = Mve = Mvo = None
        if len(vp):
            Xve, Mve = gather_topk(np.array([id2row[t] for t in vp["ext_id"]]))
            Xvo, Mvo = gather_topk(np.array([id2row[t] for t in vp["outgroup_id"]]))

        # ---- train subset: tier=hm, then class-balanced subsample
        tr = pool[pool["split"] == "train"]
        hm_mask = (tr["_y"] == 0) | (tr["label_confidence"].isin(["high", "medium"]))
        tr = tr[hm_mask]
        tr_pos = tr[tr["_y"] == 1]; tr_neg = tr[tr["_y"] == 0]
        n_keep = min(len(tr_neg), int(round(args.neg_per_pos * len(tr_pos))))
        tr_neg_s = tr_neg.sample(n=n_keep, random_state=args.seed) if n_keep < len(tr_neg) else tr_neg
        tr_sub = pd.concat([tr_pos, tr_neg_s]).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

        tr_rows = np.array([id2row[t] for t in tr_sub["tagged_id"]])
        tr_y = torch.tensor(tr_sub["_y"].values.astype(float), device=dev)
        tr_w = torch.tensor(tr_sub["label_confidence"].map(confidence_to_weight).astype(float).values, device=dev)
        Xtr, Mtr = gather_topk(tr_rows)   # CPU fp16
        eff_pos = float(tr_w[tr_y == 1].sum()); eff_neg = float(tr_w[tr_y == 0].sum())
        pw = torch.tensor(eff_neg / max(eff_pos, 1.0), device=dev)

        # ---- train pairs (both endpoints train, locked scope)
        pp = pairs[pairs["class"] == PHENO]
        pp = pp[pp["ext_id"].isin(id2row) & pp["outgroup_id"].isin(id2row)]
        pp = pp[(pp["ext_id"].map(split_of) == "train") & (pp["outgroup_id"].map(split_of) == "train")]
        if args.scope == "secreted":
            pp = pp[pp["ext_id"].map(secr_of).fillna(False) & pp["outgroup_id"].map(secr_of).fillna(False)]
        Xte = Xto = Mte = Mto = None
        if len(pp):
            Xte, Mte = gather_topk(np.array([id2row[t] for t in pp["ext_id"]]))
            Xto, Mto = gather_topk(np.array([id2row[t] for t in pp["outgroup_id"]]))
        n_tp = 0 if Xte is None else Xte.shape[0]

        print("[attn] %s lam=%s scope=%s | train_sub=%d (pos=%d neg=%d, npp=%.1f) "
              "pw=%.2f train_pairs=%d | clean val pos=%d neg=%d base=%.4f val_pairs=%d"
              % (PHENO, lam, args.scope, len(tr_sub), len(tr_pos), n_keep, args.neg_per_pos,
                 float(pw), n_tp, len(va_pos), len(va_neg), base_rate, 0 if Xve is None else Xve.shape[0]),
              flush=True)

        torch.manual_seed(args.seed)
        head = build_pooling_head("attention", H, d_hidden=args.hidden,
                                  dropout=args.dropout, attn_dim=args.attn_dim).to(dev)
        opt = torch.optim.Adam(head.parameters(), lr=args.lr)
        bce = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
        ntr = Xtr.shape[0]
        best_auc = -1.0; best = None; best_ep = -1; hist = []
        odir = out / ("attn_%s" % PHENO); odir.mkdir(parents=True, exist_ok=True)

        def score_all(Xc, Mc):
            outs = []
            for i in range(0, Xc.shape[0], args.eval_batch_size):
                xb = Xc[i:i + args.eval_batch_size].to(dev).float()
                mb = Mc[i:i + args.eval_batch_size].to(dev).float()
                outs.append(head(xb, mb).float().cpu().numpy())
            return np.concatenate(outs) if outs else np.array([])

        for ep in range(args.epochs):
            head.train(); perm = torch.randperm(ntr)
            pp_cur = 0; pair_perm = torch.randperm(n_tp) if n_tp else None
            for i in range(0, ntr, args.batch_size):
                idx = perm[i:i + args.batch_size]
                xb = Xtr[idx].to(dev).float(); mb = Mtr[idx].to(dev).float()
                s = head(xb, mb)
                loss = (bce(s, tr_y[idx.to(dev)]) * tr_w[idx.to(dev)]).mean()
                if n_tp and lam > 0:
                    if pp_cur + args.pair_batch_size > n_tp:
                        pair_perm = torch.randperm(n_tp); pp_cur = 0
                    pj = pair_perm[pp_cur:pp_cur + args.pair_batch_size]; pp_cur += args.pair_batch_size
                    se = head(Xte[pj].to(dev).float(), Mte[pj].to(dev).float())
                    so = head(Xto[pj].to(dev).float(), Mto[pj].to(dev).float())
                    loss = loss + lam * torch.clamp(args.margin - (se - so), min=0).mean()
                opt.zero_grad(); loss.backward(); opt.step()

            head.eval()
            with torch.no_grad():
                sv = score_all(Xva, Mva)
                ap = float(average_precision_score(va_y, sv))
                auc = float(roc_auc_score(va_y, sv))
                pair_auc = pair_acc = float("nan")
                if Xve is not None and Xve.shape[0] > 0:
                    sve = score_all(Xve, Mve); svo = score_all(Xvo, Mvo)
                    yp = np.concatenate([np.ones(len(sve)), np.zeros(len(svo))])
                    sp = np.concatenate([sve, svo])
                    if len(np.unique(yp)) == 2:
                        pair_auc = float(roc_auc_score(yp, sp))
                    pair_acc = float(np.mean((sve > svo) + 0.5 * (sve == svo)))
            hist.append({"epoch": ep, "val_auroc": round(auc, 4), "val_auprc": round(ap, 4),
                         "val_pair_auc": (round(pair_auc, 4) if pair_auc == pair_auc else None),
                         "val_pair_acc": (round(pair_acc, 4) if pair_acc == pair_acc else None)})
            if auc > best_auc:
                best_auc = auc
                best = {"epoch": ep, "val_auroc": round(auc, 4), "val_auprc": round(ap, 4),
                        "val_pair_auc": (round(pair_auc, 4) if pair_auc == pair_auc else None),
                        "val_pair_acc": (round(pair_acc, 4) if pair_acc == pair_acc else None)}
                best_ep = ep
                torch.save(head.state_dict(), str(odir / "head_best.pt"))
                if Xve is not None:
                    with torch.no_grad():
                        ne = min(512, Xve.shape[0])
                        a = head.alpha(Xve[:ne].to(dev).float(), Mve[:ne].to(dev).float()).cpu().numpy()
                    np.save(str(odir / "alpha_ext_best.npy"), a.astype(np.float32))
            if ep % 5 == 0 or ep == args.epochs - 1:
                print("[attn] %-16s ep %3d AUROC %.4f AUPRC %.4f pair_auc %s"
                      % (PHENO, ep, auc, ap, hist[-1]["val_pair_auc"]), flush=True)

        rec = {"lam": lam, "pooling": "attention", "scope": args.scope, "tier": "hm",
               "best_epoch": best_ep, "val_auroc": best["val_auroc"], "val_auprc": best["val_auprc"],
               "val_pair_auc": best["val_pair_auc"], "val_pair_acc": best["val_pair_acc"],
               "clean_val_pos": int(len(va_pos)), "clean_val_neg": int(len(va_neg)),
               "base_rate": round(base_rate, 4), "train_n": int(ntr),
               "train_pos": int(len(tr_pos)), "train_neg": int(n_keep),
               "neg_per_pos": args.neg_per_pos, "pos_weight": round(float(pw), 2),
               "train_pairs": int(n_tp), "val_pairs": int(0 if Xve is None else Xve.shape[0]),
               "seconds": round(time.time() - t0, 1)}
        result["phenotypes"][PHENO] = rec
        (odir / "metrics.json").write_text(json.dumps(rec, indent=2))
        (odir / "history.json").write_text(json.dumps(hist, indent=2))
        (out / "attn_heads_summary.json").write_text(json.dumps(result, indent=2))
        print("[attn] %s BEST AUROC %.4f (AUPRC %.4f pair_auc %s) @ep %d (%.0fs)"
              % (PHENO, best["val_auroc"], best["val_auprc"], best["val_pair_auc"], best_ep, rec["seconds"]),
              flush=True)

        del Xtr, Mtr, Xva, Mva
        if Xte is not None: del Xte, Xto, Mte, Mto
        if Xve is not None: del Xve, Xvo, Mve, Mvo
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    print("[attn] === ATTENTION HEADS (locked secreted x H+M, best-lam) ===", flush=True)
    print("%-16s %6s %8s %8s %8s" % ("phenotype", "lam", "AUROC", "AUPRC", "pairAUC"), flush=True)
    for p, m in result["phenotypes"].items():
        print("%-16s %6s %8.4f %8.4f %8s"
              % (p, m["lam"], m["val_auroc"], m["val_auprc"], m["val_pair_auc"]), flush=True)
    print("[attn] written %s" % (out / "attn_heads_summary.json"), flush=True)
    print("DONE_ATTN_HEADS", flush=True)


if __name__ == "__main__":
    main()
