#!/usr/bin/env python
"""Pooling ablation: mean vs attention vs top-k MIL on cached top-k residue features.

Question. The psychrophile head reached val AUPRC 0.572 / pair-AUC 0.635 while
thermophile reached 0.909 / 0.931. One candidate explanation is biological (cold
adaptation is structural/local and partly non-proteomic), another is
METHODOLOGICAL: both heads read a masked-MEAN-pooled embedding, and mean pooling
is near-sufficient for compositional thermophile adaptation but dilutes a signal
carried by ~10-30 active-site residues out of ~330 by 10-30x.

Design. Train the three pooling operators on the SAME cached (n, K, H) top-k
tensor from 09b, with an identical MLP readout, identical loss (pointwise BCE +
taxonomy-controlled pair margin), identical rubric-rank sample weights, and
identical seeds. The only thing that varies is how residues are combined, so a
delta is attributable to the pooling hypothesis.

Read-out. Run BOTH psychrophile (locality predicted to help) and thermophile
(mean predicted to be already near-optimal) so the result is a CROSSOVER test
rather than a single number:
  helps psychrophile only  -> locality confirmed; adopt per-class pooling
  helps both               -> mean pooling was lossy everywhere; full re-embed
  helps neither            -> ceiling is label noise / dynamics, not pooling
`mean` here is also a positive control: it should reproduce the stage-10 number
for the same class to within noise, since it consumes k=32 selected residues
rather than all of them.

Metrics match 10_train_cached_probe.py exactly (val AUPRC; pair_acc = within-pair
wins with ties at 0.5; pair_auc = pooled ext-vs-meso ROC-AUC) so numbers are
comparable to the lambda sweep. Attention alpha is also dumped for the best epoch
-- that is the interpretable evidence about WHERE the signal sits.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _load_topk_cache(cache_dir: str):
    """Concatenate topk/mean/lens/ids shards written by 09b."""
    d = Path(cache_dir)
    shards = sorted(int(p.stem.replace("topk_shard", ""))
                    for p in d.glob("topk_shard*.npy"))
    if not shards:
        raise SystemExit(f"no topk_shard*.npy in {cache_dir}")
    tk, mn, ln, ids = [], [], [], []
    for s in shards:
        tk.append(np.load(d / f"topk_shard{s}.npy"))
        mn.append(np.load(d / f"mean_shard{s}.npy"))
        ln.append(np.load(d / f"lens_shard{s}.npy"))
        ids += (d / f"ids_shard{s}.txt").read_text().split()
    TK = np.concatenate(tk); MN = np.concatenate(mn); LN = np.concatenate(ln)
    if not (len(TK) == len(MN) == len(LN) == len(ids)):
        raise SystemExit(f"cache length mismatch: topk={len(TK)} mean={len(MN)} "
                         f"lens={len(LN)} ids={len(ids)}")
    return TK, MN, LN, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True, help="output of 09b")
    ap.add_argument("--labeled", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--phenotypes", nargs="+", default=["psychrophile", "thermophile"])
    ap.add_argument("--poolings", nargs="+",
                    default=["mean", "attention", "topk_mil"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--mil-k", type=int, default=8)
    ap.add_argument("--attn-dim", type=int, default=128)
    ap.add_argument("--lam", type=float, default=0.5,
                    help="pair margin weight; 0.5 is the operating point from the sweep")
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--no-rubric-weights", dest="rubric_weights",
                    action="store_false", default=True)
    ap.add_argument("--pair-batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1466)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()

    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score
    from eptrans.modeling.data import phenotype_binary_labels
    from eptrans.modeling.losses import classifier_loss, confidence_to_weight
    from eptrans.modeling.pooling import build_pooling_head

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    TK, MN, LN, ids = _load_topk_cache(args.cache_dir)
    n, K, H = TK.shape
    id2row = {t: i for i, t in enumerate(ids)}
    print(f"[10b] cache: {n:,} proteins x K={K} x H={H} "
          f"({TK.nbytes/1e9:.2f} GB fp16)", flush=True)

    # (n,K,H) fp16 stays on CPU; only the indexed batch is moved+upcast per step.
    Xt = torch.from_numpy(TK)
    # validity mask per (protein, slot): slot j is real iff j < min(K, true_len)
    kv = np.minimum(LN, K)
    Mt = torch.from_numpy((np.arange(K)[None, :] < kv[:, None]).astype(np.float32))

    df = pd.read_parquet(args.labeled)
    df = df[df["tagged_id"].astype(str).isin(id2row)].reset_index(drop=True)
    pairs = pd.read_csv(args.pairs, sep="\t", dtype=str)
    print(f"[10b] labeled rows in cache: {len(df):,}", flush=True)

    def rows_for(tids):
        return torch.tensor([id2row[t] for t in tids], dtype=torch.long)

    def gather(rows):
        return (Xt[rows].to(args.device).float(),
                Mt[rows].to(args.device))

    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    for pheno in args.phenotypes:
        y = phenotype_binary_labels(df, pheno)
        sub = df.assign(_y=y)
        sub = sub[sub["_y"].notna()]
        tr = sub[sub["split"] == "train"]; va = sub[sub["split"] == "val"]
        if len(tr) == 0 or len(va) == 0:
            print(f"[10b] {pheno}: SKIP (train={len(tr)} val={len(va)})", flush=True)
            continue
        tr_rows = rows_for(tr["tagged_id"].astype(str).tolist())
        tr_y = torch.tensor(tr["_y"].values, dtype=torch.float, device=args.device)
        va_rows = rows_for(va["tagged_id"].astype(str).tolist())
        va_y = np.asarray(va["_y"].values, dtype=float)

        w = None
        if args.rubric_weights and "label_confidence" in tr.columns:
            w = torch.tensor(
                [confidence_to_weight(c) for c in tr["label_confidence"]],
                dtype=torch.float, device=args.device)

        npos = float(tr_y.sum()); nneg = float(len(tr_y) - npos)
        pos_weight = torch.tensor(max(nneg / max(npos, 1.0), 1.0),
                                  device=args.device)

        # taxonomy-matched validation pairs, both members present in the cache
        pp = pairs[pairs["class"] == pheno]
        pe = pp["ext_id"].astype(str); po = pp["outgroup_id"].astype(str)
        ok = [(a, b) for a, b in zip(pe, po)
              if a in id2row and b in id2row]
        ext_rows = rows_for([a for a, _ in ok]) if ok else None
        out_rows = rows_for([b for _, b in ok]) if ok else None
        print(f"[10b] {pheno}: train {len(tr):,} (pos {int(npos):,}) "
              f"val {len(va):,} pairs {len(ok):,} pos_weight {float(pos_weight):.3f}",
              flush=True)

        for kind in args.poolings:
            t0 = time.time()
            torch.manual_seed(args.seed)   # identical init across arms
            head = build_pooling_head(kind, H, d_hidden=args.hidden,
                                      dropout=args.dropout, mil_k=args.mil_k,
                                      attn_dim=args.attn_dim).to(args.device)
            opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
            best = -1.0; best_ep = -1; hist = []
            odir = Path(args.out_root) / f"{pheno}__{kind}"
            odir.mkdir(parents=True, exist_ok=True)

            for ep in range(args.epochs):
                head.train()
                perm = torch.randperm(len(tr_rows))
                tot = 0.0
                for i in range(0, len(perm), args.batch_size):
                    idx = perm[i:i + args.batch_size]
                    xb, mb = gather(tr_rows[idx])
                    yb = tr_y[idx]
                    wb = w[idx] if w is not None else None
                    s = head(xb, mb)
                    # pair term: sample matched pairs each step (same as stage 10)
                    se = so = None
                    if ext_rows is not None and args.lam > 0:
                        j = torch.randint(0, len(ext_rows),
                                          (min(args.pair_batch_size, len(ext_rows)),))
                        xe, me = gather(ext_rows[j]); xo, mo = gather(out_rows[j])
                        se = head(xe, me); so = head(xo, mo)
                    loss = classifier_loss(s, yb, sample_weight=wb,
                                           pos_weight=pos_weight,
                                           pair_ext=se, pair_out=so,
                                           lam=args.lam, margin=args.margin)
                    opt.zero_grad(); loss.backward(); opt.step()
                    tot += float(loss.detach()) * len(idx)

                head.eval()
                with torch.no_grad():
                    xv, mv = gather(va_rows)
                    vs = head(xv, mv)
                    au = (float(average_precision_score(
                              va_y, torch.sigmoid(vs).cpu().numpy()))
                          if len(set(va_y.tolist())) > 1 else float("nan"))
                    pa = pau = float("nan")
                    if ext_rows is not None and len(ext_rows) > 0:
                        xe, me = gather(ext_rows); xo, mo = gather(out_rows)
                        se_ = head(xe, me).cpu().numpy()
                        so_ = head(xo, mo).cpu().numpy()
                        pa = float(np.mean((se_ > so_) + 0.5 * (se_ == so_)))
                        yy = [1] * len(se_) + [0] * len(so_)
                        pau = float(roc_auc_score(
                            yy, np.concatenate([se_, so_])))
                hist.append({"epoch": ep, "train_loss": tot / max(len(tr_rows), 1),
                             "val_auprc": au, "val_pair_acc": pa, "val_pair_auc": pau})
                if au == au and au > best:
                    best, best_ep = au, ep
                    torch.save(head.state_dict(), str(odir / "head_best.pt"))
                    summary[f"{pheno}__{kind}"] = {
                        "phenotype": pheno, "pooling": kind, "epoch": ep,
                        "val_auprc": au, "val_pair_acc": pa, "val_pair_auc": pau,
                        "n_val_pairs": len(ok), "lam": args.lam,
                        "rubric_weights": bool(args.rubric_weights),
                        "pos_weight": float(pos_weight), "K": int(K),
                        "mil_k": args.mil_k if kind == "topk_mil" else None}
                    # attention alpha at the best epoch = the interpretable artefact
                    if kind == "attention" and ext_rows is not None:
                        with torch.no_grad():
                            xe, me = gather(ext_rows[:min(512, len(ext_rows))])
                            a = head.alpha(xe, me).cpu().numpy()
                        np.save(str(odir / "alpha_ext_best.npy"), a.astype(np.float32))
                if ep % 5 == 0 or ep == args.epochs - 1:
                    print(f"[10b] {pheno:14s} {kind:10s} ep {ep:3d} "
                          f"auprc {au:.4f} pair_acc {pa:.4f} pair_auc {pau:.4f}",
                          flush=True)

            (odir / "history.json").write_text(json.dumps(hist, indent=1))
            (odir / "metrics.json").write_text(
                json.dumps(summary[f"{pheno}__{kind}"], indent=1))
            print(f"[10b] {pheno} {kind} BEST auprc {best:.4f} @ep {best_ep} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    out = Path(args.out_root) / "ablation_summary.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"[10b] wrote {out}", flush=True)
    # crossover table
    print("\n[10b] === POOLING ABLATION ===", flush=True)
    print(f"{'phenotype':16s} {'pooling':11s} {'AUPRC':>8s} {'pairAUC':>8s} {'pairAcc':>8s}",
          flush=True)
    for k, v in summary.items():
        print(f"{v['phenotype']:16s} {v['pooling']:11s} {v['val_auprc']:8.4f} "
              f"{v['val_pair_auc']:8.4f} {v['val_pair_acc']:8.4f}", flush=True)


if __name__ == "__main__":
    main()
