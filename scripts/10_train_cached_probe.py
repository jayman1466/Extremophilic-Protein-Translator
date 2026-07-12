#!/usr/bin/env python
"""Train per-phenotype classifier heads on CACHED backbone embeddings.

Stage 2, frozen-backbone path (design #1). Consumes the cache written by
09_embed_secretome.py (emb_shard*.npy + ids_shard*.txt) and trains a head that is
architecturally identical to MeanPoolClassifierHead's MLP (it operates on the
already-mean-pooled vector, so it's just the 2-layer MLP part). Because the
backbone is frozen and precomputed, an epoch is seconds not days — so we drop the
neg_per_pos subsampling and train on ALL negatives, and do all 5 phenotypes in
one job.

Training loss = weighted BCE + ACTIVE matched-pair margin (design "alternative":
cache once, pair-aware). A train-split matched-pair sub-batch is scored each step
and the margin term max(0, margin - (s_ext - s_out)) is added — the anti-taxonomy
mechanism, operating on the head's readout of the frozen features (the pairs push
the head toward a taxonomy-invariant direction that exists in the fixed embedding).
This is the same L_pair as the end-to-end path, minus the ability to reshape the
representation.

Metrics per phenotype match the end-to-end path:
  * pointwise val AUPRC (average precision) — the §12 selection metric.
  * taxonomy-controlled pair metrics (pair_acc / pair_auc / margin_gap) on
    held-out matched pairs — same definition as train.evaluate_pair_metrics,
    computed here on cached vectors.

Writes per phenotype under <out-root>/clf_<pheno>_cached/:
  head_best.pt         (best-AUPRC MLP state_dict — loadable as MeanPoolClassifierHead.net)
  history.json         (train_loss / val_auprc / val_pair_acc / val_pair_auc traces)
  metrics.json         (final best-epoch summary)

Head-only: this path produces the DISCRIMINATIVE probe. For generation the frozen
(MLM-adapted) backbone + this head is the coherent (adapter, head) pair, since the
adapter is fixed by construction.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _load_cache(cache_dir: str):
    d = Path(cache_dir)
    embs, ids = [], []
    shards = sorted(d.glob("emb_shard*.npy"),
                    key=lambda p: int(p.stem.replace("emb_shard", "")))
    if not shards:
        raise FileNotFoundError(f"no emb_shard*.npy under {cache_dir}")
    for ep in shards:
        s = ep.stem.replace("emb_shard", "")
        ip = d / f"ids_shard{s}.txt"
        e = np.load(ep)
        i = ip.read_text().split("\n")
        i = [x for x in i if x]
        assert len(i) == e.shape[0], f"shard {s}: {len(i)} ids vs {e.shape[0]} rows"
        embs.append(e)
        ids.extend(i)
    X = np.concatenate(embs, axis=0)
    print(f"[10] cache: {X.shape[0]:,} vectors x {X.shape[1]} dim from {len(shards)} shards")
    return X.astype(np.float32), ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--labeled", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--fasta", required=True, help="only to attach sequences for pair lookup ids")
    ap.add_argument("--phenotypes", nargs="+",
                    default=["acidophile", "alkaliphile", "halophile",
                             "thermophile", "hyperthermophile"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lam", type=float, default=1.0, help="pair margin loss weight")
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--pair-batch-size", type=int, default=256,
                    help="matched-pair sub-batch per step for the active margin term")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1466)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from sklearn.metrics import average_precision_score, roc_auc_score
    from eptrans.modeling.data import attach_sequences, phenotype_binary_labels
    from eptrans.modeling.losses import classifier_loss

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X, ids = _load_cache(args.cache_dir)
    id2row = {t: i for i, t in enumerate(ids)}
    Xt = torch.tensor(X, device=args.device)

    df = attach_sequences(pd.read_parquet(args.labeled), args.fasta)
    df = df[df["tagged_id"].astype(str).isin(id2row)].reset_index(drop=True)
    split_of = dict(zip(df["tagged_id"].astype(str), df["split"]))
    pairs = pd.read_csv(args.pairs, sep="\t", dtype=str)

    def rows_for(tagged_ids):
        return torch.tensor([id2row[t] for t in tagged_ids], device=args.device)

    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    summary = {}

    for pheno in args.phenotypes:
        y = phenotype_binary_labels(df, pheno)
        sub = df.assign(_y=y)
        sub = sub[sub["_y"].notna()]
        tr = sub[sub["split"] == "train"]
        va = sub[sub["split"] == "val"]
        tr_rows = rows_for(tr["tagged_id"].astype(str).tolist())
        tr_y = torch.tensor(tr["_y"].values, dtype=torch.float, device=args.device)
        va_rows = rows_for(va["tagged_id"].astype(str).tolist())
        va_y = va["_y"].values.astype(float)

        # matched pairs (val split), ids -> cached rows
        pp = pairs[pairs["class"] == pheno] if "class" in pairs.columns else pairs.iloc[0:0]
        pp = pp[pp["ext_id"].astype(str).isin(id2row) & pp["outgroup_id"].astype(str).isin(id2row)]
        vpp = pp[(pp["ext_id"].map(split_of) == "val") & (pp["outgroup_id"].map(split_of) == "val")]
        tpp = pp[(pp["ext_id"].map(split_of) == "train") & (pp["outgroup_id"].map(split_of) == "train")]
        n_pos = int((tr_y == 1).sum())
        pw = float((tr_y == 0).sum()) / max(n_pos, 1)
        print(f"[10] {pheno}: train {len(tr):,} (pos {n_pos:,}) / val {len(va):,} "
              f"| train pairs {len(tpp):,} / val pairs {len(vpp):,} | pos_weight {pw:.1f}",
              flush=True)
        if n_pos == 0:
            print(f"[10] {pheno}: no positives, skipping"); continue

        head = nn.Sequential(nn.Linear(X.shape[1], args.hidden), nn.GELU(),
                             nn.Dropout(args.dropout), nn.Linear(args.hidden, 1)).to(args.device)
        opt = torch.optim.Adam(head.parameters(), lr=args.lr)
        pw_t = torch.tensor(pw, device=args.device)

        ext_rows = rows_for(vpp["ext_id"].astype(str).tolist()) if len(vpp) else None
        out_rows = rows_for(vpp["outgroup_id"].astype(str).tolist()) if len(vpp) else None
        # TRAIN pairs -> active margin term (the anti-taxonomy mechanism). A pair
        # sub-batch is drawn each step and cycled if fewer pairs than steps; the
        # margin pushes s_ext > s_out on taxonomy-matched orthologs so the head
        # finds a taxonomy-invariant direction in the frozen features.
        tr_ext = rows_for(tpp["ext_id"].astype(str).tolist()) if len(tpp) else None
        tr_out = rows_for(tpp["outgroup_id"].astype(str).tolist()) if len(tpp) else None
        n_tp = int(tr_ext.shape[0]) if tr_ext is not None else 0

        hist = {"train_loss": [], "val_auprc": [], "val_pair_acc": [], "val_pair_auc": []}
        best = -1.0
        odir = Path(args.out_root) / f"clf_{pheno}_cached"
        odir.mkdir(parents=True, exist_ok=True)
        n_tr = tr_rows.shape[0]
        pair_bs = args.pair_batch_size
        for ep in range(args.epochs):
            head.train()
            perm = torch.randperm(n_tr, device=args.device)
            pair_perm = torch.randperm(n_tp, device=args.device) if n_tp else None
            pp_cur = 0
            tot = 0.0
            for i in range(0, n_tr, args.batch_size):
                idx = perm[i:i + args.batch_size]
                s = head(Xt[tr_rows[idx]]).squeeze(-1)
                pe = po = None
                if n_tp:
                    if pp_cur + pair_bs > n_tp:          # cycle when exhausted
                        pair_perm = torch.randperm(n_tp, device=args.device); pp_cur = 0
                    pidx = pair_perm[pp_cur:pp_cur + pair_bs]; pp_cur += pair_bs
                    pe = head(Xt[tr_ext[pidx]]).squeeze(-1)
                    po = head(Xt[tr_out[pidx]]).squeeze(-1)
                loss, _ = classifier_loss(s, tr_y[idx], pos_weight=pw_t,
                                          pair_ext=pe, pair_out=po,
                                          lam=args.lam, margin=args.margin)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss.detach()) * len(idx)
            hist["train_loss"].append(tot / n_tr)

            head.eval()
            with torch.no_grad():
                vs = head(Xt[va_rows]).squeeze(-1)
                au = (float(average_precision_score(va_y, torch.sigmoid(vs).cpu().numpy()))
                      if len(set(va_y.tolist())) > 1 else float("nan"))
                pa = pau = float("nan")
                if ext_rows is not None and len(ext_rows) > 0:
                    se = head(Xt[ext_rows]).squeeze(-1).cpu().numpy()
                    so = head(Xt[out_rows]).squeeze(-1).cpu().numpy()
                    wins = np.mean((se > so) + 0.5 * (se == so))
                    pa = float(wins)
                    yy = [1] * len(se) + [0] * len(so)
                    pau = float(roc_auc_score(yy, np.concatenate([se, so])))
            hist["val_auprc"].append((ep, au))
            hist["val_pair_acc"].append((ep, pa))
            hist["val_pair_auc"].append((ep, pau))
            print(f"[10] {pheno} epoch {ep} auprc {au:.4f} pair_acc {pa:.4f} "
                  f"pair_auc {pau:.4f}", flush=True)
            if au == au and au > best:  # au==au: not NaN
                best = au
                torch.save(head.state_dict(), str(odir / "head_best.pt"))
                summary[pheno] = {"epoch": ep, "val_auprc": au,
                                  "val_pair_acc": pa, "val_pair_auc": pau,
                                  "n_val_pairs": int(len(vpp))}
            json.dump(hist, open(odir / "history.json", "w"), indent=2)
        json.dump(summary.get(pheno, {}), open(odir / "metrics.json", "w"), indent=2)

    json.dump(summary, open(Path(args.out_root) / "cached_probe_summary.json", "w"), indent=2)
    print(f"[10] ALL DONE -> {args.out_root}/cached_probe_summary.json")
    for p, m in summary.items():
        print(f"[10] {p:17} auprc {m['val_auprc']:.4f} pair_auc {m['val_pair_auc']:.4f} "
              f"(ep {m['epoch']}, {m['n_val_pairs']} val pairs)")


if __name__ == "__main__":
    main()
