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
    ap.add_argument("--no-rubric-weights", dest="rubric_weights", action="store_false",
                    help="disable rubric-rank confidence sample weighting (default: on)")
    ap.set_defaults(rubric_weights=True)
    ap.add_argument("--pair-batch-size", type=int, default=256,
                    help="matched-pair sub-batch per step for the active margin term")
    ap.add_argument("--pointwise-scope", action="store_true",
                    help="INV-SCOPE-E: restrict the POINTWISE (BCE) term to each "
                         "phenotype's configured protein scope. The corpus admits a "
                         "genome's whole proteome when that genome is an ext member of "
                         "any whole_proteome-class pair (the INV-SCOPE-D union, keyed on "
                         "GENOME not (genome,class)). For a secreted-scope class this "
                         "silently labels that genome's cytoplasmic proteins y=1, so the "
                         "BCE term trains on whole-proteome positives while the margin "
                         "term stays secreted. Measured on the emitfix corpus: 33.6%% of "
                         "alkaliphile, 27.4%% of acidophile, 8.8%% of halophile train "
                         "positives are non-secreted; thermophile 0%%.")
    ap.add_argument("--scope-config", default=None,
                    help="path to config.yaml supplying dataset.protein_scope "
                         "(--pointwise-scope only; default: repo config)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--emb-device", default="auto",
                    help="where the embedding matrix lives: auto|cpu|cuda. "
                         "'auto' keeps it on GPU only if it fits in 70%% of free "
                         "VRAM, else CPU RAM with per-batch transfer (the ~18M x "
                         "2560 scoped corpus is ~185 GB fp32, so 'auto' picks cpu).")
    ap.add_argument("--seed", type=int, default=1466)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from sklearn.metrics import average_precision_score, roc_auc_score
    from eptrans.modeling.data import attach_sequences, phenotype_binary_labels
    from eptrans.modeling.losses import classifier_loss, confidence_to_weight

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X, ids = _load_cache(args.cache_dir)
    id2row = {t: i for i, t in enumerate(ids)}

    # Embedding-matrix placement. The scope-corrected corpus is ~18M x 2560, which
    # is ~185 GB fp32 / ~92 GB fp16 -- far past any single-GPU VRAM. Hold the matrix
    # in CPU RAM (the H200 node has ~2 TB) and move only the per-step indexed batch
    # to the GPU. --emb-device auto keeps it on-GPU only when it comfortably fits.
    emb_dev = args.emb_device
    if emb_dev == "auto":
        need_gb = X.shape[0] * X.shape[1] * 4 / 1e9  # fp32 tensor on device
        fits = False
        if args.device.startswith("cuda") and torch.cuda.is_available():
            free_b, _ = torch.cuda.mem_get_info()
            fits = need_gb < 0.7 * (free_b / 1e9)
        emb_dev = args.device if fits else "cpu"
        print(f"[10] emb matrix {need_gb:.0f} GB fp32 -> emb_device={emb_dev}", flush=True)
    Xt = torch.tensor(X, device=emb_dev)

    def gather(rows):
        # rows live on emb_dev; move the gathered batch to the compute device
        return Xt[rows].to(args.device, non_blocking=True) if emb_dev != args.device else Xt[rows]

    df = attach_sequences(pd.read_parquet(args.labeled), args.fasta)
    df = df[df["tagged_id"].astype(str).isin(id2row)].reset_index(drop=True)
    split_of = dict(zip(df["tagged_id"].astype(str), df["split"]))
    pairs = pd.read_csv(args.pairs, sep="\t", dtype=str)

    def rows_for(tagged_ids):
        return torch.tensor([id2row[t] for t in tagged_ids], device=emb_dev)

    # protein-scope map for --pointwise-scope (same config key stage 06 reads, so
    # the two terms cannot drift apart)
    secreted_col = "is_secreted"
    scope_by_class, default_scope = None, "secreted"
    if args.pointwise_scope:
        from eptrans.config import load_config
        cfg = load_config(args.scope_config)
        ps = cfg.get_path("dataset.protein_scope", {}) or {}
        scope_by_class = dict(ps.get("by_phenotype", {}) or {})
        default_scope = ps.get("default", "secreted")
        if not scope_by_class:
            raise SystemExit("--pointwise-scope given but config dataset.protein_scope."
                             "by_phenotype is empty")
        print(f"[10] pointwise scope active: default={default_scope} "
              f"by_phenotype={scope_by_class}", flush=True)

    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    summary = {}

    for pheno in args.phenotypes:
        y = phenotype_binary_labels(df, pheno)
        sub = df.assign(_y=y)
        sub = sub[sub["_y"].notna()]

        # INV-SCOPE-E: align the pointwise term with the pair term's scope.
        # _derive_protein_pairs builds its representative map from
        # lab[lab.is_secreted] for a secreted-scope class, so the margin term
        # only ever sees secreted proteins. The pointwise term above selects on
        # the LABEL alone, and the corpus contains whole proteomes for any genome
        # in the INV-SCOPE-D whole_ext union -- a union keyed on genome, not on
        # (genome, class). A soda-lake genome labelled `alkaliphile` that is also
        # an ext member of a psychrophile pair therefore contributes its entire
        # cytoplasm as alkaliphile positives. Verified on the emitfix corpus:
        # every leaking genome (alkaliphile 25/25, acidophile 21/21, halophile
        # 32/32) is in whole_ext; thermophile has none and leaks nothing.
        if args.pointwise_scope and scope_by_class:
            sc = scope_by_class.get(pheno, default_scope)
            if sc == "secreted":
                if secreted_col not in sub.columns:
                    raise SystemExit(
                        f"--pointwise-scope: {pheno} is secreted-scope but the labeled "
                        f"table has no {secreted_col!r} column")
                keep = sub[secreted_col].fillna(False).astype(bool)
                n_drop = int((~keep).sum())
                n_pos_drop = int(((~keep) & (sub["_y"] == 1)).sum())
                sub = sub[keep]
                print(f"[10] {pheno}: pointwise scope={sc} -> dropped {n_drop:,} rows "
                      f"({n_pos_drop:,} positives) outside scope", flush=True)
            else:
                print(f"[10] {pheno}: pointwise scope={sc} -> no filter", flush=True)
        tr = sub[sub["split"] == "train"]
        va = sub[sub["split"] == "val"]
        tr_rows = rows_for(tr["tagged_id"].astype(str).tolist())
        tr_y = torch.tensor(tr["_y"].values, dtype=torch.float, device=emb_dev)
        va_rows = rows_for(va["tagged_id"].astype(str).tolist())
        va_y = va["_y"].values.astype(float)

        # RUBRIC-RANK sample weights w_i (design §12): a genome's confidence tier
        # (high/medium/low, or 'none' for a confident mesophile negative) maps to a
        # per-protein weight via CONFIDENCE_WEIGHTS {high 1.0, medium 0.5, none 1.0,
        # low 0.25}. This down-weights low-confidence positives (e.g. psychrophile_low
        # cold calls) so a weak label contributes a quarter of a high-confidence one,
        # rather than being dropped or counted equally. Applied on TOP of pos_weight
        # (which handles class imbalance). --no-rubric-weights reverts to uniform w_i.
        if args.rubric_weights:
            tr_w = torch.tensor(
                tr["label_confidence"].map(confidence_to_weight).astype(float).values,
                dtype=torch.float, device=emb_dev)
        else:
            tr_w = None

        # matched pairs (val split), ids -> cached rows
        pp = pairs[pairs["class"] == pheno] if "class" in pairs.columns else pairs.iloc[0:0]
        pp = pp[pp["ext_id"].astype(str).isin(id2row) & pp["outgroup_id"].astype(str).isin(id2row)]
        vpp = pp[(pp["ext_id"].map(split_of) == "val") & (pp["outgroup_id"].map(split_of) == "val")]
        tpp = pp[(pp["ext_id"].map(split_of) == "train") & (pp["outgroup_id"].map(split_of) == "train")]
        n_pos = int((tr_y == 1).sum())
        pw = float((tr_y == 0).sum()) / max(n_pos, 1)
        wdesc = "rubric" if args.rubric_weights else "uniform"
        print(f"[10] {pheno}: train {len(tr):,} (pos {n_pos:,}) / val {len(va):,} "
              f"| train pairs {len(tpp):,} / val pairs {len(vpp):,} | pos_weight {pw:.1f} "
              f"| lam {args.lam} | w_i {wdesc}",
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
            perm = torch.randperm(n_tr, device=emb_dev)
            pair_perm = torch.randperm(n_tp, device=emb_dev) if n_tp else None
            pp_cur = 0
            tot = 0.0
            for i in range(0, n_tr, args.batch_size):
                idx = perm[i:i + args.batch_size]
                s = head(gather(tr_rows[idx])).squeeze(-1)
                pe = po = None
                if n_tp:
                    if pp_cur + pair_bs > n_tp:          # cycle when exhausted
                        pair_perm = torch.randperm(n_tp, device=emb_dev); pp_cur = 0
                    pidx = pair_perm[pp_cur:pp_cur + pair_bs]; pp_cur += pair_bs
                    pe = head(gather(tr_ext[pidx])).squeeze(-1)
                    po = head(gather(tr_out[pidx])).squeeze(-1)
                yb = tr_y[idx].to(args.device)
                wb = tr_w[idx].to(args.device) if tr_w is not None else None
                loss, _ = classifier_loss(s, yb, pos_weight=pw_t,
                                          sample_weight=wb,
                                          pair_ext=pe, pair_out=po,
                                          lam=args.lam, margin=args.margin)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss.detach()) * len(idx)
            hist["train_loss"].append(tot / n_tr)

            head.eval()
            with torch.no_grad():
                vs = head(gather(va_rows)).squeeze(-1)
                au = (float(average_precision_score(va_y, torch.sigmoid(vs).cpu().numpy()))
                      if len(set(va_y.tolist())) > 1 else float("nan"))
                pa = pau = float("nan")
                if ext_rows is not None and len(ext_rows) > 0:
                    se = head(gather(ext_rows)).squeeze(-1).cpu().numpy()
                    so = head(gather(out_rows)).squeeze(-1).cpu().numpy()
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
                                  "n_val_pairs": int(len(vpp)),
                                  "lam": args.lam, "margin": args.margin,
                                  "rubric_weights": bool(args.rubric_weights),
                                  "n_train": int(len(tr)), "n_train_pos": n_pos,
                                  "pos_weight": pw, "n_train_pairs": n_tp}
            json.dump(hist, open(odir / "history.json", "w"), indent=2)
        json.dump(summary.get(pheno, {}), open(odir / "metrics.json", "w"), indent=2)

    json.dump(summary, open(Path(args.out_root) / "cached_probe_summary.json", "w"), indent=2)
    print(f"[10] ALL DONE -> {args.out_root}/cached_probe_summary.json")
    for p, m in summary.items():
        print(f"[10] {p:17} auprc {m['val_auprc']:.4f} pair_auc {m['val_pair_auc']:.4f} "
              f"(ep {m['epoch']}, {m['n_val_pairs']} val pairs)")


if __name__ == "__main__":
    main()
