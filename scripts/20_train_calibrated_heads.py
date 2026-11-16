#!/usr/bin/env python
"""Stage-2b: calibrated mean-pool phenotype heads with taxonomy-grouped splits.

Supersedes ``19_train_meanpool_production.py``. Five changes, each targeting a
specific failure documented in ``docs/ranker_qc_and_recalibration.md``:

1. **Balanced sampling instead of extreme ``pos_weight``.**  The old loss used
   ``BCEWithLogitsLoss(pos_weight = n_neg/n_pos)``, which ranges from 12.9
   (halophile) to 975 (hyperthermophile).  A ``pos_weight`` of w shifts the
   decision point by ``+ln(w)`` logits -- up to 6.88 for hyperthermophile -- so
   the sigmoid output is not a probability and saturates at 1.0 for almost any
   input.  Here each epoch draws ``n_pos`` negatives without replacement from a
   pre-gathered pool, and ``pos_weight`` is dropped (set to 1).
2. **Post-hoc calibration.**  The balanced head is still trained at a 50/50
   prior, which is not the corpus prior.  A Platt rescale ``z' = a*z + b`` is
   fitted on a held-out *calibration* split at the **true** class prior, so the
   consumer (``11_generate_pipelineB.py``) gets a genuine probability.  ``a``
   and ``b`` ship inside ``head_best.pt`` under key ``calibration``.
3. **Taxonomy-grouped splits.**  The old split grouped on ``cluster_id50``,
   which blocks homolog leakage but not *lineage* leakage: the same genome (and
   the same family) appears on both sides through different clusters.  Splits
   here are assigned on a taxonomic rank (default ``family``), so no family is
   ever on two sides.  Cluster co-occurrence across splits is measured and
   reported as a diagnostic rather than assumed to be zero.
4. **Weight decay.**  ``AdamW`` with ``--weight-decay`` (default 1e-2) replaces
   plain ``Adam``, to penalise the high-norm directions a head uses to memorise
   a small number of lineages.
5. **Selection on calibrated log-loss.**  The old script kept the best-AUROC
   epoch.  AUROC is rank-only and is blind to the saturation in (1).  Here the
   calibration of (2) is refitted every epoch and the epoch minimising
   **val log-loss** is kept; AUROC/AUPRC/pair-AUC are still recorded.

Also emits ``negative_control_panel.json``: calibrated scores for a panel of
held-out confident-mesophile proteins, which is the diagnostic that would have
caught the saturation directly.

Reads the same cached mean-pool embeddings as stage 19 (``emb_shard*.npy`` +
``ids_shard*.txt``), so no GPU forward pass is needed; ``--device cpu`` is a
supported and tested configuration.
"""
import argparse, json, sys, copy, hashlib, re
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             log_loss, brier_score_loss)

sys.path.insert(0, "/groups/cress/projects/jaymin/eptrans_scratch/repo/src")
from eptrans.modeling.data import phenotype_binary_labels
from eptrans.modeling.losses import confidence_to_weight

DEFAULT_PHENOS = ["psychrophile", "thermophile", "hyperthermophile",
                  "acidophile", "alkaliphile", "halophile"]
LOCKED_LAM = {"psychrophile": 2.0, "thermophile": 1.0, "hyperthermophile": 4.0,
              "acidophile": 1.0, "alkaliphile": 2.0, "halophile": 0.5}
_PREFIX_RE = re.compile(r"^(GB_|RS_|CU_)")


def bare(acc: str) -> str:
    return _PREFIX_RE.sub("", str(acc))


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
    assert int(sum(sizes)) == n, "emb rows %d != ids %d" % (sum(sizes), n)
    return mm, shard_of, local_of, mm[0].shape[1]


def group_split(groups, seed, fracs=(0.70, 0.15, 0.15)):
    """Deterministic hashed assignment of whole taxonomic groups to splits.

    Hashing (rather than shuffling) makes the assignment stable when the group
    set grows -- adding genomes never reshuffles existing groups across splits.
    """
    names = ["train", "calib", "val"]
    cum = np.cumsum(fracs)
    out = {}
    for g in groups:
        h = hashlib.blake2b(f"{seed}:{g}".encode(), digest_size=8).digest()
        u = int.from_bytes(h, "big") / 2 ** 64
        out[g] = names[int(np.searchsorted(cum, u, side="right"))]
    return out


def fit_platt(logits, y, iters=300, lr=0.05):
    """Fit z' = a*z + b by minimising log-loss at the split's own class prior."""
    z = torch.tensor(np.asarray(logits, dtype=np.float64))
    t = torch.tensor(np.asarray(y, dtype=np.float64))
    a = torch.ones(1, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([a, b], lr=lr, max_iter=iters)
    lossf = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad(); l = lossf(a * z + b, t); l.backward(); return l
    opt.step(closure)
    return float(a.detach()), float(b.detach())


def ece(p, y, bins=15):
    """Expected calibration error, equal-width bins."""
    p = np.asarray(p); y = np.asarray(y, dtype=float)
    edges = np.linspace(0, 1, bins + 1); tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--labeled", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--taxonomy", required=True,
                    help="combined_labels.parquet (accession + rank columns)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scope", required=True, choices=["whole", "secreted"])
    ap.add_argument("--phenotypes", default=",".join(DEFAULT_PHENOS))
    ap.add_argument("--group-rank", default="family",
                    choices=["family", "genus", "order", "class", "phylum"])
    ap.add_argument("--min-groups", type=int, default=20,
                    help="fall back to the next coarser rank below this many groups")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--pair-batch-size", type=int, default=256)
    ap.add_argument("--neg-pool-mult", type=float, default=4.0,
                    help="negatives pre-gathered per positive; each epoch draws n_pos of them")
    ap.add_argument("--max-val-neg", type=int, default=1_000_000)
    ap.add_argument("--panel-size", type=int, default=20000,
                    help="held-out mesophile proteins per phenotype for the negative control")
    ap.add_argument("--seed", type=int, default=1466)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    dev = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ids = load_ids(args.cache_dir)
    id2row = {t: i for i, t in enumerate(ids)}
    mm, shard_of, local_of, Hd = open_emb(args.cache_dir, len(ids))
    print("[20] cache rows=%d dim=%d device=%s" % (len(ids), Hd, dev), flush=True)

    def gather(rows):
        r = np.asarray(rows); sh = shard_of[r]; lo = local_of[r]
        o = np.empty((len(r), Hd), dtype=np.float32)
        for j in np.unique(sh):
            sel = sh == j; o[sel] = mm[j][lo[sel]].astype(np.float32)
        return o

    df = pd.read_parquet(args.labeled)
    df = df[df["tagged_id"].isin(id2row)].copy()
    if args.scope == "secreted":
        df = df[df["is_secreted"].fillna(False).astype(bool)]
    df["acc"] = df["genome"].map(bare)

    tax = pd.read_parquet(args.taxonomy,
                          columns=["accession", "family", "genus", "order",
                                   "class", "phylum"])
    tax["acc"] = tax["accession"].map(bare)
    tax = tax.drop_duplicates("acc").set_index("acc")

    # Ordered fine -> coarse. The auto-coarsen loop below steps forward in this
    # list, so a mis-ordered entry would step to a FINER rank and make the group
    # shortage worse. Previously began ["family", "genus", ...], which coarsened
    # family -> genus; no shipped head was affected because every production run
    # passed an explicit --group-rank with enough groups to skip the loop.
    RANKS = ["genus", "family", "order", "class", "phylum"]
    rank = args.group_rank
    gmap = tax[rank].reindex(df["acc"].unique())
    ngroups = int(gmap.dropna().nunique())
    while ngroups < args.min_groups and RANKS.index(rank) < len(RANKS) - 1:
        rank = RANKS[RANKS.index(rank) + 1]
        gmap = tax[rank].reindex(df["acc"].unique())
        ngroups = int(gmap.dropna().nunique())
        print("[20] too few groups, coarsening to rank=%s (%d)" % (rank, ngroups), flush=True)
    # Genomes with no taxonomy get their own singleton group keyed on accession,
    # so they are never silently pooled into one giant "unknown" group that
    # would straddle splits.
    acc2grp = {a: (g if isinstance(g, str) and g else "UNK::" + a)
               for a, g in zip(df["acc"].unique(), gmap.values)}
    df["grp"] = df["acc"].map(acc2grp)
    split_of_grp = group_split(sorted(set(df["grp"])), args.seed)
    df["tsplit"] = df["grp"].map(split_of_grp)
    print("[20] group rank=%s groups=%d split sizes=%s"
          % (rank, df["grp"].nunique(), df["tsplit"].value_counts().to_dict()), flush=True)

    pairs = pd.read_csv(args.pairs, sep="\t", dtype=str)
    tsplit_of = dict(zip(df["tagged_id"], df["tsplit"]))
    results = {"scope": args.scope, "group_rank": rank, "seed": args.seed,
               "selection": "min val log-loss after per-epoch Platt refit",
               "sampling": "balanced (n_pos negatives/epoch), pos_weight=1",
               "phenotypes": {}}
    panel = {}

    for PHENO in args.phenotypes.split(","):
        y = phenotype_binary_labels(df, PHENO)
        sub = df.assign(_y=y); sub = sub[sub["_y"].notna()]
        hm = (sub["_y"] == 0) | (sub["label_confidence"].isin(["high", "medium"]))
        sub = sub[hm]

        tr = sub[sub["tsplit"] == "train"]; ca = sub[sub["tsplit"] == "calib"]
        va = sub[sub["tsplit"] == "val"]
        tr_pos = tr[tr["_y"] == 1]; tr_neg = tr[tr["_y"] == 0]
        n_pos = len(tr_pos)
        if n_pos < 50:
            print("[20] SKIP %s: only %d train positives" % (PHENO, n_pos), flush=True)
            continue
        n_pool = min(len(tr_neg), int(args.neg_pool_mult * n_pos))
        neg_idx = rng.choice(len(tr_neg), size=n_pool, replace=False)
        tr_neg = tr_neg.iloc[neg_idx]

        Xp = torch.tensor(gather([id2row[t] for t in tr_pos["tagged_id"]]))
        Xn = torch.tensor(gather([id2row[t] for t in tr_neg["tagged_id"]]))
        wp = torch.tensor(tr_pos["label_confidence"].map(confidence_to_weight)
                          .astype(float).values, dtype=torch.float32)

        def eval_split(s, cap):
            p = s[s["_y"] == 1]; n = s[s["_y"] == 0]
            frac = 1.0
            if len(n) > cap:
                frac = cap / len(n)
                n = n.iloc[rng.choice(len(n), size=cap, replace=False)]
            u = pd.concat([p, n])
            X = torch.tensor(gather([id2row[t] for t in u["tagged_id"]]))
            return X, u["_y"].values.astype(float), frac, len(p), len(n)

        Xca, yca, _, _, _ = eval_split(ca, args.max_val_neg // 2)
        Xva, yva, va_frac, n_vp, n_vn = eval_split(va, args.max_val_neg)
        base_rate = float(np.mean(yva))

        pp = pairs[pairs["class"] == PHENO]
        pp = pp[pp["ext_id"].isin(id2row) & pp["outgroup_id"].isin(id2row)]
        in_tr = ((pp["ext_id"].map(tsplit_of) == "train") &
                 (pp["outgroup_id"].map(tsplit_of) == "train"))
        in_va = ((pp["ext_id"].map(tsplit_of) == "val") &
                 (pp["outgroup_id"].map(tsplit_of) == "val"))
        ptr, pva = pp[in_tr], pp[in_va]
        Te = To = Ve = Vo = None
        if len(ptr):
            Te = torch.tensor(gather([id2row[t] for t in ptr["ext_id"]]))
            To = torch.tensor(gather([id2row[t] for t in ptr["outgroup_id"]]))
        if len(pva):
            Ve = torch.tensor(gather([id2row[t] for t in pva["ext_id"]]))
            Vo = torch.tensor(gather([id2row[t] for t in pva["outgroup_id"]]))

        # cluster leakage diagnostic: fraction of val clusters also seen in train
        cl_tr = set(tr["cluster_id50"]); cl_va = set(va["cluster_id50"])
        clus_overlap = len(cl_tr & cl_va) / max(len(cl_va), 1)

        lam = LOCKED_LAM.get(PHENO, 1.0)
        head = nn.Sequential(nn.Linear(Hd, args.hidden), nn.GELU(),
                             nn.Dropout(args.dropout), nn.Linear(args.hidden, 1)).to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
        bce = nn.BCEWithLogitsLoss(reduction="none")  # pos_weight deliberately absent
        best = {"val_logloss": float("inf")}; best_state = None; history = []

        def logits(X):
            head.eval(); o = []
            with torch.no_grad():
                for i in range(0, X.shape[0], 8192):
                    o.append(head(X[i:i + 8192].to(dev)).squeeze(-1).float().cpu().numpy())
            return np.concatenate(o) if o else np.zeros(0)

        for ep in range(args.epochs):
            head.train()
            sel = torch.tensor(rng.choice(n_pool, size=min(n_pos, n_pool), replace=False))
            Xe = torch.cat([Xp, Xn[sel]]); ye = torch.cat(
                [torch.ones(n_pos), torch.zeros(len(sel))])
            we = torch.cat([wp, torch.ones(len(sel))])
            perm = torch.randperm(len(ye))
            npair = 0 if Te is None else Te.shape[0]
            pc = 0; pperm = torch.randperm(npair) if npair else None
            for i in range(0, len(perm), args.batch_size):
                idx = perm[i:i + args.batch_size]
                s = head(Xe[idx].to(dev)).squeeze(-1)
                loss = (bce(s, ye[idx].to(dev)) * we[idx].to(dev)).mean()
                if npair and lam > 0:
                    if pc + args.pair_batch_size > npair:
                        pperm = torch.randperm(npair); pc = 0
                    pj = pperm[pc:pc + args.pair_batch_size]; pc += args.pair_batch_size
                    se = head(Te[pj].to(dev)).squeeze(-1)
                    so = head(To[pj].to(dev)).squeeze(-1)
                    loss = loss + lam * torch.clamp(args.margin - (se - so), min=0).mean()
                opt.zero_grad(); loss.backward(); opt.step()

            zc, zv = logits(Xca), logits(Xva)
            a, b = fit_platt(zc, yca)
            pv = 1.0 / (1.0 + np.exp(-(a * zv + b)))
            ll = float(log_loss(yva, np.clip(pv, 1e-7, 1 - 1e-7), labels=[0, 1]))
            auc = float(roc_auc_score(yva, zv)); apr = float(average_precision_score(yva, zv))
            pair_auc = None
            if Ve is not None:
                sve, svo = logits(Ve), logits(Vo)
                yp = np.concatenate([np.ones(len(sve)), np.zeros(len(svo))])
                pair_auc = float(roc_auc_score(yp, np.concatenate([sve, svo])))
            rec = {"epoch": ep, "val_logloss": round(ll, 5), "val_auroc": round(auc, 4),
                   "val_auprc": round(apr, 4), "val_pair_auc": (round(pair_auc, 4)
                                                               if pair_auc else None),
                   "platt_a": round(a, 4), "platt_b": round(b, 4)}
            history.append(rec)
            if ll < best["val_logloss"]:
                best = dict(rec); best.update({
                    "val_brier": round(float(brier_score_loss(yva, pv)), 5),
                    "val_ece": round(ece(pv, yva), 5)})
                best_state = copy.deepcopy(head.state_dict())
            print("[20] %s ep%02d logloss=%.5f auroc=%.4f auprc=%.4f pair=%s"
                  % (PHENO, ep, ll, auc, apr, pair_auc), flush=True)

        head.load_state_dict(best_state); head.eval()
        a, b = best["platt_a"], best["platt_b"]

        # ---- negative control panel: held-out confident mesophiles, never trained on
        mes = va[va["_y"] == 0]
        k = min(args.panel_size, len(mes))
        mp = mes.iloc[rng.choice(len(mes), size=k, replace=False)]
        zm = logits(torch.tensor(gather([id2row[t] for t in mp["tagged_id"]])))
        pm = 1.0 / (1.0 + np.exp(-(a * zm + b)))
        panel[PHENO] = {
            "n": int(k), "corpus_base_rate": round(base_rate, 5),
            "calibrated_p": {q: round(float(np.quantile(pm, v)), 5) for q, v in
                             [("p01", .01), ("p25", .25), ("p50", .50),
                              ("p75", .75), ("p95", .95), ("p99", .99), ("max", 1.0)]},
            "frac_above_0.5": round(float((pm > 0.5).mean()), 5),
            "frac_above_0.9": round(float((pm > 0.9).mean()), 5),
            "raw_logit": {"p50": round(float(np.median(zm)), 4),
                          "max": round(float(zm.max()), 4)}}

        sp = out / ("clf_%s" % PHENO); sp.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": {k_: v.detach().cpu() for k_, v in head.state_dict().items()},
                    "arch": {"in_dim": Hd, "hidden": args.hidden, "dropout": args.dropout,
                             "pooling": "mean",
                             "layers": "Linear(H,hidden)->GELU->Dropout->Linear(hidden,1)"},
                    "calibration": {"kind": "platt", "a": a, "b": b,
                                    "note": "p = sigmoid(a*logit + b); fitted on the "
                                            "calib split at its true class prior"},
                    "best_epoch": best["epoch"]}, str(sp / "head_best.pt"))
        m = dict(best)
        m.update({"pheno": PHENO, "scope": args.scope, "group_rank": rank, "lam": lam,
                  "train_pos": int(n_pos), "train_neg_pool": int(n_pool),
                  "sampling": "balanced", "pos_weight": 1.0,
                  "train_pairs": int(0 if Te is None else Te.shape[0]),
                  "val_pairs": int(0 if Ve is None else Ve.shape[0]),
                  "val_pos": int(n_vp), "val_neg": int(n_vn),
                  "val_neg_subsample_frac": round(va_frac, 5),
                  "base_rate": round(base_rate, 5),
                  "train_val_cluster_overlap_frac": round(clus_overlap, 4),
                  "weight_decay": args.weight_decay,
                  "saved_head": str(sp / "head_best.pt")})
        (sp / "metrics.json").write_text(json.dumps(m, indent=2))
        (sp / "history.json").write_text(json.dumps(history, indent=2))
        results["phenotypes"][PHENO] = m
        print("[20] %s DONE logloss=%.5f ece=%.4f auroc=%.4f mesophile p95=%.4f"
              % (PHENO, m["val_logloss"], m["val_ece"], m["val_auroc"],
                 panel[PHENO]["calibrated_p"]["p95"]), flush=True)

        del Xp, Xn, Xca, Xva
        if Te is not None: del Te, To
        if Ve is not None: del Ve, Vo
        import gc; gc.collect()

    (out / "calibrated_heads_all.json").write_text(json.dumps(results, indent=2))
    (out / "negative_control_panel.json").write_text(json.dumps(panel, indent=2))
    print("DONE_CALIBRATED_HEADS", flush=True)


if __name__ == "__main__":
    main()
