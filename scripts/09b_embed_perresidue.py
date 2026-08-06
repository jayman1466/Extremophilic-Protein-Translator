#!/usr/bin/env python
"""Cache PER-RESIDUE (top-k) backbone features for the pooling ablation.

Why this exists
---------------
09_embed_secretome.py masked-MEAN-pools each protein to one vector (line ~116)
and caches that. Mean pooling is an implicit hypothesis: the adaptive signal is
UNIFORM over residues. That is very nearly the sufficient statistic for
thermophile adaptation (IVYWREL frequency, charge fraction, E+K enrichment are
literally residue averages), and there mean pooling also DENOISES.

Cold adaptation is believed to be the opposite shape: catalytic-rate adaptation
at low temperature comes from increased LOCAL flexibility in and around the
active site, while distal surface residues look mesophile-like. If the signal
lives in ~10-30 residues of a ~330 aa protein, masked mean dilutes it by 10-30x.
Testing that requires features that PRESERVE locality, which the existing cache
has already thrown away.

What is cached
--------------
Storing all residues is not an option at corpus scale: 18.06M proteins x ~327 aa
x 2560 d x fp16 = ~30 TB. This script caches a fixed-size TOP-K token summary
per protein instead (k=32 default):

  topk_shard{i}.npy   (n, k, H) float16   -- k selected residue vectors
  mean_shard{i}.npy   (n, H)    float16   -- masked mean, IDENTICAL to stage 09
  lens_shard{i}.npy   (n,)      int32     -- true residue count (pre-truncation)
  ids_shard{i}.txt                        -- tagged_ids, same order

At k=32 that is ~22 GB for the psychrophile-relevant subset -- tractable -- and
carrying the mean in the SAME file means the three arms (mean / attention / MIL
top-k) train on features from one identical forward pass, so any AUPRC delta is
attributable to the pooling operator and not to a different embedding run.

Selection of the k residues is deliberately LABEL-FREE (see --select): we have no
active-site annotations, so choosing by any label-dependent criterion would leak.
  norm    (default) : largest L2 norm. Unsupervised saliency proxy -- ESM gives
                      high-norm vectors to residues in constrained/unusual local
                      environments, which is where catalytic machinery sits.
  stride            : evenly spaced positions. A null control -- if `norm` beats
                      `stride`, selection is doing work; if not, any gain is just
                      from having k vectors instead of 1.
Proteins shorter than k are zero-padded and the true length recorded in lens, so
downstream pooling can mask padding exactly.

Usage (one shard):
  python scripts/09b_embed_perresidue.py \
    --labeled $W/labeled_dataset.parquet --fasta $W/corpus_all.faa \
    --mlm-adapter $MODEL_ROOT/mlm_adapt/mlm_adapter_best \
    --phenotypes psychrophile thermophile --pairs-only \
    --pairs $W/labeled_dataset_protein_pairs.tsv \
    --topk 32 --select norm --shard 0 --nshards 4 \
    --out-dir $EMB_ROOT/perresidue_k32
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--mlm-adapter", required=True)
    ap.add_argument("--backbone-size", default="3B")
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--full-attention", action="store_true", default=True)
    ap.add_argument("--qv-only", dest="full_attention", action="store_false")
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--select", choices=["norm", "stride"], default="norm",
                    help="label-free rule for which k residues to keep")
    ap.add_argument("--phenotypes", nargs="+", default=None,
                    help="restrict to proteins whose label is in this set (plus "
                         "mesophile partners when --pairs-only)")
    ap.add_argument("--pairs-only", action="store_true",
                    help="restrict to proteins appearing in the pair table for the "
                         "selected phenotypes -- the ablation only needs the paired "
                         "evaluation set, which is what makes k=32 affordable")
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1022)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    import torch
    from eptrans.modeling.model import (build_lora_backbone,
                                        load_mlm_adapter_into_classifier)
    from eptrans.modeling.data import attach_sequences
    from eptrans.modeling.train import _encode

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tk_path = out / f"topk_shard{args.shard}.npy"
    mn_path = out / f"mean_shard{args.shard}.npy"
    ln_path = out / f"lens_shard{args.shard}.npy"
    ids_path = out / f"ids_shard{args.shard}.txt"
    done_path = out / f".done_shard{args.shard}"
    if done_path.exists() and tk_path.exists() and ids_path.exists():
        print(f"[09b] shard {args.shard} already complete -> skipping", flush=True)
        return

    df = pd.read_parquet(args.labeled)
    n_all = len(df)

    # ---- restrict the protein set BEFORE embedding (this is the cost control) ----
    if args.pairs_only:
        if not args.pairs:
            raise SystemExit("--pairs-only requires --pairs")
        pr = pd.read_csv(args.pairs, sep="\t", dtype=str)
        if args.phenotypes:
            pr = pr[pr["class"].isin(args.phenotypes)]
        keep_ids = set(pr["ext_id"].dropna()) | set(pr["outgroup_id"].dropna())
        df = df[df["tagged_id"].astype(str).isin(keep_ids)]
        print(f"[09b] pairs-only: {len(pr):,} pairs -> {len(df):,} proteins "
              f"(of {n_all:,})", flush=True)
    elif args.phenotypes:
        df = df[df["label"].astype(str).isin(args.phenotypes)]
        print(f"[09b] phenotype filter -> {len(df):,} proteins (of {n_all:,})",
              flush=True)
    if df.empty:
        raise SystemExit("[09b] FATAL: protein set is empty after filtering")

    df = attach_sequences(df, args.fasta).reset_index(drop=True)
    shard_df = df.iloc[args.shard::args.nshards].copy()
    shard_df["_len"] = shard_df["sequence"].str.len()
    shard_df = shard_df.sort_values("_len")   # length-sorted batches pad less
    n = len(shard_df)

    backbone, tok, hidden = build_lora_backbone(
        size=args.backbone_size, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        for_mlm=False, gradient_checkpointing=False, full_attention=args.full_attention)
    n_lora = load_mlm_adapter_into_classifier(backbone, args.mlm_adapter,
                                              adapter_name="mlm")
    backbone.set_adapter("mlm")
    backbone.eval().to(args.device)
    print(f"[09b] backbone ready: hidden={hidden}, {n_lora} MLM LoRA tensors; "
          f"shard {args.shard}/{args.nshards}: {n:,} proteins, k={args.topk}, "
          f"select={args.select}", flush=True)

    K = args.topk
    topk = np.zeros((n, K, hidden), dtype=np.float16)
    means = np.empty((n, hidden), dtype=np.float16)
    lens = np.zeros((n,), dtype=np.int32)
    seqs = shard_df["sequence"].tolist()
    ids = shard_df["tagged_id"].astype(str).tolist()
    bs = args.batch_size
    t0 = time.time()

    with torch.no_grad():
        for i in range(0, n, bs):
            chunk = [s[:args.max_len] for s in seqs[i:i + bs]]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_len + 2)
            enc = {k: v.to(args.device) for k, v in enc.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(args.device == "cuda")):
                h = _encode(backbone, enc["input_ids"], enc["attention_mask"])
                am = enc["attention_mask"]
                m = am.unsqueeze(-1).to(h.dtype)
                # masked mean -- byte-for-byte the same operation as stage 09
                pooled = (h * m).sum(1) / m.sum(1).clamp_min(1.0)

                hf = h.float()
                B, L, H = hf.shape
                valid = am.bool()
                if args.select == "norm":
                    score = hf.norm(dim=-1)                    # (B, L)
                    score = score.masked_fill(~valid, -1.0)    # never pick padding
                    kk = min(K, L)
                    idx = score.topk(kk, dim=1).indices        # (B, kk)
                    # keep ascending sequence order: locality, not rank order
                    idx, _ = idx.sort(dim=1)
                else:  # stride -- null control
                    kk = min(K, L)
                    idx = torch.stack([
                        torch.linspace(0, max(int(v.sum()) - 1, 0), kk,
                                       device=hf.device).round().long()
                        for v in valid])
                sel = torch.gather(
                    hf, 1, idx.unsqueeze(-1).expand(-1, -1, H))   # (B, kk, H)
                # zero out any gathered slot that is padding (short proteins)
                selv = torch.gather(valid, 1, idx).unsqueeze(-1)
                sel = sel * selv.to(sel.dtype)

            b = pooled.shape[0]
            means[i:i + b] = pooled.float().cpu().numpy().astype(np.float16)
            topk[i:i + b, :sel.shape[1]] = sel.cpu().numpy().astype(np.float16)
            lens[i:i + b] = valid.sum(1).cpu().numpy().astype(np.int32)

            if (i // bs) % 50 == 0:
                el = time.time() - t0
                rate = (i + bs) / max(el, 1e-6)
                print(f"[09b] shard {args.shard} {i + len(chunk):,}/{n:,} "
                      f"({rate:.1f} seq/s, eta {(n - i - bs)/max(rate,1e-6)/60:.1f} min)",
                      flush=True)

    # atomic-ish writes: tmp then rename, so a killed job never leaves a half file
    for arr, p in ((topk, tk_path), (means, mn_path), (lens, ln_path)):
        tmp = str(p) + ".tmp.npy"
        np.save(tmp, arr)
        Path(tmp).rename(p)
    ids_path.write_text("\n".join(ids) + "\n")
    done_path.touch()
    gb = topk.nbytes / 1e9
    print(f"[09b] shard {args.shard} done: topk {topk.shape} ({gb:.2f} GB) + "
          f"mean {means.shape}; {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
