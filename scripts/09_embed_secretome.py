#!/usr/bin/env python
"""Embed the whole secretome ONCE through the frozen coupling-aware backbone.

Frozen-backbone cached-embedding path (design #1): instead of re-encoding every
protein through the 3B backbone on every classifier step/epoch (the dominant
Stage-2 cost), embed each protein a single time with the MLM-adapted backbone
frozen, mean-pool to a fixed vector, and cache to disk. Per-phenotype heads then
train on the cached vectors in seconds/epoch (see 10_train_cached_probe.py).

The representation is IDENTICAL to what the end-to-end classifier starts from:
a bare EsmModel wrapped with the SAME Stage-1 MLM LoRA adapter
(load_mlm_adapter_into_classifier -> set_adapter('mlm')), in eval/no_grad. The
only thing the cached path forgoes is the lr_adapter=1e-5 nudge during
classifier training -- a deliberate trade (loss collapses in ~60 steps, so the
nudge buys little for ~1000x the compute).

Pooling matches MeanPoolClassifierHead exactly (masked mean over non-pad tokens),
so a head trained on these vectors is directly loadable as that head for
generation-time steering.

DUAL-EMIT (--emit-topk): from the SAME forward pass, additionally cache a
fixed-size TOP-K per-residue summary for attention pooling. This is the byte-for-
byte selection convention of 09b_embed_perresidue.py (--select norm): per-residue
L2 norm, padding forced to -1 so it is never picked, topk(min(K,L)), indices
re-sorted ASCENDING so sequence locality (not rank order) is preserved, gathered
to (B,K,H), padded slots zeroed. The mean vector is unchanged and identical to
the mean-only path, so mean and attention heads train on features from ONE
identical forward pass -- any AUPRC delta is pooling, not representation.
Emitting K=32 vs K=16 costs the GPU nothing (same forward; only the on-disk
summary size differs).

Shardable for a GPU array: --shard i --nshards N processes rows i::N (strided,
so every shard sees a mix of lengths). Each shard writes emb_shard{i}.npy
(float16, (n_i, H)) + ids_shard{i}.txt (n_i tagged_ids, same order); with
--emit-topk also topk_shard{i}.npy (float16, (n_i, K, H)) + lens_shard{i}.npy
(int32, (n_i,), true pre-truncation residue count). Merge is implicit:
10_train_cached_probe.py concatenates all shards.

Usage (one shard, dual-emit K=32):
  python scripts/09_embed_secretome.py \
    --labeled $PERSIST/labeled_dataset.parquet \
    --fasta   $PERSIST/corpus_all.faa \
    --mlm-adapter $PERSIST/models/mlm_adapt/mlm_adapter_best \
    --backbone-size 3B --lora-rank 32 --lora-alpha 64 \
    --shard 0 --nshards 16 --batch-size 8 --max-len 1022 --device cuda \
    --emit-topk --topk 32 --select norm \
    --out-dir $PERSIST/embeddings/secretome_scoped
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
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1022)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", required=True)
    # dual-emit top-k (attention pooling); OFF by default -> mean-only, unchanged.
    ap.add_argument("--emit-topk", action="store_true",
                    help="also cache top-k per-residue block from the same forward pass")
    ap.add_argument("--topk", type=int, default=32,
                    help="K residues to keep (09b default 32); only used with --emit-topk")
    ap.add_argument("--select", choices=["norm", "stride"], default="norm",
                    help="norm=largest L2 (saliency proxy); stride=null control")
    args = ap.parse_args()

    import torch
    from eptrans.modeling.model import (build_lora_backbone,
                                        load_mlm_adapter_into_classifier)
    from eptrans.modeling.data import attach_sequences
    from eptrans.modeling.train import _encode

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb_path = out / f"emb_shard{args.shard}.npy"
    ids_path = out / f"ids_shard{args.shard}.txt"
    tk_path = out / f"topk_shard{args.shard}.npy"
    ln_path = out / f"lens_shard{args.shard}.npy"
    pos_path = out / f"pos_shard{args.shard}.npy"
    done_path = out / f".done_shard{args.shard}"
    # done requires the top-k outputs too when they were requested
    have_all = done_path.exists() and emb_path.exists() and ids_path.exists() and (
        (not args.emit_topk) or (tk_path.exists() and ln_path.exists() and pos_path.exists()))
    if have_all:
        print(f"[09] shard {args.shard} already complete -> skipping", flush=True)
        return

    # Backbone: bare EsmModel + Stage-1 MLM adapter (same as the classifier branch).
    backbone, tok, hidden = build_lora_backbone(
        size=args.backbone_size, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        for_mlm=False, gradient_checkpointing=False, full_attention=args.full_attention)
    n_lora = load_mlm_adapter_into_classifier(backbone, args.mlm_adapter, adapter_name="mlm")
    backbone.set_adapter("mlm")
    backbone.eval().to(args.device)
    print(f"[09] backbone ready: hidden={hidden}, {n_lora} MLM LoRA tensors transferred",
          flush=True)

    df = attach_sequences(pd.read_parquet(args.labeled), args.fasta)
    df = df.reset_index(drop=True)
    shard_df = df.iloc[args.shard::args.nshards].copy()
    # sort by length so each batch pads to a similar length (throughput)
    shard_df["_len"] = shard_df["sequence"].str.len()
    shard_df = shard_df.sort_values("_len")
    n = len(shard_df)
    K = args.topk
    print(f"[09] shard {args.shard}/{args.nshards}: {n:,} proteins (of {len(df):,}); "
          f"emit_topk={args.emit_topk} K={K} select={args.select}", flush=True)

    seqs = shard_df["sequence"].tolist()
    ids = shard_df["tagged_id"].astype(str).tolist()
    vecs = np.empty((n, hidden), dtype=np.float16)
    topk = np.zeros((n, K, hidden), dtype=np.float16) if args.emit_topk else None
    lens = np.zeros((n,), dtype=np.int32) if args.emit_topk else None
    # pos[i, s] = token index in h of the residue chosen for top-k slot s
    # (0=CLS, r=residue r 1-based, L+1=EOS); -1 marks padding-gathered slots.
    pos = np.full((n, K), -1, dtype=np.int32) if args.emit_topk else None
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
                pooled = (h * m).sum(1) / m.sum(1).clamp_min(1.0)  # masked mean

                if args.emit_topk:
                    hf = h.float()
                    B, L, H = hf.shape
                    valid = am.bool()
                    if args.select == "norm":
                        score = hf.norm(dim=-1)                    # (B, L)
                        score = score.masked_fill(~valid, -1.0)    # never pick padding
                        kk = min(K, L)
                        idx = score.topk(kk, dim=1).indices        # (B, kk)
                        idx, _ = idx.sort(dim=1)                    # ascending seq order
                    else:  # stride -- null control
                        kk = min(K, L)
                        idx = torch.stack([
                            torch.linspace(0, max(int(v.sum()) - 1, 0), kk,
                                           device=hf.device).round().long()
                            for v in valid])
                    sel = torch.gather(
                        hf, 1, idx.unsqueeze(-1).expand(-1, -1, H))   # (B, kk, H)
                    selv = torch.gather(valid, 1, idx).unsqueeze(-1)
                    sel = sel * selv.to(sel.dtype)

            b = pooled.shape[0]
            vecs[i:i + b] = pooled.float().cpu().numpy().astype(np.float16)
            if args.emit_topk:
                topk[i:i + b, :sel.shape[1]] = sel.cpu().numpy().astype(np.float16)
                lens[i:i + b] = valid.sum(1).cpu().numpy().astype(np.int32)
                idx_np = idx.cpu().numpy().astype(np.int32)          # (b, kk) token idx
                selm = selv.squeeze(-1).cpu().numpy().astype(bool)   # (b, kk) real-residue mask
                idx_np[~selm] = -1
                pos[i:i + b, :idx_np.shape[1]] = idx_np
            if (i // bs) % 50 == 0:
                el = time.time() - t0
                rate = (i + bs) / max(el, 1e-6)
                eta = (n - i - bs) / max(rate, 1e-6)
                print(f"[09] shard {args.shard} {i + len(chunk):,}/{n:,} "
                      f"({rate:.1f} seq/s, eta {eta/60:.1f} min)", flush=True)

    def _atomic_save(arr, path):
        tmp = str(path) + ".tmp.npy"
        np.save(tmp, arr)
        Path(tmp).replace(path)

    _atomic_save(vecs, emb_path)
    with open(ids_path, "w") as fh:
        fh.write("\n".join(ids) + "\n")
    if args.emit_topk:
        _atomic_save(topk, tk_path)
        _atomic_save(lens, ln_path)
        _atomic_save(pos, pos_path)
    done_path.touch()
    extra = ""
    if args.emit_topk:
        extra = f" + topk {topk.shape} ({topk.nbytes/1e9:.2f} GB) + lens + pos {pos.shape}"
    print(f"[09] shard {args.shard} DONE -> {emb_path} ({vecs.shape}){extra} + {ids_path}",
          flush=True)


if __name__ == "__main__":
    main()
