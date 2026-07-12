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
classifier training — a deliberate trade (loss collapses in ~60 steps, so the
nudge buys little for ~1000x the compute).

Pooling matches MeanPoolClassifierHead exactly (masked mean over non-pad tokens),
so a head trained on these vectors is directly loadable as that head for
generation-time steering.

Shardable for a GPU array: --shard i --nshards N processes rows i::N (strided,
so every shard sees a mix of lengths). Each shard writes emb_shard{i}.npy
(float16, (n_i, H)) + ids_shard{i}.txt (n_i tagged_ids, same order). Merge is
implicit: 10_train_cached_probe.py concatenates all shards.

Usage (one shard):
  python scripts/09_embed_secretome.py \
    --labeled $PERSIST/labeled_dataset_r232_clustered.parquet \
    --fasta   $PERSIST/secreted_proteins_r232.faa \
    --mlm-adapter $PERSIST/models/mlm_adapt/mlm_adapter_best \
    --backbone-size 3B --lora-rank 32 --lora-alpha 64 \
    --shard 0 --nshards 8 --batch-size 8 --max-len 1022 --device cuda \
    --out-dir $PERSIST/embeddings/secretome_r232
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
    done_path = out / f".done_shard{args.shard}"
    if done_path.exists() and emb_path.exists() and ids_path.exists():
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
    print(f"[09] shard {args.shard}/{args.nshards}: {n:,} proteins (of {len(df):,})",
          flush=True)

    seqs = shard_df["sequence"].tolist()
    ids = shard_df["tagged_id"].astype(str).tolist()
    vecs = np.empty((n, hidden), dtype=np.float16)
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
                m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                pooled = (h * m).sum(1) / m.sum(1).clamp_min(1.0)  # masked mean
            vecs[i:i + pooled.shape[0]] = pooled.float().cpu().numpy().astype(np.float16)
            if (i // bs) % 50 == 0:
                el = time.time() - t0
                rate = (i + bs) / max(el, 1e-6)
                eta = (n - i - bs) / max(rate, 1e-6)
                print(f"[09] shard {args.shard} {i + len(chunk):,}/{n:,} "
                      f"({rate:.1f} seq/s, eta {eta/60:.1f} min)", flush=True)

    tmp_emb = str(emb_path) + ".tmp.npy"
    np.save(tmp_emb, vecs)
    Path(tmp_emb).replace(emb_path)
    with open(ids_path, "w") as fh:
        fh.write("\n".join(ids) + "\n")
    done_path.touch()
    print(f"[09] shard {args.shard} DONE -> {emb_path} ({vecs.shape}) + {ids_path}",
          flush=True)


if __name__ == "__main__":
    main()
