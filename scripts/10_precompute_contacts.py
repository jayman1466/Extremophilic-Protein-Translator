#!/usr/bin/env python
"""Precompute ESM-2 contact-pair lists once, cache keyed by tagged_id.

Contact-mode coupling-aware masking (design §15 #1) needs, per sequence, the set
of coupled residue pairs (i, j) where the ESM-2 contact head predicts
p_contact >= threshold and |i-j| >= min_sep. Deriving these inside the training
DataLoader recomputes a 3B forward pass per item per epoch — infeasible at 420k x
3. This script runs the contact head ONCE per unique sequence on GPU and writes a
parquet (tagged_id, contact_pairs) that build_mlm_dataset consumes via
``contact_pairs_col`` for free across all epochs.

Pairs are stored in FULL-sequence residue coordinates (0-based, CLS excluded);
build_mlm_dataset remaps them into each sliding window's local coords.

Usage:
  python scripts/10_precompute_contacts.py \
    --labeled /home/jl_fs/data/labeled_mlm_subsample.parquet \
    --fasta   /home/jl_fs/data/mlm_subsample_mature.faa.gz \
    --out     /home/jl_fs/data/contact_pairs.parquet \
    --backbone-size 3B --threshold 0.5 --min-sep 6 --top-k 128 --device cuda
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labeled", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone-size", default="3B")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-sep", type=int, default=6)
    ap.add_argument("--top-k", type=int, default=128,
                    help="keep only the top-K highest-prob pairs per sequence (bounds storage)")
    ap.add_argument("--max-len", type=int, default=1022)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true",
                    help="skip ids already present in an existing --out parquet")
    args = ap.parse_args()

    import torch
    from transformers import EsmForMaskedLM, EsmTokenizer
    from eptrans.modeling.data import attach_sequences
    from eptrans.modeling.masking import contact_pairs_from_map
    from eptrans.modeling.model import ESM2_CHECKPOINTS

    df = attach_sequences(pd.read_parquet(args.labeled), args.fasta)
    # unique sequences only (many tagged_ids may share a cluster rep sequence)
    uniq = df.drop_duplicates(subset=["tagged_id"])[["tagged_id", "sequence"]].reset_index(drop=True)

    done = set()
    if args.resume and Path(args.out).exists():
        done = set(pd.read_parquet(args.out, columns=["tagged_id"])["tagged_id"])
        uniq = uniq[~uniq["tagged_id"].isin(done)].reset_index(drop=True)
    print(f"[10] {len(uniq):,} sequences to process ({len(done):,} already cached)", flush=True)

    ckpt = ESM2_CHECKPOINTS[args.backbone_size]
    tok = EsmTokenizer.from_pretrained(ckpt)
    model = EsmForMaskedLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16).to(args.device).eval()

    rows, n = [], 0
    for _, r in uniq.iterrows():
        seq = r["sequence"][: args.max_len]
        enc = tok(seq, return_tensors="pt", truncation=True, max_length=args.max_len + 2).to(args.device)
        with torch.no_grad():
            cm = model.predict_contacts(enc["input_ids"])[0].float().cpu().numpy()
        pairs = contact_pairs_from_map(cm, threshold=args.threshold,
                                       min_sep=args.min_sep, top_k=args.top_k)
        rows.append({"tagged_id": r["tagged_id"],
                     "contact_pairs": [[int(a), int(b)] for a, b in pairs]})
        n += 1
        if n % 2000 == 0:
            print(f"[10] {n:,}/{len(uniq):,}  (last: {len(pairs)} pairs)", flush=True)

    out = pd.DataFrame(rows)
    if args.resume and Path(args.out).exists():
        out = pd.concat([pd.read_parquet(args.out), out], ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    tot = int(out["contact_pairs"].map(len).sum())
    print(f"[10] wrote {len(out):,} rows, {tot:,} total pairs -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
