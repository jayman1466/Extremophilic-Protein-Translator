#!/usr/bin/env python
"""Post-hoc taxonomy-controlled pair eval for a Stage-2 classifier epoch snapshot.

Computes held-out matched-pair metrics (pair_acc / pair_auc / margin_gap; see
train.evaluate_pair_metrics) for a per-epoch snapshot written by train_classifier
(``<out-dir>/clf_epoch<E>/clf_matched.pt``). This is the metric that isolates
genuine phenotype signal from organism-level taxonomic signal — the matched
outgroup holds taxonomy ~constant within each pair, so pair_acc > 0.5 means the
head learned thermoadaptation (etc.) rather than "which taxon is this".

Rebuilds a fresh LoRA backbone and overlays the snapshot's trained weights
(``backbone_trainable`` are the final LoRA deltas + ``head``), exactly as the
train_classifier resume path does — so no MLM-adapter reload is needed. Runs
purely offline against an already-written snapshot, so it can score a run that
is still training (or already finished) without touching it.

Usage:
  python scripts/08d_eval_pairs.py \
    --snapshot $PERSIST/models/clf_hyperthermophile/clf_epoch0/clf_matched.pt \
    --phenotype hyperthermophile \
    --labeled $PERSIST/labeled_dataset_r232_clustered.parquet \
    --fasta   $PERSIST/secreted_proteins_r232.faa \
    --pairs   $PERSIST/labeled_dataset_r232_clustered_protein_pairs.tsv \
    --split val --backbone-size 3B --lora-rank 32 --lora-alpha 64 \
    --device cuda --out $PERSIST/models/clf_hyperthermophile/pair_eval_epoch0.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _load(labeled: str, fasta: str) -> pd.DataFrame:
    """Mirror 08_train_backbone._load: join labeled rows to sequences by tagged_id."""
    df = pd.read_parquet(labeled)
    seqs = {}
    tid = None
    buf = []
    with open(fasta) as fh:
        for line in fh:
            if line.startswith(">"):
                if tid is not None:
                    seqs[tid] = "".join(buf)
                tid = line[1:].strip().split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if tid is not None:
        seqs[tid] = "".join(buf)
    df["sequence"] = df["tagged_id"].astype(str).map(seqs)
    n0 = len(df)
    df = df[df["sequence"].notna()].reset_index(drop=True)
    print(f"[08d] joined {len(df):,} proteins to sequences ({n0 - len(df):,} missing dropped)")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True, help="path to clf_matched.pt")
    ap.add_argument("--phenotype", required=True)
    ap.add_argument("--labeled", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--backbone-size", default="3B")
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=1022)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--full-attention", action="store_true", default=True)
    ap.add_argument("--qv-only", dest="full_attention", action="store_false")
    ap.add_argument("--out", default=None, help="write metrics JSON here")
    args = ap.parse_args()

    import torch
    from eptrans.modeling.model import build_lora_backbone, build_classifier_head
    from eptrans.modeling.data import build_pair_dataset
    from eptrans.modeling.train import evaluate_pair_metrics

    backbone, tok, hidden = build_lora_backbone(
        size=args.backbone_size, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        for_mlm=False, full_attention=args.full_attention)
    head = build_classifier_head(hidden)

    ck = torch.load(args.snapshot, map_location="cpu")
    missing = backbone.load_state_dict(ck["backbone_trainable"], strict=False)
    head.load_state_dict(ck["head"])
    print(f"[08d] loaded snapshot epoch={ck.get('epoch')} step={ck.get('step')} "
          f"val_auprc={ck.get('val_auprc')}; {len(missing.unexpected_keys)} unexpected keys")
    backbone.to(args.device); head.to(args.device)

    df = _load(args.labeled, args.fasta)
    pairs = pd.read_csv(args.pairs, sep="\t")
    pair_ds = build_pair_dataset(df, pairs, tok, args.phenotype, args.split, max_len=args.max_len)
    print(f"[08d] {args.phenotype} {args.split}-split matched pairs: {len(pair_ds):,}")

    pm = evaluate_pair_metrics(backbone, head, tok, pair_ds, args.device, args.batch_size)
    pm.update({"phenotype": args.phenotype, "split": args.split,
               "snapshot": args.snapshot, "epoch": ck.get("epoch"),
               "step": ck.get("step"), "val_auprc": ck.get("val_auprc")})
    print(f"[08d] RESULT pair_acc={pm['pair_acc']:.4f} pair_auc={pm['pair_auc']:.4f} "
          f"margin_gap={pm['margin_gap']:.4f} n={pm['n']}")
    if args.out:
        json.dump(pm, open(args.out, "w"), indent=2)
        print(f"[08d] wrote {args.out}")


if __name__ == "__main__":
    main()
