#!/usr/bin/env python
"""Stage 08 - fine-tune the shared ESM-2 backbone (design doc Section 11).

Two subcommands on one adapted backbone:
  mlm         domain-adaptive continued MLM (LoRA) on TRAIN-only clusters
  classifier  per-phenotype classifier head (weighted BCE + matched-pair margin)

Loads the Stage-06 labeled dataset (parquet) + the mature-chain FASTA (joined by
tagged_id) + the protein-pairs TSV. Runs on GPU (biotite) or CPU (smoke-test via
--backbone-size 35M). Adapter/head/history saved under --out-dir.

Examples:
  # domain-adaptive MLM on the full train split, 3B backbone
  python scripts/08_train_backbone.py mlm \\
      --labeled results/labeled_dataset_r232_clustered.parquet \\
      --fasta   $PERSIST/secreted_proteins_r232.faa \\
      --backbone-size 3B --out-dir $PERSIST/models/mlm_adapt

  # per-phenotype classifier (loads the MLM adapter first)
  python scripts/08_train_backbone.py classifier --phenotype thermophile \\
      --labeled ... --fasta ... --pairs results/..._protein_pairs.tsv \\
      --mlm-adapter $PERSIST/models/mlm_adapt/mlm_adapter_best \\
      --backbone-size 3B --out-dir $PERSIST/models/clf_thermophile
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _load(labeled, fasta):
    from eptrans.modeling.data import attach_sequences
    df = (pd.read_parquet(labeled) if str(labeled).endswith(".parquet")
          else pd.read_csv(labeled, sep="\t"))
    df = attach_sequences(df, fasta)
    print(f"[08] joined {len(df):,} proteins to sequences "
          f"({df.attrs.get('n_missing_sequences', 0)} missing dropped)")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("mlm", "classifier"):
        p = sub.add_parser(name)
        p.add_argument("--labeled", required=True)
        p.add_argument("--fasta", required=True)
        p.add_argument("--backbone-size", default="3B")
        p.add_argument("--lora-rank", type=int, default=16)
        p.add_argument("--lora-alpha", type=int, default=32)
        p.add_argument("--out-dir", required=True)
        p.add_argument("--batch-size", type=int, default=8)
        p.add_argument("--max-len", type=int, default=1022)
        p.add_argument("--device", default="cuda")
        p.add_argument("--max-steps", type=int, default=None)
        if name == "mlm":
            p.add_argument("--epochs", type=int, default=3)
            p.add_argument("--lr", type=float, default=1e-4)
            p.add_argument("--mask-rate", type=float, default=0.15)
            p.add_argument("--gamma", type=float, default=1.0,
                           help="conservation-mask exponent (Section 13); uniform if no MSA")
            p.add_argument("--coupling-mode", default=None,
                           choices=[None, "span", "contact", "both"],
                           help="mask coupled positions jointly (Section 15 #1)")
            p.add_argument("--span-len", type=int, default=3)
            p.add_argument("--contact-threshold", type=float, default=0.5)
            p.add_argument("--contact-min-sep", type=int, default=6)
            p.add_argument("--contact-pairs", default=None,
                           help="parquet of precomputed (tagged_id, contact_pairs) "
                                "from scripts/10_precompute_contacts.py; avoids the "
                                "per-item 3B contact forward pass")
            p.add_argument("--beta-kl", type=float, default=0.0)
            p.add_argument("--ckpt-every", type=int, default=0,
                           help="write a resumable step-checkpoint every N steps (spot preemption safety)")
            p.add_argument("--no-resume", dest="resume", action="store_false",
                           help="ignore any existing mlm_ckpt.pt and start fresh")
        else:
            p.add_argument("--phenotype", required=True)
            p.add_argument("--pairs", default=None)
            p.add_argument("--mlm-adapter", default=None,
                           help="path to a trained MLM adapter to branch from")
            p.add_argument("--epochs", type=int, default=5)
            p.add_argument("--lr-head", type=float, default=1e-3)
            p.add_argument("--lr-adapter", type=float, default=1e-5)
            p.add_argument("--lam", type=float, default=1.0)
            p.add_argument("--margin", type=float, default=1.0)
            p.add_argument("--pos-weight", type=float, default=None)
            p.add_argument("--neg-per-pos", type=float, default=3.0,
                           help="cap negatives at this multiple of positives (None=all)")
        p.add_argument("--full-attention", action="store_true", default=True,
                       help="LoRA on query/key/value + attention-output dense (Section 15 #3)")
        p.add_argument("--qv-only", dest="full_attention", action="store_false",
                       help="lighter LoRA: query/value only")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    df = _load(args.labeled, args.fasta)

    if args.cmd == "mlm":
        from eptrans.modeling.model import build_lora_backbone
        from eptrans.modeling.data import build_mlm_dataset
        from eptrans.modeling.train import train_mlm
        model, tok, hidden = build_lora_backbone(
            size=args.backbone_size, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
            for_mlm=True, full_attention=args.full_attention)
        # Contact pairs: prefer the precomputed cache (fast); fall back to the
        # per-item 3B contact head only if no cache is supplied.
        cpairs_col = None
        if args.coupling_mode in ("contact", "both") and args.contact_pairs:
            cp = pd.read_parquet(args.contact_pairs)
            df = df.merge(cp, on="tagged_id", how="left")
            cpairs_col = "contact_pairs"
            n_cached = int(df["contact_pairs"].notna().sum())
            print(f"[08] contact-pair cache: {n_cached:,}/{len(df):,} rows have pairs")
        contact_model = (model if args.coupling_mode in ("contact", "both")
                         and not args.contact_pairs else None)
        _dk = dict(max_len=args.max_len, gamma=args.gamma, mask_rate=args.mask_rate,
                   coupling_mode=args.coupling_mode, span_len=args.span_len,
                   contact_threshold=args.contact_threshold,
                   contact_min_sep=args.contact_min_sep, contact_model=contact_model,
                   contact_pairs_col=cpairs_col)
        tr = build_mlm_dataset(df, tok, "train", **_dk)
        va = build_mlm_dataset(df, tok, "val", **_dk)
        print(f"[08] MLM: train {len(tr):,} / val {len(va):,} (train-only clusters)")
        hist = train_mlm(model, tok, tr, va, epochs=args.epochs, lr=args.lr,
                         batch_size=args.batch_size, beta_kl=args.beta_kl,
                         device=args.device, out_dir=args.out_dir, max_steps=args.max_steps,
                         ckpt_every=args.ckpt_every, resume=args.resume)
        print(f"[08] done; val_ppl trace: {hist['val_ppl']}")
    else:
        from eptrans.modeling.model import build_lora_backbone, build_classifier_head
        from eptrans.modeling.data import build_classifier_dataset, build_pair_dataset
        from eptrans.modeling.train import train_classifier
        model, tok, hidden = build_lora_backbone(
            size=args.backbone_size, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
            for_mlm=False, full_attention=args.full_attention)
        if args.mlm_adapter:
            print(f"[08] loading MLM adapter weights from {args.mlm_adapter}")
            model.load_adapter(args.mlm_adapter, adapter_name="mlm")
        head = build_classifier_head(hidden)
        pairs = pd.read_csv(args.pairs, sep="\t") if args.pairs else None
        tr = build_classifier_dataset(df, tok, args.phenotype, "train", max_len=args.max_len,
                                      neg_per_pos=args.neg_per_pos)
        va = build_classifier_dataset(df, tok, args.phenotype, "val", max_len=args.max_len,
                                      neg_per_pos=args.neg_per_pos)
        pair_tr = (build_pair_dataset(df, pairs, tok, args.phenotype, "train",
                                      max_len=args.max_len) if pairs is not None else None)
        n_pairs = len(pair_tr) if pair_tr is not None else 0
        print(f"[08] classifier[{args.phenotype}]: train {len(tr):,} / val {len(va):,} "
              f"| matched pairs {n_pairs:,}")
        hist = train_classifier(model, head, tok, tr, va, pair_ds=pair_tr, epochs=args.epochs,
                                lr_head=args.lr_head, lr_adapter=args.lr_adapter,
                                batch_size=args.batch_size, lam=args.lam, margin=args.margin,
                                pos_weight=args.pos_weight, device=args.device,
                                out_dir=args.out_dir, max_steps=args.max_steps)
        print(f"[08] done; val_auprc trace: {hist['val_auprc']}")


if __name__ == "__main__":
    main()
