#!/usr/bin/env python
"""Cluster-stratified subsample of the secretome for domain-adaptive MLM.

The labeled dataset's ``group`` column IS the mmseqs cluster id (cluster regime:
split groups = clusters). Domain-adaptive MLM only needs the dominant
distributional shift of the extremophilic secretome, which a LoRA adapter
(rank-32, ~0.1% params) extracts from a fraction of the 1.59M train proteins.

Strategy (label-agnostic; MLM uses no phenotype labels):
  1. Dedupe to ONE representative protein per cluster — removes within-cluster
     near-duplicate redundancy (the main thing to cut), keeping inter-cluster
     sequence diversity intact.
  2. Uniformly sample ``--n-train`` cluster representatives (seed-deterministic).
     Uniform-over-clusters is already diversity-preserving vs uniform-over-
     proteins because step 1 stripped the family-size amplification from
     duplicated sequence.
  3. Take a fixed ``--n-val`` val subsample (one per cluster) so early-stop
     pseudo-perplexity is a fast, stable, comparable metric across epochs.

Writes a minimal parquet (tagged_id/label/is_mesophile/label_confidence/split)
consumed by ``08_train_backbone.py mlm`` — sequences are attached at train time
from the mature-chain FASTA by tagged_id. TEST split is omitted (unused by MLM).
"""
import argparse
import pandas as pd


def subsample(labeled: pd.DataFrame, n_train: int, n_val: int, seed: int) -> pd.DataFrame:
    cols = ["tagged_id", "label", "is_mesophile", "label_confidence", "split", "group"]
    df = labeled[cols].copy()
    out = []
    for split, n in [("train", n_train), ("val", n_val)]:
        s = df[df["split"] == split]
        # one representative per cluster (deterministic: first after a seeded shuffle)
        reps = (s.sample(frac=1.0, random_state=seed)
                 .drop_duplicates(subset="group", keep="first"))
        if len(reps) > n:
            reps = reps.sample(n=n, random_state=seed)
        out.append(reps)
    return pd.concat(out).drop(columns="group").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", required=True, help="labeled_dataset_*_clustered.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=400_000)
    ap.add_argument("--n-val", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=1466)
    args = ap.parse_args()

    labeled = pd.read_parquet(args.labeled)
    sub = subsample(labeled, args.n_train, args.n_val, args.seed)
    sub.to_parquet(args.out, compression="zstd", index=False)

    n_tr = (sub["split"] == "train").sum()
    n_va = (sub["split"] == "val").sum()
    print(f"train {n_tr:,} | val {n_va:,} | total {len(sub):,}")
    print("train label mix:", sub[sub.split == "train"]["label"].value_counts().head(6).to_dict())


if __name__ == "__main__":
    main()