#!/usr/bin/env python3
"""Pick the lambda that maximises a held-out val metric for one phenotype.

Reads <model-root>/cached_probes_lam<LAM>/cached_probe_summary.json for each
lam in --lams, extracts summary[pheno][<metric-key>], prints the winning lam
to stdout (nothing else, so it can be captured in a shell var).

--metric {auprc,auroc,pair_auc} selects the primary key
(val_auprc | val_auroc | val_pair_auc).
Ties broken by val_pair_auc, then by the smaller lam (less pair-loss weight).

For the mhk32 production run the LOCK policy is --metric pair_auc: lambda is
chosen to maximise held-out matched-pair AUC, the mechanism the ortholog pairs
exist to serve (the margin loss trades a little pointwise AUROC for better
pair-ranking). scope/signal decisions still use AUROC (--metric auroc) and
deployment lift uses AUPRC (--metric auprc, the default for back-compat).
"""
import argparse, json, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-root", required=True)
    ap.add_argument("--lams", required=True, help="space-separated, e.g. '0.5 1 2 4'")
    ap.add_argument("--pheno", required=True)
    ap.add_argument("--metric", default="auprc",
                    choices=["auprc", "auroc", "pair_auc"],
                    help="primary selection metric (default auprc for "
                         "back-compat; mhk32 lock policy uses pair_auc)")
    a = ap.parse_args()

    key = f"val_{a.metric}"
    cands = []  # (primary, pair_auc, -lam_float, lam_str)
    for lam in a.lams.split():
        f = Path(a.model_root) / f"cached_probes_lam{lam}" / "cached_probe_summary.json"
        if not f.exists():
            print(f"[select] missing {f}", file=sys.stderr)
            continue
        try:
            summ = json.load(open(f))
        except Exception as e:
            print(f"[select] unreadable {f}: {e}", file=sys.stderr)
            continue
        m = summ.get(a.pheno)
        if not m or key not in m:
            print(f"[select] {a.pheno}/{key} absent from {f}", file=sys.stderr)
            continue
        pr = float(m[key])
        pa = float(m.get("val_pair_auc") or 0.0)
        cands.append((pr, pa, -float(lam), lam))

    if not cands:
        print(f"[select] no scored lam for {a.pheno}/{key}", file=sys.stderr)
        sys.exit(2)
    cands.sort(reverse=True)
    best = cands[0]
    print(f"[select] {a.pheno}: chose lam={best[3]} "
          f"({a.metric}={best[0]:.4f} pair_auc={best[1]:.4f}); "
          f"candidates={[(c[3], round(c[0],4)) for c in cands]}", file=sys.stderr)
    # stdout: ONLY the lam string, for $(...) capture
    print(best[3])


if __name__ == "__main__":
    main()
