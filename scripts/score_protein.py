#!/usr/bin/env python
"""Per-residue phenotype-saliency for a protein (and its orthologs) -> structure.

What this answers
-----------------
"Which residues make the classifier call this protein <phenotype>?" The pooling
head trained in 10b (attention or top-k MIL) assigns an interpretable weight to
every residue; this script runs a trained head over FULL-LENGTH sequences and
emits that weight per residue, ready to paint onto a structure.

Why it does NOT read the 09b cache
-----------------------------------
The 09b cache keeps only k=32 residues per protein (a disk-scale compromise) and,
until the pos_shard patch, discarded which positions they were. For a handful of
hand-picked proteins that compromise buys nothing: a full-length forward pass is
a few GPU-seconds and yields a DENSE weight over every residue -- including
low-norm residues the k=32 `norm` rule would never have kept. So we re-embed.

Saliency by head type
----------------------
  attention  alpha_i = softmax_i( w^T (tanh(V h_i) * sigmoid(U h_i)) ), the head's
             own pooling weights. This is the object the 10b design calls "the
             interpretable evidence about WHERE the signal sits."
  topk_mil   per-residue logit s_i = MLP(h_i); we report sigmoid(s_i). The head
             pools the top-k of these, so a high s_i means "this residue pushes
             the protein toward the phenotype."
  mean       uniform by construction -> no per-residue localization; reported as
             such rather than faking a signal.

Caveats baked into the output
-----------------------------
  * Softmax scale: the attention head was trained pooling over K=32 slots; over a
    ~330-residue protein alpha is smaller and more diffuse. RANK/percentile is the
    trustworthy readout, not the absolute alpha -- so a percentile column is
    always emitted.
  * Label noise: the phenotype label is genome-level, inherited by every residue.
    High saliency = "correlated with the phenotype across training pairs", NOT
    "catalytic". Structural co-location with a known active site is corroboration
    you add downstream.
  * Special tokens: CLS/EOS receive attention mass too; the per-residue table
    excludes them but their combined weight is reported as `special_token_mass`
    so a diffuse head (most mass on CLS) is visible rather than hidden.

Usage
-----
  # score sequences from a FASTA, dump per-residue TSVs
  python scripts/score_protein.py \
    --fasta my_orthologs.faa \
    --mlm-adapter $MODEL_ROOT/mlm_adapt/mlm_adapter_best \
    --head $MODEL_ROOT/models/pooling/psychrophile__attention/head_best.pt \
    --out-dir results/saliency_psychro

  # also write alpha into the B-factor column of matching structures
  python scripts/score_protein.py --fasta my_orthologs.faa ... \
    --pdb-dir structures/ --bfactor percentile

  # pull an extremophile protein + its taxonomy-matched outgroup partners
  # straight from the pair table + corpus FASTA, then score them together
  python scripts/score_protein.py \
    --from-pairs $W/labeled_dataset_protein_pairs.tsv --protein-id <ext_id> \
    --corpus-fasta $W/secretome.faa --phenotype psychrophile ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ----------------------------- IO helpers ---------------------------------- #
def read_fasta(path: str) -> "list[tuple[str, str]]":
    """Minimal FASTA reader -> [(id, seq)]. id = first whitespace token of header."""
    recs, cur_id, cur = [], None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line[0] == ">":
                if cur_id is not None:
                    recs.append((cur_id, "".join(cur)))
                cur_id = line[1:].split()[0]
                cur = []
            else:
                cur.append(line.strip())
    if cur_id is not None:
        recs.append((cur_id, "".join(cur)))
    return recs


def extract_ids_from_fasta(fasta: str, wanted: "set[str]") -> "dict[str, str]":
    """Pull specific ids out of a large corpus FASTA in one streaming pass."""
    want = set(wanted)
    found, cur_id, keep, cur = {}, None, False, []
    with open(fasta) as fh:
        for line in fh:
            if line[0] == ">":
                if cur_id is not None and keep:
                    found[cur_id] = "".join(cur)
                cur_id = line[1:].split()[0]
                keep = cur_id in want
                cur = []
                if len(found) == len(want):
                    break
            elif keep:
                cur.append(line.strip())
    if cur_id is not None and keep and cur_id not in found:
        found[cur_id] = "".join(cur)
    return found


# --------------------------- head reconstruction --------------------------- #
def infer_head_kind(state: dict) -> str:
    """Identify the pooling head from its saved parameter keys."""
    keys = set(state.keys())
    if {"V.weight", "U.weight", "w.weight"} <= keys:
        return "attention"
    # mean and topk_mil share an identical parameter set (just `net`), so they
    # are indistinguishable from weights alone -> caller must disambiguate.
    return "ambiguous"


def build_head_from_state(state: dict, kind_hint: str):
    """Rebuild a pooling head sized to match a saved state_dict and load it."""
    import torch
    from eptrans.modeling.pooling import build_pooling_head

    kind = infer_head_kind(state)
    if kind == "ambiguous":
        if kind_hint not in ("mean", "topk_mil"):
            raise SystemExit(
                "head has no attention params and --pooling was not one of "
                "{mean, topk_mil}; cannot tell which non-attention head this is. "
                "Pass --pooling mean|topk_mil to disambiguate.")
        kind = kind_hint

    d_hidden, d_in = state["net.0.weight"].shape           # (d_hidden, d_in)
    attn_dim = state["V.weight"].shape[0] if kind == "attention" else 128
    head = build_pooling_head(kind, d_in, d_hidden=d_hidden, attn_dim=attn_dim)
    missing, unexpected = head.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"head state_dict mismatch: missing={missing} "
                         f"unexpected={unexpected}")
    return head, kind, d_in


# ------------------------------ PDB B-factor ------------------------------- #
def write_bfactor_pdb(pdb_in: str, pdb_out: str, per_res: "dict[int, float]",
                      resseq_offset: int = 0) -> "tuple[int, int]":
    """Rewrite the B-factor column (cols 61-66) of every ATOM/HETATM line with the
    per-residue value keyed by resSeq (cols 23-26). Fixed-column PDB format, so no
    parser dependency. Returns (n_lines_written, n_residues_hit)."""
    hit = set()
    n = 0
    out = []
    with open(pdb_in) as fh:
        for line in fh:
            if line[:6] in ("ATOM  ", "HETATM"):
                try:
                    resseq = int(line[22:26])
                except ValueError:
                    out.append(line)
                    continue
                val = per_res.get(resseq - resseq_offset)
                if val is None:
                    b = 0.0
                else:
                    b = float(val)
                    hit.add(resseq - resseq_offset)
                line = f"{line[:60]}{b:6.2f}{line[66:]}"
                n += 1
            out.append(line)
    Path(pdb_out).write_text("".join(out))
    return n, len(hit)


# --------------------------------- main ------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fasta", help="sequences to score")
    src.add_argument("--from-pairs", help="pair table (10b) to pull ids from")
    ap.add_argument("--protein-id", help="with --from-pairs: the ext_id to center on")
    ap.add_argument("--corpus-fasta", help="with --from-pairs: FASTA to pull seqs from")
    ap.add_argument("--phenotype", help="with --from-pairs: restrict to this class")

    ap.add_argument("--mlm-adapter", required=True)
    ap.add_argument("--head", required=True, help="head_best.pt from 10b")
    ap.add_argument("--pooling", default=None,
                    help="disambiguate a non-attention head: mean|topk_mil")
    ap.add_argument("--backbone-size", default="3B")
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--full-attention", action="store_true", default=True)
    ap.add_argument("--qv-only", dest="full_attention", action="store_false")
    ap.add_argument("--max-len", type=int, default=1022)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pdb-dir", default=None,
                    help="if set, look for <id>.pdb here and write <id>_saliency.pdb")
    ap.add_argument("--bfactor", choices=["alpha", "percentile"], default="percentile",
                    help="what to write into the B-factor column")
    ap.add_argument("--resseq-offset", type=int, default=0,
                    help="subtract from PDB resSeq before matching residue index")
    args = ap.parse_args()

    import torch
    from eptrans.modeling.model import (build_lora_backbone,
                                        load_mlm_adapter_into_classifier)
    from eptrans.modeling.train import _encode

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ----- assemble the sequence set ----------------------------------------
    if args.fasta:
        recs = read_fasta(args.fasta)
    else:
        if not (args.protein_id and args.corpus_fasta):
            raise SystemExit("--from-pairs requires --protein-id and --corpus-fasta")
        import pandas as pd
        pr = pd.read_csv(args.from_pairs, sep="\t", dtype=str)
        if args.phenotype:
            pr = pr[pr["class"] == args.phenotype]
        rows = pr[pr["ext_id"] == args.protein_id]
        if rows.empty:
            raise SystemExit(f"{args.protein_id} not found as ext_id in pair table")
        ids = {args.protein_id} | set(rows["outgroup_id"].dropna())
        seqmap = extract_ids_from_fasta(args.corpus_fasta, ids)
        missing = ids - set(seqmap)
        if missing:
            print(f"[score] WARNING: {len(missing)} ids not in corpus FASTA: "
                  f"{sorted(missing)[:5]}", flush=True)
        recs = [(args.protein_id, seqmap[args.protein_id])] + \
               [(i, seqmap[i]) for i in sorted(ids - {args.protein_id}) if i in seqmap]
    if not recs:
        raise SystemExit("[score] no sequences to score")
    print(f"[score] {len(recs)} sequence(s) to score", flush=True)

    # ----- load backbone + adapter + head -----------------------------------
    backbone, tok, hidden = build_lora_backbone(
        size=args.backbone_size, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        for_mlm=False, gradient_checkpointing=False, full_attention=args.full_attention)
    n_lora = load_mlm_adapter_into_classifier(backbone, args.mlm_adapter,
                                              adapter_name="mlm")
    backbone.set_adapter("mlm")
    backbone.eval().to(args.device)

    state = torch.load(args.head, map_location=args.device)
    head, kind, d_in = build_head_from_state(state, args.pooling)
    head.eval().to(args.device)
    if d_in != hidden:
        raise SystemExit(f"head d_in={d_in} != backbone hidden={hidden}")
    print(f"[score] backbone {n_lora} MLM LoRA tensors; head kind={kind} "
          f"(hidden={hidden})", flush=True)

    if kind == "mean":
        print("[score] NOTE: mean pooling is uniform over residues by construction "
              "-> no per-residue localization to report. Writing uniform weights.",
              flush=True)

    summary = {}
    for pid, seq in recs:
        seq = seq[:args.max_len]
        L = len(seq)
        if L == 0:
            print(f"[score] {pid}: empty sequence, skipping", flush=True)
            continue
        enc = tok([seq], return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_len + 2)
        enc = {k: v.to(args.device) for k, v in enc.items()}
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(args.device == "cuda")):
                h = _encode(backbone, enc["input_ids"], enc["attention_mask"])
            am = enc["attention_mask"]
            hf = h.float()

            if kind == "attention":
                a = head.alpha(hf, am).squeeze(0).cpu().numpy()        # (T,)
                per_tok = a
            elif kind == "topk_mil":
                s = head.net(hf).squeeze(-1).squeeze(0)                # (T,) logits
                per_tok = torch.sigmoid(s).cpu().numpy()
            else:  # mean
                per_tok = np.full(hf.shape[1], 1.0 / hf.shape[1], dtype=np.float32)

        # token layout: [CLS] res_1 ... res_L [EOS]; residue i (1-based) = token i
        T = per_tok.shape[0]
        res_tok = per_tok[1:1 + L]                       # residues only
        special = float(per_tok[0] + (per_tok[L + 1] if L + 1 < T else 0.0))

        # percentile within this protein's residues (rank readout, scale-robust)
        order = res_tok.argsort()
        pct = np.empty(L, dtype=np.float32)
        pct[order] = np.linspace(0, 100, L, endpoint=True) if L > 1 else 50.0

        # per-residue TSV
        tsv = out / f"{_safe(pid)}_residue_scores.tsv"
        with open(tsv, "w") as fh:
            fh.write("residue_index\taa\tsaliency\tpercentile\n")
            for i in range(L):
                fh.write(f"{i+1}\t{seq[i]}\t{res_tok[i]:.6g}\t{pct[i]:.2f}\n")

        # top residues for the quick-look summary
        topn = min(15, L)
        top_idx = res_tok.argsort()[::-1][:topn]
        top = [{"residue_index": int(j + 1), "aa": seq[j],
                "saliency": float(res_tok[j]), "percentile": float(pct[j])}
               for j in sorted(top_idx, key=lambda j: -res_tok[j])]
        summary[pid] = {"length": L, "head_kind": kind,
                        "special_token_mass": special,
                        "saliency_sum_residues": float(res_tok.sum()),
                        "top_residues": top, "tsv": str(tsv)}
        print(f"[score] {pid}: L={L} special_mass={special:.3f} "
              f"top1=res{top[0]['residue_index']}({top[0]['aa']}) "
              f"pct={top[0]['percentile']:.1f}", flush=True)

        # optional structure paint
        if args.pdb_dir:
            pdb_in = Path(args.pdb_dir) / f"{pid}.pdb"
            if not pdb_in.exists():
                pdb_in = Path(args.pdb_dir) / f"{_safe(pid)}.pdb"
            if pdb_in.exists():
                vals = pct if args.bfactor == "percentile" else res_tok
                per_res = {i + 1: float(vals[i]) for i in range(L)}
                pdb_out = out / f"{_safe(pid)}_saliency.pdb"
                nlines, nhit = write_bfactor_pdb(str(pdb_in), str(pdb_out), per_res,
                                                 resseq_offset=args.resseq_offset)
                summary[pid]["pdb_out"] = str(pdb_out)
                summary[pid]["pdb_residues_painted"] = nhit
                print(f"[score]   painted {nhit}/{L} residues into {pdb_out.name} "
                      f"(B-factor={args.bfactor})", flush=True)
            else:
                print(f"[score]   no structure {pid}.pdb in {args.pdb_dir}", flush=True)

    (out / "saliency_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[score] wrote {len(summary)} protein summaries -> "
          f"{out / 'saliency_summary.json'}", flush=True)


def _safe(s: str) -> str:
    """Filesystem-safe id (the ~ separator and any slashes -> _)."""
    return s.replace("/", "_").replace("~", "_")


if __name__ == "__main__":
    main()
