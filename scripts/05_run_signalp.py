#!/usr/bin/env python3
"""Stage 05 - run SignalP 6.0 on selected genomes and extract secreted proteins.

Two roles:
  1. ``--emit-slurm`` : write a SLURM batch script that runs SignalP 6.0 on a set
     of per-genome proteome FASTAs on biotite (SignalP is a pyenv shim there and
     needs PYENV_ROOT/PATH exported; torch-heavy, so a cluster job is preferred).
  2. default (local/parse) : given already-produced SignalP output dir(s), parse
     ``prediction_results.txt`` and extract secreted-protein sequences to a FASTA
     plus a per-protein table.

The secreted set = proteins whose SignalP class is any signal-peptide type
(SP / LIPO / TAT / TATLIPO / PILIN), i.e. not OTHER.

Usage
-----
    # 1) emit a SLURM script to run SignalP on genomes listed in a file
    python scripts/05_run_signalp.py --emit-slurm \
        --accessions results/selection.extremophiles.tsv \
        --acc-col accession \
        --slurm-out scripts/slurm/05_signalp.sbatch

    # 2) parse + extract after the job finishes
    python scripts/05_run_signalp.py \
        --signalp-out-root <scratch>/signalp \
        --out-fasta results/secreted_proteins.faa \
        --out-table results/secreted_proteins.tsv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from eptrans.config import load_config
from eptrans.signalp import (
    parse_prediction_results, extract_secreted, summarize, SP_CLASSES,
)


SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=eptrans_signalp
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={log_dir}/signalp_%A_%a.out
#SBATCH --array=0-{max_chunk_idx}%{concurrency}

set -euo pipefail

# SignalP 6.0 is installed as a pyenv shim on biotite - activate pyenv.
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"

ACC_LIST="{acc_list}"
PROT_DIR="{protein_faa_reps}"      # per-genome proteomes: <domain>/<GENOME>_protein.faa.gz
OUT_ROOT="{out_root}"
ORGANISM="{organism}"
MODE="{mode}"
CHUNK_SIZE={chunk_size}
TORCH_THREADS={torch_threads}
WRITE_PROCS={write_procs}
BSIZE={bsize}

# This task processes accessions [START, END) of the list.
START=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE + 1))
END=$((START + CHUNK_SIZE - 1))
mkdir -p "$OUT_ROOT"

# SignalP batches internally: concatenate this chunk's proteomes into ONE FASTA
# (protein ids namespaced by genome as {{GENOME}}~{{PROTID}}) and run once.
CHUNK_FAA=$(mktemp --suffix=.faa)
N=0
for i in $(seq "$START" "$END"); do
  ACC=$(sed -n "${{i}}p" "$ACC_LIST")
  [ -z "$ACC" ] && continue
  FAA=""
  for DOM in bacteria archaea; do
    CAND="$PROT_DIR/$DOM/${{ACC}}_protein.faa.gz"
    if [ -f "$CAND" ]; then FAA="$CAND"; break; fi
  done
  if [ -z "$FAA" ]; then echo "proteome not found for $ACC"; continue; fi
  # prefix each header with the genome id so proteins stay attributable
  gunzip -c "$FAA" | awk -v g="$ACC" '/^>/{{sub(/^>/,">"g"~")}}1' >> "$CHUNK_FAA"
  N=$((N+1))
done
echo "chunk $SLURM_ARRAY_TASK_ID: $N genomes -> $(grep -c "^>" "$CHUNK_FAA") proteins"

OUT="$OUT_ROOT/chunk_${{SLURM_ARRAY_TASK_ID}}"
mkdir -p "$OUT"
signalp6 --fastafile "$CHUNK_FAA" --output_dir "$OUT" \\
         --format none --organism "$ORGANISM" --mode "$MODE" \\
         --torch_num_threads "$TORCH_THREADS" --write_procs "$WRITE_PROCS" --bsize "$BSIZE"

rm -f "$CHUNK_FAA"
echo "done chunk $SLURM_ARRAY_TASK_ID -> $OUT"
"""


def emit_slurm(args, cfg) -> None:
    accs = _load_accessions(args.accessions, args.acc_col)
    acc_list_path = Path(args.slurm_out).with_suffix(".accessions.txt").resolve()
    acc_list_path.parent.mkdir(parents=True, exist_ok=True)
    acc_list_path.write_text("\n".join(accs) + "\n")

    scratch = cfg.get_path("biotite.scratch_root", "/groups/cress/projects/jaymin/eptrans_scratch")
    out_root = args.out_root or f"{scratch}/signalp"
    log_dir = args.log_dir or f"{scratch}/logs/signalp"

    from eptrans.slurm import plan_array
    plan = plan_array(len(accs), args.chunk_size, args.max_tasks, args.concurrency)
    if plan.grown:
        print(f"[05] chunk-size grown {args.chunk_size} -> {plan.chunk_size} to keep "
              f"array <= {args.max_tasks} tasks (MaxSubmitJobs/MaxArraySize)")
    n = len(accs)
    n_chunks = plan.n_tasks
    args.chunk_size = plan.chunk_size
    args.concurrency = plan.concurrency
    script = SLURM_TEMPLATE.format(
        partition=args.partition or cfg.get_path("signalp.partition", "standard"),
        cpus=args.cpus, mem=args.mem, time=args.time,
        log_dir=log_dir, max_chunk_idx=max(n_chunks - 1, 0), concurrency=args.concurrency,
        acc_list=str(acc_list_path),
        protein_faa_reps=cfg.get_path("biotite.protein_faa_reps"),
        out_root=out_root,
        organism=cfg.get_path("signalp.organism", "other"),
        mode=cfg.get_path("signalp.mode", "fast"),
        chunk_size=args.chunk_size, torch_threads=args.cpus,
        write_procs=args.cpus, bsize=args.bsize,
    )
    Path(args.slurm_out).write_text(script)
    print(f"[05] wrote SLURM script: {args.slurm_out}")
    print(f"[05] accession list: {acc_list_path} ({n} genomes)")
    print(f"[05] chunking: {n_chunks} array tasks x {args.chunk_size} genomes/task "
          f"(concurrency {args.concurrency}, {args.cpus} threads/task)")
    print(f"[05] output root: {out_root} (per-chunk dirs; protein ids namespaced GENOME~PROTID)")
    print(f"[05] submit with:  sbatch {args.slurm_out}")


def _load_accessions(path: str, acc_col: str) -> list[str]:
    if path.endswith(".tsv") or path.endswith(".csv"):
        sep = "\t" if path.endswith(".tsv") else ","
        df = pd.read_csv(path, sep=sep, dtype=str)
        col = acc_col if acc_col in df.columns else df.columns[0]
        return df[col].dropna().astype(str).tolist()
    return [l.strip() for l in open(path) if l.strip()]


def _split_genome_protein(pred_id: str, fallback_genome: str) -> tuple[str, str]:
    """Recover (genome, protein_id) from a SignalP prediction id.

    Chunked runs namespace ids as ``GENOME~PROTID`` (split on the FIRST ``~``);
    per-genome runs have a bare protein id, so the genome comes from the dir name.
    """
    if "~" in pred_id:
        genome, protid = pred_id.split("~", 1)
        return genome, protid
    return fallback_genome, pred_id


def parse_and_extract(args, cfg) -> None:
    root = Path(args.signalp_out_root)
    # find every prediction_results.txt (per-genome OR per-chunk dir)
    res_files = sorted(root.rglob("prediction_results.txt")) if root.exists() else []
    if not res_files:
        raise SystemExit(f"no prediction_results.txt found under {root}")

    classes = set(args.classes or SP_CLASSES)
    all_rows = []
    all_secreted = []
    from collections import defaultdict
    per_genome_preds: dict[str, list] = defaultdict(list)

    for res_file in res_files:
        fallback_genome = res_file.parent.name
        preds = parse_prediction_results(res_file)
        for p in preds:
            genome, protid = _split_genome_protein(p.protein_id, fallback_genome)
            per_genome_preds[genome].append((protid, p))
            if p.prediction in classes:
                all_rows.append({
                    "genome": genome, "protein_id": protid,
                    "signalp_class": p.prediction, "cs_after": p.cs_after,
                    "cs_prob": p.cs_prob,
                    **{f"p_{k}": v for k, v in p.probs.items()},
                })

    # per-genome summary + sequence extraction
    combined_summary = {"n_proteins": 0, "n_secreted": 0, "by_genome": {}}
    faa_root = args.protein_faa_reps or cfg.get_path("biotite.protein_faa_reps")
    for genome, items in per_genome_preds.items():
        preds = [p for _pid, p in items]
        summ = summarize(preds)
        combined_summary["n_proteins"] += summ["n_proteins"]
        combined_summary["n_secreted"] += summ["n_secreted"]
        combined_summary["by_genome"][genome] = summ
        if faa_root:
            faa = _find_proteome(genome, faa_root)
            if faa:
                # rebuild predictions keyed by bare protid for extraction
                bare_preds = []
                for protid, p in items:
                    p2 = p
                    p2.protein_id = protid
                    bare_preds.append(p2)
                for pid, cls, seq in extract_secreted(bare_preds, faa, mature=args.mature,
                                                      classes=list(classes)):
                    all_secreted.append((f"{genome}~{pid}", cls, seq))

    # write per-protein table
    table = pd.DataFrame(all_rows)
    Path(args.out_table).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_table, sep="\t", index=False)

    # write FASTA
    if all_secreted:
        with open(args.out_fasta, "w") as fh:
            for hid, cls, seq in all_secreted:
                fh.write(f">{hid} class={cls}\n")
                for i in range(0, len(seq), 60):
                    fh.write(seq[i:i+60] + "\n")

    json.dump(combined_summary, open(Path(args.out_table).with_suffix(".summary.json"), "w"), indent=2)
    n = combined_summary["n_proteins"]
    ns = combined_summary["n_secreted"]
    print(f"[05] genomes parsed: {len(combined_summary['by_genome'])}")
    print(f"[05] proteins: {n:,} | secreted: {ns:,} "
          f"({(ns/n*100 if n else 0):.1f}%)")
    print(f"[05] wrote {args.out_table} ({len(table):,} secreted rows)")
    if all_secreted:
        print(f"[05] wrote {args.out_fasta} ({len(all_secreted):,} sequences)")


def _find_proteome(acc: str, protein_faa_reps: str) -> str | None:
    for dom in ("bacteria", "archaea"):
        cand = Path(protein_faa_reps) / dom / f"{acc}_protein.faa.gz"
        if cand.exists():
            return str(cand)
    return None


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emit-slurm", action="store_true")
    # emit-slurm args
    ap.add_argument("--accessions", help="TSV/CSV/txt of accessions to run")
    ap.add_argument("--acc-col", default="accession")
    ap.add_argument("--slurm-out", default="scripts/slurm/05_signalp.sbatch")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--partition", default=None)
    ap.add_argument("--cpus", type=int, default=16)
    ap.add_argument("--mem", default="24G")
    ap.add_argument("--time", default="08:00:00")
    ap.add_argument("--concurrency", type=int, default=10,
                    help="max simultaneously-running array tasks (biotite standard QOS caps at 10)")
    ap.add_argument("--chunk-size", type=int, default=1000,
                    help="genomes per array task (keeps array size under SLURM limits)")
    ap.add_argument("--max-tasks", type=int, default=200,
                    help="max array tasks = min(MaxSubmitJobs, MaxArraySize); biotite=200. "
                         "chunk-size auto-grows so n_chunks stays <= this.")
    ap.add_argument("--bsize", type=int, default=32, help="SignalP batch size")
    # parse/extract args
    ap.add_argument("--signalp-out-root", help="root dir of per-genome SignalP outputs")
    ap.add_argument("--protein-faa-reps", default=None, help="dir of per-genome proteomes for seq extraction")
    ap.add_argument("--out-fasta", default="results/secreted_proteins.faa")
    ap.add_argument("--out-table", default="results/secreted_proteins.tsv")
    ap.add_argument("--classes", nargs="*", default=None, help=f"SP classes to keep (default {SP_CLASSES})")
    ap.add_argument("--mature", action="store_true", help="extract mature chain (after cleavage) instead of precursor")
    args = ap.parse_args()

    if args.emit_slurm:
        if not args.accessions:
            ap.error("--emit-slurm requires --accessions")
        emit_slurm(args, cfg)
    else:
        if not args.signalp_out_root:
            ap.error("parse mode requires --signalp-out-root")
        parse_and_extract(args, cfg)


if __name__ == "__main__":
    main()
