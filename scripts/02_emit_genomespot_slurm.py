#!/usr/bin/env python3
"""Stage 02 (cluster) - emit a chunked, multicore SLURM array for GenomeSPOT.

GenomeSPOT runs one genome per invocation (~5s, single-threaded, CPU-bound), so
throughput comes from running many genomes concurrently *within* a node. Each
array task processes a CHUNK of genomes with ``xargs -P <cpus>`` across cores;
chunking keeps the array within SLURM limits.

biotite standard QOS: <=10 running tasks, <=200 submitted, MaxArraySize 1001.
The array is sized so n_tasks <= 200 (chunk-size auto-grows), throttled at %10.

Usage
-----
    python scripts/02_emit_genomespot_slurm.py \
        --accessions results/genomespot_reconciled_r232.delta_accessions.txt \
        --models /groups/.../GenomeSPOT/models \
        --genomespot-python /groups/.../genomespot_env/bin/python \
        --slurm-out scripts/slurm/02_genomespot.sbatch
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eptrans.config import load_config
from eptrans.slurm import plan_array, MAX_TASKS, MAX_RUNNING


TEMPLATE = """#!/bin/bash
#SBATCH --job-name=eptrans_genomespot
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={log_dir}/genomespot_%A_%a.out
#SBATCH --array=0-{max_chunk_idx}%{concurrency}

set -euo pipefail

ACC_LIST="{acc_list}"
GENOME_INDEX="{genome_index}"
PROT_DIR="{protein_faa_reps}"
MODELS="{models}"
OUT_ROOT="{out_root}"
GS_PY="{genomespot_python}"
CHUNK_SIZE={chunk_size}
CPUS={cpus}

START=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE + 1))
END=$((START + CHUNK_SIZE - 1))
OUT="$OUT_ROOT/chunk_${{SLURM_ARRAY_TASK_ID}}"
mkdir -p "$OUT"

run_one() {{
  local ACC="$1"
  # bare accession for genome-index lookup (strip GB_/RS_)
  local BARE=$(echo "$ACC" | sed -E 's/^(GB_|RS_)//')
  local FNA=$(awk -F'\\t' -v a="$BARE" '$1==a{{print $2; exit}}' "$GENOME_INDEX")
  if [ -z "$FNA" ]; then echo "no fna: $ACC"; return 0; fi
  local FAA=""
  for DOM in bacteria archaea; do
    local CAND="$PROT_DIR/$DOM/${{ACC}}_protein.faa.gz"
    if [ -f "$CAND" ]; then FAA="$CAND"; break; fi
  done
  if [ -z "$FAA" ]; then echo "no faa: $ACC"; return 0; fi
  "$GS_PY" -m genome_spot.genome_spot --models "$MODELS" \\
      -c "$FNA" -p "$FAA" -o "$OUT/$BARE" >/dev/null 2>&1 || echo "gs failed: $ACC"
}}
export -f run_one
export GENOME_INDEX PROT_DIR MODELS OUT GS_PY

sed -n "${{START}},${{END}}p" "$ACC_LIST" | xargs -P "$CPUS" -I{{}} bash -c 'run_one "$@"' _ {{}}
echo "done chunk $SLURM_ARRAY_TASK_ID ($(ls "$OUT" | wc -l) outputs)"
"""


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--accessions", required=True)
    ap.add_argument("--acc-col", default="accession")
    ap.add_argument("--models", required=True, help="GenomeSPOT models dir on the cluster")
    ap.add_argument("--genomespot-python", required=True,
                    help="python interpreter with genome_spot installed (sklearn==1.2.2)")
    ap.add_argument("--slurm-out", default="scripts/slurm/02_genomespot.sbatch")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--partition", default=None)
    ap.add_argument("--cpus", type=int, default=16)
    ap.add_argument("--mem", default="24G")
    ap.add_argument("--time", default="08:00:00")
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--max-tasks", type=int, default=MAX_TASKS)
    ap.add_argument("--concurrency", type=int, default=MAX_RUNNING)
    args = ap.parse_args()

    # load accessions
    if args.accessions.endswith((".tsv", ".csv")):
        sep = "\t" if args.accessions.endswith(".tsv") else ","
        df = pd.read_csv(args.accessions, sep=sep, dtype=str)
        col = args.acc_col if args.acc_col in df.columns else df.columns[0]
        accs = df[col].dropna().astype(str).tolist()
    else:
        accs = [l.strip() for l in open(args.accessions) if l.strip()]

    plan = plan_array(len(accs), args.chunk_size, args.max_tasks, args.concurrency)
    if plan.grown:
        print(f"[02-slurm] chunk-size grown {args.chunk_size} -> {plan.chunk_size} "
              f"to keep array <= {args.max_tasks} tasks")

    acc_list_path = Path(args.slurm_out).with_suffix(".accessions.txt").resolve()
    acc_list_path.parent.mkdir(parents=True, exist_ok=True)
    acc_list_path.write_text("\n".join(accs) + "\n")

    scratch = cfg.get_path("biotite.scratch_root", "/groups/cress/projects/jaymin/eptrans_scratch")
    out_root = args.out_root or f"{scratch}/genomespot"
    log_dir = args.log_dir or f"{scratch}/logs/genomespot"

    script = TEMPLATE.format(
        partition=args.partition or cfg.get_path("signalp.partition", "standard"),
        cpus=args.cpus, mem=args.mem, time=args.time, log_dir=log_dir,
        max_chunk_idx=max(plan.n_tasks - 1, 0), concurrency=plan.concurrency,
        acc_list=str(acc_list_path), genome_index=cfg.get_path("biotite.genome_index"),
        protein_faa_reps=cfg.get_path("biotite.protein_faa_reps"),
        models=args.models, out_root=out_root, genomespot_python=args.genomespot_python,
        chunk_size=plan.chunk_size,
    )
    Path(args.slurm_out).write_text(script)
    print(f"[02-slurm] wrote {args.slurm_out}")
    print(f"[02-slurm] {len(accs)} genomes | {plan.n_tasks} tasks x {plan.chunk_size}/task "
          f"| {args.cpus} cores/task | throttle %{plan.concurrency}")
    print(f"[02-slurm] output root: {out_root}")
    print(f"[02-slurm] submit with:  sbatch {args.slurm_out}")


if __name__ == "__main__":
    main()
