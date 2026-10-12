#!/usr/bin/env python
"""Calibrate the Tier-1b RMSD caps against an empirical null distribution.

Why
---
The production caps (active site 1.0 A, interfaces 1.5 A) are round numbers.
``docs/ranker_qc_and_recalibration.md`` showed that rejected designs pile up
*immediately* above them rather than forming a tail, i.e. the caps sit in the
dense part of the RMSD distribution and are therefore discriminating at a
resolution finer than ESMFold's own fold-to-fold reproducibility.  A cap is
only meaningful relative to what a structurally **neutral** change of the same
size does to the same predictor.

What this measures
------------------
For each mutation count ``n`` in ``--counts``, sample ``--reps`` conservative
variants of the wild type: ``n`` substitutions drawn only from

  * positions that are not active-site, not interface, and not highly conserved
    (so the variant is a priori structurally innocuous), and
  * within-class exchanges (BLOSUM-style groups below), so hydrophobic cores
    stay hydrophobic and charges stay charges.

Fold each with the same ``RefoldClient`` the generation pipeline uses, and
record RMSD over each protected set.  The cap for mutation count ``n`` is then
the ``--quantile`` of that null -- the distance a *harmless* variant of that
size already moves the prediction.  Anything beyond it is signal, not noise.

Two floors are enforced so the table never gets tighter than the predictor is
reproducible:

  * the self-consistency floor: the WT sequence is refolded ``--self-reps``
    times and the spread of those folds is a hard lower bound on any cap;
  * ``--min-cap``.

Caps are *not* clamped here -- clamping is the caller's job
(``11_generate_pipelineB.py --core-rmsd-ceiling/--interface-rmsd-ceiling``), so
the table records what was measured and the pipeline records what it enforced.

Output
------
``{"caps": {"active_site": {"5": 0.41, ...}, "interface": {...}},
   "null": {...raw per-count RMSDs...}, "self_consistency": {...}, "meta": {...}}``
consumable by ``11_generate_pipelineB.py --rmsd-cap-table``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from importlib import import_module  # noqa: E402

_gen = import_module("11_generate")
RefoldClient = _gen.RefoldClient
detect_active_site = _gen.detect_active_site
run_msa_conservation = _gen.run_msa_conservation

# Conservative exchange groups.  Substitutions are drawn within a group, so the
# variant preserves charge, polarity and rough volume at every mutated site.
EXCHANGE_GROUPS = [
    "AGS", "ST", "ILVM", "FYW", "KR", "DE", "NQ", "DN", "EQ", "HKR", "AV",
]


def conservative_variant(seq: str, allowed: list[int], n: int,
                         rng: np.random.Generator) -> str:
    """``n`` within-exchange-group substitutions at ``allowed`` positions."""
    s = list(seq)
    pool = [i for i in allowed
            if any(s[i] in g and len(g) > 1 for g in EXCHANGE_GROUPS)]
    if len(pool) < n:
        n = len(pool)
    for i in rng.choice(pool, size=n, replace=False):
        groups = [g for g in EXCHANGE_GROUPS if s[i] in g and len(g) > 1]
        g = groups[int(rng.integers(len(groups)))]
        alts = [a for a in g if a != s[i]]
        s[i] = alts[int(rng.integers(len(alts)))]
    return "".join(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wt-pdb", required=True)
    ap.add_argument("--wt-fasta", required=True)
    ap.add_argument("--refold-workdir", required=True)
    ap.add_argument("--interfaces-json", default="",
                    help="interfaces JSON emitted by 11_generate_pipelineB.py")
    ap.add_argument("--uniref-db", default="",
                    help="mmseqs UniRef50 db; omit to block on protected sets "
                         "only (conservation array left uniform)")
    ap.add_argument("--workdir", default=".",
                    help="MSA scratch; reuses hits.m8 if the generator already ran")
    ap.add_argument("--transfer-json", default="",
                    help="11a_annotate.py transfer JSON, so the calibrated caps "
                         "cover the same active site the generator protects")
    ap.add_argument("--conservation-freeze", type=float, default=0.90)
    ap.add_argument("--counts", default="5,10,20,40,60,80,100")
    ap.add_argument("--reps", type=int, default=24)
    ap.add_argument("--self-reps", type=int, default=12)
    ap.add_argument("--quantile", type=float, default=0.95)
    ap.add_argument("--min-cap", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=1466)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    seq = "".join(l.strip() for l in Path(args.wt_fasta).read_text().splitlines()
                  if not l.startswith(">"))
    L = len(seq)

    # NOTE: an earlier revision called detect_active_site(args.wt_pdb), which is
    # neither its signature nor its return type (it takes the sequence plus a
    # conservation array and returns a 4-tuple) -- that call raised on entry, so
    # this script had never completed a run.  Conservation must therefore be
    # computed before the active site, not after it.
    cons = (run_msa_conservation(seq, args.uniref_db, args.workdir)[0]
            if args.uniref_db else np.zeros(L, dtype=float))
    transferred = []
    if args.transfer_json and Path(args.transfer_json).exists():
        transferred = json.loads(Path(args.transfer_json).read_text()).get("transferred", [])
    _frozen_act, active, _assigned, _src = detect_active_site(
        seq, cons, freeze_thresh=args.conservation_freeze, transferred=transferred)
    protected_sets: dict[str, list[int]] = {}
    if active:
        protected_sets["active_site"] = sorted(active)
    if args.interfaces_json:
        for lbl, info in json.loads(Path(args.interfaces_json).read_text()).items():
            protected_sets[lbl] = info["positions"]
    if not protected_sets:
        raise SystemExit("[22] no protected sets -- nothing to calibrate")

    blocked = {p - 1 for ps in protected_sets.values() for p in ps}
    blocked |= {i for i in range(L) if cons[i] >= args.conservation_freeze}
    allowed = [i for i in range(L) if i not in blocked]
    print(f"[22] L={L} protected_sets={list(protected_sets)} "
          f"blocked={len(blocked)} allowed={len(allowed)}", flush=True)

    refolder = RefoldClient(args.refold_workdir, args.wt_pdb)
    if not refolder.wait_ready():
        raise SystemExit("[22] refold worker never became READY")

    # ---- self-consistency floor: same sequence, repeated folds ----
    self_r: dict[str, list[float]] = {lbl: [] for lbl in protected_sets}
    for _ in range(args.self_reps):
        for lbl, v in refolder.refold_rmsd_multi(seq, protected_sets).items():
            if v is not None:
                self_r[lbl].append(float(v))
    floor = {lbl: (float(np.quantile(v, args.quantile)) if v else 0.0)
             for lbl, v in self_r.items()}
    print(f"[22] self-consistency floor at q={args.quantile}: "
          + ", ".join(f"{k}={v:.3f}" for k, v in floor.items()), flush=True)

    counts = [int(x) for x in args.counts.split(",") if x.strip()]
    null: dict[str, dict[str, list[float]]] = {lbl: {} for lbl in protected_sets}
    for n in counts:
        got = {lbl: [] for lbl in protected_sets}
        for _ in range(args.reps):
            v = conservative_variant(seq, allowed, n, rng)
            for lbl, r in refolder.refold_rmsd_multi(v, protected_sets).items():
                if r is not None:
                    got[lbl].append(float(r))
        for lbl in protected_sets:
            null[lbl][str(n)] = got[lbl]
        print(f"[22] n={n:3d}  " + "  ".join(
            f"{lbl}: med={np.median(got[lbl]):.3f} q{args.quantile:g}="
            f"{np.quantile(got[lbl], args.quantile):.3f} (n={len(got[lbl])})"
            for lbl in protected_sets if got[lbl]), flush=True)

    # Collapse the interface sets into one table: the pipeline applies a single
    # interface cap to every interface, so calibrate the pooled, most permissive
    # interface rather than per-interface caps it cannot express.
    caps: dict[str, dict[str, float]] = {"active_site": {}, "interface": {}}
    iface = [lbl for lbl in protected_sets if lbl != "active_site"]
    for n in counts:
        k = str(n)
        if "active_site" in null and null["active_site"].get(k):
            caps["active_site"][k] = max(
                args.min_cap, floor.get("active_site", 0.0),
                float(np.quantile(null["active_site"][k], args.quantile)))
        pooled = [x for lbl in iface for x in null[lbl].get(k, [])]
        if pooled:
            caps["interface"][k] = max(
                args.min_cap, max([floor.get(l, 0.0) for l in iface] or [0.0]),
                float(np.quantile(pooled, args.quantile)))

    # Monotone in mutation count: a bigger change may not be judged more harshly
    # than a smaller one just because the null happened to be narrower there.
    for tbl in caps.values():
        run = 0.0
        for k in sorted(tbl, key=int):
            run = max(run, tbl[k])
            tbl[k] = round(run, 3)

    out = {
        "caps": caps,
        "null": null,
        "self_consistency": {"per_set": self_r, "floor_at_quantile": floor},
        "meta": {
            "wt_pdb": args.wt_pdb, "L": L, "counts": counts, "reps": args.reps,
            "self_reps": args.self_reps, "quantile": args.quantile,
            "min_cap": args.min_cap, "seed": args.seed,
            "protected_sets": {k: len(v) for k, v in protected_sets.items()},
            "allowed_positions": len(allowed),
            "note": "caps are unclamped; 11_generate_pipelineB.py applies "
                    "--core-rmsd-ceiling / --interface-rmsd-ceiling",
        },
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("[22] caps:", json.dumps(caps), flush=True)
    print("DONE_CAP_CALIBRATION", flush=True)


if __name__ == "__main__":
    main()
