#!/usr/bin/env python
"""Assemble results.json in the webapp contract from the stage outputs.

Merges one-or-more candidates.json (per phenotype) + their folded.json + the shared
mpnn.json into the exact bundle make_demo_results documents:
  results.json { wt_structure, wt_sequence, by_phenotype: { ph: [ {design_id, sequence,
      classifier_score, active_site_rmsd, n_mutations, structure_file, metrics{}, track{}} ] } }
Runs in any env (stdlib only).
"""
import sys, json, argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", required=True, help="candidates.json per phenotype")
    ap.add_argument("--folded", nargs="+", default=None,
                    help="folded.json per phenotype (same order); Pipeline A only. "
                         "Pipeline B candidates carry per-design refold RMSDs in-line "
                         "(rmsds{active_site, iface_*}), so --folded may be omitted or "
                         "shorter than --candidates — missing entries fall back to the "
                         "in-loop gate values.")
    ap.add_argument("--mpnn", required=True, help="shared mpnn.json")
    ap.add_argument("--out", required=True, help="results.json")
    args = ap.parse_args()

    mpnn = json.loads(Path(args.mpnn).read_text())
    by_phenotype = {}
    wt_seq = None
    folded_paths = list(args.folded or [])
    # pad with None so zip lines up; a candidate without a folded.json will
    # fall back to per-design rmsds carried inside the candidate row itself.
    while len(folded_paths) < len(args.candidates):
        folded_paths.append(None)
    for cpath, fpath in zip(args.candidates, folded_paths):
        cand = json.loads(Path(cpath).read_text())
        fold = json.loads(Path(fpath).read_text()) if fpath else {}
        # Pipeline B: derive a folded-shape dict from candidate rows themselves.
        # Each row already carries rmsds{active_site, iface_*} from the in-loop
        # gate; passes_rmsd is True because non-passing designs are rejected
        # BEFORE ranking (Tier-1 hard gate). plddt isn't stored per-design in B,
        # so leave it None — the webapp already handles missing plddt.
        if not fold and cand.get("pipeline") == "B":
            core_cap = cand.get("core_rmsd_cap")
            iface_cap = cand.get("interface_rmsd_cap")
            for d in cand.get("designs", []):
                rmsds = d.get("rmsds", {}) or {}
                core_rmsd = rmsds.get("active_site")
                fold[d["design_id"]] = {
                    "plddt": None,
                    "active_site_rmsd": core_rmsd,       # webapp .active_site_rmsd
                    "catalytic_core_rmsd": core_rmsd,    # metrics.catalytic_core_rmsd
                    "passes_rmsd": True,                 # survivors passed Tier-1 gate
                    "rmsd_cap": iface_cap,
                    "core_rmsd_cap": core_cap,
                    "structure_file": f"{d['design_id']}.pdb",
                }
        wt_seq = cand["wt_sequence"]
        ph = cand["phenotype"]
        cons = cand.get("conservation", [])
        active = cand.get("active_site", [])
        assigned = cand.get("active_site_assigned", False)
        # §16b interface constraints (may be absent on older candidates.json)
        iface_faces = cand.get("interfaces", {}) or {}
        # Flat 1-based union across all populated faces, for the .iface channel:
        iface_positions_set = set()
        # Per-position -> list of face labels for tooltip aggregation:
        iface_by_pos = {}
        for face_label, face in iface_faces.items():
            for p in face.get("positions", []):
                iface_positions_set.add(int(p))
                iface_by_pos.setdefault(int(p), []).append(face_label)
        iface_positions = sorted(iface_positions_set)
        # keys as strings so jsonification preserves them across the tojson filter
        iface_by_pos_str = {str(k): v for k, v in iface_by_pos.items()}
        rows = []
        for d in cand["designs"]:
            did = d["design_id"]
            ff = fold.get(did, {})
            track = dict(seq=d["sequence"], wt=wt_seq, conservation=cons,
                         active_site=active, active_site_assigned=assigned,
                         mutations=d.get("mutations", []),
                         interfaces=iface_positions,
                         interfaces_by_position=iface_by_pos_str)
            metrics = {
                "biophysical_score": d.get("biophysical_score"),
                "plddt": ff.get("plddt"),
                "catalytic_core_rmsd": ff.get("catalytic_core_rmsd"),
                "passes_rmsd": ff.get("passes_rmsd"),
                "rmsd_cap": ff.get("rmsd_cap"),
                "core_rmsd_cap": ff.get("core_rmsd_cap"),
                "mpnn_model": mpnn.get("model_type"),
                "wt_mpnn_confidence": mpnn.get("wt_mpnn_confidence"),
                "n_msa_hits": cand.get("n_msa_hits"),
                "wt_classifier_score": cand.get("wt_classifier_score"),
            }
            rows.append(dict(design_id=did, sequence=d["sequence"],
                             classifier_score=d.get("classifier_score"),
                             active_site_rmsd=ff.get("active_site_rmsd"),
                             passes_rmsd=ff.get("passes_rmsd", True),
                             n_mutations=d.get("n_mutations"),
                             structure_file=ff.get("structure_file", f"{did}.pdb"),
                             metrics=metrics, track=track))
        # rank: structurally-valid designs (passes_rmsd) first, then by classifier score.
        # A gaming design with a high score but a collapsed active site sinks below a
        # lower-scoring but structurally-sound one instead of topping the list.
        rows.sort(key=lambda r: (bool(r.get("passes_rmsd")), r["classifier_score"] or 0),
                  reverse=True)
        by_phenotype[ph] = rows

    out = dict(wt_structure="wt.pdb", wt_sequence=wt_seq, by_phenotype=by_phenotype)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[11d] wrote {args.out}: {sum(len(v) for v in by_phenotype.values())} designs "
          f"across {len(by_phenotype)} phenotype(s)", flush=True)


if __name__ == "__main__":
    main()
