#!/usr/bin/env python
"""Stage 6b/6c: fold WT + designs with ESMFold, compute active-site CA-RMSD.

Runs in the `esmfold` env. Reads candidates.json (from 11_generate.py), folds the
WT and every design sequence, writes PDBs into <structures>/, and augments each
design with mean pLDDT + active-site CA-RMSD to WT (Kabsch superposition over the
frozen active-site residues, the catalytic-geometry gate of the spec). Writes
folded.json (design_id -> {plddt, active_site_rmsd, structure_file}).
"""
import sys, os, json, argparse
from pathlib import Path
import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def ca_coords(pdb_text):
    """Return dict {resseq(1-based): np.array([x,y,z])} of CA atoms, in order."""
    out = {}
    for ln in pdb_text.splitlines():
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
            resseq = int(ln[22:26])
            out[resseq] = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    return out


def kabsch_rmsd(P, Q):
    """RMSD after optimal superposition (P,Q are Nx3, matched order)."""
    if len(P) < 3:
        return None
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    H = Pc.T @ Qc
    V, S, Wt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Wt.T @ V.T))
    D = np.diag([1, 1, d])
    R = Wt.T @ D @ V.T
    Pr = Pc @ R.T
    return float(np.sqrt(((Pr - Qc) ** 2).sum(1).mean()))


# Metal-coordinating / common catalytic residue types. The catalytic-core RMSD is
# taken over the active-site positions whose WT residue is one of these — the rigid
# functional geometry (e.g. the laccase copper His/Cys ligands), which should barely
# move in a good design even when flexible active-site loops flex. Reported alongside
# the full active-site RMSD; the full-set RMSD remains the hard gate.
CORE_RESTYPES = set("HCMDE")


def rmsd_over(wt_ca, dca, positions):
    idx = [p for p in positions if p in wt_ca and p in dca]
    if len(idx) < 3:
        return None
    P = np.array([dca[p] for p in idx])
    Q = np.array([wt_ca[p] for p in idx])
    return kabsch_rmsd(P, Q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", help="candidates.json (design folding mode)")
    ap.add_argument("--wt-only", help="a bare WT sequence: fold ONLY wt.pdb and exit "
                    "(run before annotation, which needs the WT structure)")
    ap.add_argument("--structures", required=True, help="output dir for PDBs")
    ap.add_argument("--out", help="folded.json path (design mode)")
    ap.add_argument("--max-len", type=int, default=1000)
    ap.add_argument("--rmsd-cap", type=float, default=2.0,
                    help="hard cap on full active-site CA-RMSD (A); designs above this "
                         "get passes_rmsd=False (fold-collapse / classifier gaming)")
    ap.add_argument("--core-rmsd-cap", type=float, default=1.5,
                    help="hard cap on catalytic-core CA-RMSD (A), the rigid metal/catalytic "
                         "geometry; tighter than the full-set cap. A design must pass BOTH.")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, EsmForProteinFolding
    sd = Path(args.structures); sd.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1",
                                                 low_cpu_mem_usage=True).cuda().eval()
    model.esm = model.esm.half()

    @torch.no_grad()
    def fold(seq):
        seq = seq[:args.max_len]
        ids = tok([seq], return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
        out = model(ids)
        plddt = out["plddt"][0, :, 1].mean().item()
        pdb = model.output_to_pdb(out)[0]
        return pdb, plddt

    # --- WT-only mode: fold wt.pdb and exit (runs before Stage-3 annotation) ---
    if args.wt_only:
        wt_pdb, wt_plddt = fold(args.wt_only.strip())
        (sd / "wt.pdb").write_text(wt_pdb)
        print(f"[11b] WT-only fold plddt={wt_plddt:.3f} -> {sd/'wt.pdb'}", flush=True)
        return

    cand = json.loads(Path(args.candidates).read_text())
    active = cand.get("active_site", [])  # 1-based
    wt_seq = cand["wt_sequence"]
    # catalytic-core positions = active-site residues whose WT type coordinates metals /
    # is commonly catalytic (rigid functional geometry, tighter tolerance)
    core = [p for p in active if 1 <= p <= len(wt_seq) and wt_seq[p - 1] in CORE_RESTYPES]
    # reuse a pre-folded WT if present (folded once in the WT-only pre-step)
    wt_path = sd / "wt.pdb"
    if wt_path.exists():
        wt_pdb = wt_path.read_text()
        wt_plddt = None
        print("[11b] reusing existing wt.pdb", flush=True)
    else:
        wt_pdb, wt_plddt = fold(cand["wt_sequence"])
        wt_path.write_text(wt_pdb)
        print(f"[11b] WT folded plddt={wt_plddt:.3f}", flush=True)
    wt_ca = ca_coords(wt_pdb)

    folded = {"wt": {"plddt": (round(wt_plddt, 3) if wt_plddt is not None else None),
                     "structure_file": "wt.pdb"}}
    for d in cand["designs"]:
        did, seq = d["design_id"], d["sequence"]
        pdb, plddt = fold(seq)
        (sd / f"{did}.pdb").write_text(pdb)
        dca = ca_coords(pdb)
        # full active-site CA-RMSD (the hard gate); fall back to global if no AS annotated
        rmsd = rmsd_over(wt_ca, dca, active) if active else \
               rmsd_over(wt_ca, dca, sorted(wt_ca))
        core_rmsd = rmsd_over(wt_ca, dca, core) if core else None
        # hard gate: a design whose active-site backbone deviates past the cap is a
        # fold-collapse / classifier-gaming reject, regardless of classifier score. Two
        # gates, both must hold: the full active-site set (fold integrity, looser cap)
        # AND the rigid catalytic core (functional geometry, tighter cap). The core gate
        # is skipped only when no catalytic-core residues were annotated (core is None).
        passes_full = (rmsd is not None and rmsd <= args.rmsd_cap)
        passes_core = (core_rmsd is None) or (core_rmsd <= args.core_rmsd_cap)
        passes = passes_full and passes_core
        folded[did] = {"plddt": round(plddt, 3),
                       "active_site_rmsd": (round(rmsd, 3) if rmsd is not None else None),
                       "catalytic_core_rmsd": (round(core_rmsd, 3) if core_rmsd is not None else None),
                       "passes_rmsd": passes, "passes_full": passes_full, "passes_core": passes_core,
                       "rmsd_cap": args.rmsd_cap, "core_rmsd_cap": args.core_rmsd_cap,
                       "structure_file": f"{did}.pdb"}
        print(f"[11b] {did} plddt={plddt:.3f} as_rmsd={rmsd} core_rmsd={core_rmsd} "
              f"passes={passes} (full<={args.rmsd_cap}:{passes_full} core<={args.core_rmsd_cap}:{passes_core})",
              flush=True)

    Path(args.out).write_text(json.dumps(folded, indent=2))
    print(f"[11b] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
