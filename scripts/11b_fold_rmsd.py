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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--structures", required=True, help="output dir for PDBs")
    ap.add_argument("--out", required=True, help="folded.json path")
    ap.add_argument("--max-len", type=int, default=1000)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, EsmForProteinFolding
    cand = json.loads(Path(args.candidates).read_text())
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

    active = cand.get("active_site", [])  # 1-based
    wt_pdb, wt_plddt = fold(cand["wt_sequence"])
    (sd / "wt.pdb").write_text(wt_pdb)
    wt_ca = ca_coords(wt_pdb)
    print(f"[11b] WT folded plddt={wt_plddt:.3f}", flush=True)

    folded = {"wt": {"plddt": round(wt_plddt, 3), "structure_file": "wt.pdb"}}
    for d in cand["designs"]:
        did, seq = d["design_id"], d["sequence"]
        pdb, plddt = fold(seq)
        (sd / f"{did}.pdb").write_text(pdb)
        dca = ca_coords(pdb)
        # active-site CA-RMSD (matched by residue number; falls back to global if no AS)
        idx = [p for p in active if p in wt_ca and p in dca] or \
              [p for p in wt_ca if p in dca]
        P = np.array([dca[p] for p in idx])
        Q = np.array([wt_ca[p] for p in idx])
        rmsd = kabsch_rmsd(P, Q)
        folded[did] = {"plddt": round(plddt, 3),
                       "active_site_rmsd": (round(rmsd, 3) if rmsd is not None else None),
                       "structure_file": f"{did}.pdb"}
        print(f"[11b] {did} plddt={plddt:.3f} as_rmsd={rmsd}", flush=True)

    Path(args.out).write_text(json.dumps(folded, indent=2))
    print(f"[11b] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
