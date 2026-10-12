#!/usr/bin/env python
"""Pipeline B: MPNN-in-the-loop generation with PLM filter + in-loop fold gate.

Contrast with Pipeline A (scripts/11_generate.py):
    A: PLM proposes -> Metropolis-annealed accept -> periodic ESMFold gate ->
       final one-shot MPNN gate on the WT backbone (single confidence copied
       to every design). Sequence-first, structure-check-second.
    B: MPNN proposes (structure-first, sequence-second). Per round, LigandMPNN
       generates K candidates on the WT backbone with active-site + interface
       residues held fixed and per-phenotype bias_AA. Each candidate then
       passes a two-tier oracle:
         Tier 1 (hard gates, any fail => reject):
           - active-site CA-RMSD <= --core-rmsd-cap
           - per-interface CA-RMSD <= --interface-rmsd-cap
           - no mutation at conservation c_i > --conservation-freeze
           - refolded pLDDT >= --min-plddt (via 11e worker or on-demand ESMFold)
         Tier 2 (soft product for RANKING survivors):
           score = p_classifier * seqLL_ratio * p_coupling
             p_classifier: sigmoid(head(mean-pool(esm(design))))  in [0,1]
             seqLL_ratio : exp(seqLL(design) - seqLL(WT))         in ~[0,1]
             p_coupling  : frac of WT strong-contact pairs preserved  in [0,1]
       All four sub-scores are emitted per design so ranking can be re-run
       offline with different weight schemes.

    Every candidate is folded (or looked up if already folded within the
    round-local cache) -- Tier 1 uses that fold. This matches the pipeline
    diagram's "fold every Tier-2 survivor" cadence except we fold every Tier-1
    check (which is cheaper to reason about; the diagram's ordering is a
    performance optimization that assumes soft-score correlation with fold
    quality that we haven't measured yet).

Compatibility: writes candidates_<pheno>.json in the SAME schema as
Pipeline A so 11d_assemble.py, 11b_fold_rmsd.py (for offline re-fold),
and downstream harvest tooling work unchanged. Sub-score details go in
designs[*].subscores.

MPNN integration: LigandMPNN lives in a separate conda env (`ligandmpnn`),
so each round shells out to `conda run -n <mpnn-env> python <mpnn-repo>/run.py`
with --fixed_residues, --bias_AA, --temperature, --number_of_batches=1,
--batch_size=K. FASTA output is parsed for the K design sequences.
"""

from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np


# ---------- shared helpers imported from 11_generate.py ----------
# Rather than duplicate them we import from the sibling script; both live in
# scripts/ so add that dir to sys.path.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from importlib import import_module
_gen_mod = import_module("11_generate")  # noqa: E402
_ca_coords = _gen_mod._ca_coords
_kabsch_rmsd = _gen_mod._kabsch_rmsd
_rmsd_over = _gen_mod._rmsd_over
RefoldClient = _gen_mod.RefoldClient
biophysical_score = _gen_mod.biophysical_score
run_msa_conservation = _gen_mod.run_msa_conservation
detect_active_site = _gen_mod.detect_active_site


AA20 = list("ACDEFGHIKLMNPQRSTVWY")


def contact_graph_from_pdb(pdb_text, chain="A", cutoff=8.0, min_sep=6,
                           plddt=None, plddt_min=None):
    """Cbeta-Cbeta contact graph from PDB text.

    Glycine has no Cbeta, so CA stands in for it -- as it does for
    backbone-only files such as the LigandMPNN scaffolds.  ``min_sep`` drops
    sequence-local contacts, which carry no coupling information; it matches
    ``modeling.contact_min_sep`` in config.yaml.

    When ``plddt`` is supplied (a predicted model), a contact is dropped unless
    *both* residues clear ``plddt_min``.  A conserved residue whose only
    partners are low-confidence therefore ends up with no partner, and the
    caller freezes it -- the conservative direction.
    """
    cb, ca = {}, {}
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM") or len(line) < 54 or line[21] != chain:
            continue
        name = line[12:16].strip()
        if name not in ("CA", "CB"):
            continue
        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        (cb if name == "CB" else ca)[int(line[22:26])] = xyz
    # CB where it exists, CA otherwise -- covers glycine and backbone-only PDBs
    # such as the LigandMPNN scaffolds, which carry no side-chain atoms at all.
    coords = {**ca, **cb}
    keys = sorted(coords)
    if not keys:
        return {}
    X = np.asarray([coords[k] for k in keys], dtype=float)
    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    idx = np.arange(len(keys))
    near = (d < cutoff) & (np.abs(idx[:, None] - idx[None, :]) >= min_sep)
    if plddt is not None and plddt_min is not None:
        ok = np.asarray([float(plddt.get(k, 0.0)) >= float(plddt_min) for k in keys])
        near &= ok[:, None] & ok[None, :]
    g = {k: set() for k in keys}
    for a, b in zip(*np.where(np.triu(near, 1))):
        g[keys[a]].add(keys[b]); g[keys[b]].add(keys[a])
    return {k: v for k, v in g.items() if v}


def _plddt_from_pdb(pdb_text, chain="A"):
    """Per-residue pLDDT from the B-factor column of a predicted model."""
    out = {}
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and len(line) >= 66 and line[21] == chain \
                and line[12:16].strip() == "CA":
            try:
                out[int(line[22:26])] = float(line[60:66])
            except ValueError:
                pass
    return out


def ensure_wt_structure(seq, wt_pdb, refold_workdir, ready_timeout=1200.0,
                        fold_timeout=900.0):
    """Guarantee a wild-type structure file, folding one if none is supplied.

    ``--wt-pdb`` is load-bearing in three places -- the LigandMPNN backbone,
    the CA-RMSD reference for every Tier-1a refold, and the Cbeta contact
    graph behind the coupling-aware conservation mask -- so an input enzyme
    with no experimental structure cannot simply skip it.  Rather than
    degrade each consumer separately (the old behaviour silently dropped the
    coupling mask back to a blanket conservation freeze), fold the wild type
    once up front through the same persistent ESMFold worker the in-loop gate
    uses, and hand that single model to all three.

    Done here, before ``RefoldClient`` is constructed, because the client
    reads the WT structure in ``__init__`` to cache its CA coordinates.

    Returns ``(path, provenance)``.  Provenance is persisted to results.json
    so a campaign run on a predicted backbone is never mistaken for one run
    on a crystal structure: every RMSD in it is measured against a model.
    """
    if wt_pdb and Path(wt_pdb).exists():
        return str(wt_pdb), {"source": "experimental", "path": str(wt_pdb)}
    wd = Path(refold_workdir)
    req, resp = wd / "requests", wd / "responses"
    t0 = time.time()
    while not (wd / "READY").exists():
        if time.time() - t0 > ready_timeout:
            raise SystemExit("[11B] no WT structure supplied and the refold "
                             "worker never became READY -- cannot bootstrap")
        time.sleep(0.5)
    print("[11B] no experimental WT structure -- folding the wild type once "
          "for backbone + RMSD reference + contact map", flush=True)
    rid = uuid.uuid4().hex[:12]
    tmp = req / f"{rid}.fasta.tmp"
    tmp.write_text(seq)
    tmp.rename(req / f"{rid}.fasta")                 # atomic submit
    out, err = resp / f"{rid}.pdb", resp / f"{rid}.err"
    t0 = time.time()
    while True:
        if out.exists():
            pdb_text = out.read_text()
            out.unlink(missing_ok=True)
            break
        if err.exists():
            err.unlink(missing_ok=True)
            raise SystemExit("[11B] WT fold failed; cannot bootstrap a structure")
        if time.time() - t0 > fold_timeout:
            raise SystemExit("[11B] WT fold timed out; cannot bootstrap a structure")
        time.sleep(0.5)
    dest = wd / "wt_predicted.pdb"
    dest.write_text(pdb_text)
    pl = _plddt_from_pdb(pdb_text, chain="A")
    mean_pl = float(np.mean(list(pl.values()))) if pl else 0.0
    print(f"[11B] WT folded: {dest} (mean pLDDT {mean_pl:.1f})", flush=True)
    return str(dest), {"source": "predicted", "model": "esmfold",
                       "path": str(dest), "mean_plddt": round(mean_pl, 2),
                       "n_residues": len(pl)}


# =====================================================================
# LigandMPNN subprocess wrapper
# =====================================================================
def _mpnn_fasta_parse(fa_path: Path) -> list[dict]:
    """Parse a LigandMPNN output FASTA.

    Header format (LigandMPNN 2024+):
        >T=..., id=..., seed=..., overall_confidence=..., ligand_confidence=..., ...
    Each design gets one header + one sequence line. Returns a list of dicts:
        {"sequence": str, "T": float|None, "overall_confidence": float|None,
         "ligand_confidence": float|None, "id": int|None}
    The first record is the input (WT threading) and MUST be filtered out by
    the caller if it's not needed.
    """
    out: list[dict] = []
    header = None
    for ln in fa_path.read_text().splitlines():
        if ln.startswith(">"):
            header = ln[1:]
        elif ln.strip() and header is not None:
            rec: dict = {"sequence": ln.strip()}
            for k in ("T", "overall_confidence", "ligand_confidence", "id", "seed"):
                m = re.search(rf"{k}=([-0-9.]+)", header)
                if m:
                    try:
                        rec[k] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
                    except ValueError:
                        rec[k] = None
            out.append(rec)
            header = None
    return out


def mpnn_propose(
    *,
    mpnn_repo: str,
    mpnn_env: str,
    wt_pdb: str,
    design_chain: str,
    fixed_positions_1b: list[int],  # 1-based positions in design chain
    bias_AA_vec: dict[str, float],  # e.g., {"A": +0.31, ...}
    temperature: float,
    n_designs: int,
    workdir: Path,
    seed: int,
    model_type: str = "ligand_mpnn",
    timeout_s: int = 1800,
    bias_AA_per_residue: dict | None = None,  # {"A12": {"G": 5.0}, ...}
) -> tuple[list[dict], str]:
    """Run one LigandMPNN batch. Returns (parsed_records, stderr_tail).

    Records exclude the WT input threading (LigandMPNN emits it as the first
    FASTA entry with the seed=0 id).
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    # LigandMPNN wants "A12 A13 A14" (space-separated CHAIN+resid).
    fixed_str = " ".join(f"{design_chain}{p}" for p in fixed_positions_1b)
    bias_str = ",".join(f"{a}:{v:.4f}" for a, v in bias_AA_vec.items() if abs(v) > 1e-6)

    cmd = [
        "conda", "run", "-n", mpnn_env, "--no-capture-output",
        "python", "run.py",
        "--model_type", model_type,
        "--seed", str(seed),
        "--pdb_path", str(Path(wt_pdb).resolve()),
        "--out_folder", str(workdir.resolve()),
        "--number_of_batches", "1",
        "--batch_size", str(n_designs),
        "--temperature", f"{temperature:.4f}",
        "--chains_to_design", design_chain,
    ]
    if fixed_str:
        cmd += ["--fixed_residues", fixed_str]
    if bias_str:
        cmd += ["--bias_AA", bias_str]
    if bias_AA_per_residue:
        # LigandMPNN expects a JSON path with {"A12": {"G": 5.0}, ...}
        bias_pr_path = workdir / "bias_AA_per_residue.json"
        bias_pr_path.write_text(json.dumps(bias_AA_per_residue))
        cmd += ["--bias_AA_per_residue", str(bias_pr_path.resolve())]

    t0 = time.time()
    r = subprocess.run(cmd, cwd=mpnn_repo, capture_output=True, text=True, timeout=timeout_s)
    elapsed = time.time() - t0
    tail = (r.stdout + r.stderr)[-1500:]
    if r.returncode != 0:
        raise RuntimeError(
            f"LigandMPNN failed (exit {r.returncode}, {elapsed:.1f}s):\n{tail}"
        )

    # Parse the emitted FASTA
    fa_dir = workdir / "seqs"
    fa_files = sorted(fa_dir.glob("*.fa"))
    if not fa_files:
        raise RuntimeError(f"LigandMPNN produced no FASTA in {fa_dir}\n{tail}")
    recs = _mpnn_fasta_parse(fa_files[-1])
    # first record is input threading; drop it iff it's WT-length and has seed 0 / id 0
    if recs and recs[0].get("id", 0) == 0:
        recs = recs[1:]
    print(f"[11B-mpnn] T={temperature:.3f} n={len(recs)} in {elapsed:.1f}s", flush=True)
    return recs, tail


# =====================================================================
# bias_AA loader
# =====================================================================
def load_bias_AA(path: str, phenotype: str) -> dict[str, float]:
    """Load per-phenotype bias_AA vector from data/bias_aa_by_phenotype.json.

    Returns {AA: log_ratio}. Missing phenotypes raise; missing AAs default 0.
    """
    obj = json.loads(Path(path).read_text())
    if "bias_AA" not in obj:
        raise SystemExit(f"[11B] {path} missing 'bias_AA' key")
    if phenotype not in obj["bias_AA"]:
        available = list(obj["bias_AA"].keys())
        raise SystemExit(
            f"[11B] phenotype '{phenotype}' not in bias_AA (have {available})"
        )
    vec = {aa: float(obj["bias_AA"][phenotype].get(aa, 0.0)) for aa in AA20}
    print(f"[11B] loaded bias_AA[{phenotype}] from {path} "
          f"(reference={obj.get('reference','?')}, "
          f"pseudocount={obj.get('pseudocount','?')})",
          flush=True)
    return vec


# =====================================================================
# oracle: classifier, sequence-LL, coupling
# =====================================================================
def build_oracle(model, tok, head, device: str, wt_seq: str):
    """Return (score_classifier, seq_loglik, contact_preservation, wt_contact_pairs).

    All callables are @torch.no_grad and expect a plain string sequence.
    """
    import torch

    mask_id = tok.mask_token_id
    L = len(wt_seq)

    def encode(s: str):
        return tok(s, return_tensors="pt", add_special_tokens=True).to(device)

    @torch.no_grad()
    def score_classifier(s: str) -> float:
        enc = encode(s)
        out = model.base_model.model.esm(
            enc["input_ids"], attention_mask=enc["attention_mask"]
        )
        h = out.last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
        logit = head(pooled).squeeze(-1)
        return float(torch.sigmoid(logit).item())

    @torch.no_grad()
    def seq_loglik(s: str) -> float:
        """Sequence log-likelihood proxy: single forward on the UNMASKED sequence,
        Σ_i log P(x_i) at each position, averaged over length.

        This is NOT a true pseudo-LL (which would mask each position in turn --
        L forward passes per sequence, cost-prohibitive for K designs * R rounds).
        It is a cheap surrogate that ranks sequences by how well the adapter
        anticipates them under full context. Diagnostic value: seqLL_design -
        seqLL_WT quantifies "adapter surprise" relative to WT. Ratio in the
        Tier-2 product uses exp of this diff, clipped to [0, inf).
        """
        enc = encode(s)
        out = model(enc["input_ids"], attention_mask=enc["attention_mask"])
        logits = out.logits[0]  # (T, V)
        # residue positions map to token positions 1..L (CLS at 0)
        ids = enc["input_ids"][0]
        # avg log softmax over residue positions
        logp = torch.log_softmax(logits, dim=-1)
        ll = 0.0
        n = 0
        for i in range(L):
            tid = int(ids[i + 1])
            ll += float(logp[i + 1, tid].item())
            n += 1
        return ll / max(1, n)

    # WT contact pairs (computed once)
    wt_contact_pairs: list[tuple[int, int]] = []
    wt_contact_map = None
    try:
        from eptrans.modeling.data import _predict_contacts
        from eptrans.modeling import masking as MK
        wt_contact_map = _predict_contacts(model, tok, wt_seq)
        wt_contact_pairs = MK.contact_pairs_from_map(
            wt_contact_map, threshold=0.5, min_sep=6, top_k=128
        )
    except Exception as e:
        print(f"[11B-oracle] WARN wt contact map failed ({type(e).__name__}: {e}); "
              f"coupling term set to 1.0 for every design", flush=True)
        if os.environ.get("GEN_B_REQUIRE_COUPLING", "1") == "1":
            # Silently neutralising the coupling factor makes score_product a
            # two-term product while the logs and the output JSON still advertise
            # three. That happened across every production run before 2026-10-12.
            # Fail loudly by default; set GEN_B_REQUIRE_COUPLING=0 to opt out.
            raise RuntimeError(
                "WT contact map unavailable, so the coupling factor would be a "
                "constant 1.0 and contribute nothing to ranking. Refusing to run "
                "with a silently-degraded objective. Set GEN_B_REQUIRE_COUPLING=0 "
                "to proceed with an explicitly two-term score."
            ) from e

    @torch.no_grad()
    def coupling_preservation(s: str) -> float:
        """Fraction of WT strong-contact pairs whose predicted contact prob in
        the design's contact map >= 0.5. Equal identity => 1.0. Missing map
        (fallback) => 1.0 (neutral)."""
        if wt_contact_map is None or not wt_contact_pairs:
            return 1.0
        try:
            from eptrans.modeling.data import _predict_contacts
            cm = _predict_contacts(model, tok, s)
        except Exception as e:
            print(f"[11B-oracle] WARN design contact failed ({type(e).__name__}); "
                  f"coupling=1.0", flush=True)
            return 1.0
        kept = 0
        for i, j in wt_contact_pairs:
            if 0 <= i < cm.shape[0] and 0 <= j < cm.shape[1] and float(cm[i, j]) >= 0.5:
                kept += 1
        return kept / max(1, len(wt_contact_pairs))

    return score_classifier, seq_loglik, coupling_preservation, wt_contact_pairs


# =====================================================================
# main
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    # ---- I/O ----
    ap.add_argument("--seq", required=True, help="WT sequence (design chain, mature)")
    ap.add_argument("--phenotype", required=True)
    ap.add_argument("--out", required=True, help="output candidates_<pheno>.json")
    ap.add_argument("--mlm-adapter", required=True)
    ap.add_argument("--head", required=True, help="cached head_best.pt")
    ap.add_argument("--uniref-db", required=True)
    ap.add_argument("--transfer-json", default="")
    ap.add_argument("--bias-aa-json", required=True,
                    help="data/bias_aa_by_phenotype.json (from prep_bias_aa.py)")
    ap.add_argument("--backbone-size", default="3B")

    # ---- MPNN ----
    ap.add_argument("--mpnn-repo", required=True,
                    help="LigandMPNN repo dir (contains run.py + model_params/)")
    ap.add_argument("--mpnn-env", default="ligandmpnn",
                    help="conda env for LigandMPNN")
    ap.add_argument("--mpnn-model", default="ligand_mpnn",
                    choices=["ligand_mpnn", "protein_mpnn"])
    ap.add_argument("--n-designs", type=int, default=3,
                    help="TOTAL number of surviving designs to keep (across rounds)")
    ap.add_argument("--n-rounds", type=int, default=4,
                    help="MPNN batches per phenotype")
    ap.add_argument("--batch-per-round", type=int, default=6,
                    help="MPNN candidates proposed per round (each folded+scored)")
    ap.add_argument("--temperatures", default="0.05,0.10,0.20",
                    help="comma-separated MPNN temperatures cycled across rounds")

    # ---- interfaces / active site (same semantics as Pipeline A) ----
    ap.add_argument("--wt-pdb", default="",
                    help="cached WT structure for MPNN backbone + Tier-1 refolds "
                         "+ coupling-mask contact graph. If omitted or missing, "
                         "the WT is folded once with ESMFold and that model is "
                         "used for all three (recorded in results.json)")
    ap.add_argument("--additional-constraints", default="")
    ap.add_argument("--complex-cif", default="")
    ap.add_argument("--design-chain", default="A")
    ap.add_argument("--interface-contact-cutoff", type=float, default=4.5)
    ap.add_argument("--interface-rmsd-cap", type=float, default=1.5)
    ap.add_argument("--core-rmsd-cap", type=float, default=1.0)
    ap.add_argument("--interfaces-json-out", default="")
    # Adaptive caps, clamped.  --*-rmsd-cap stays the floor used when no
    # calibration table is supplied; --*-rmsd-ceiling is a hard clamp that the
    # calibrated cap may never exceed, so a wide null distribution cannot widen
    # the gate without bound.
    ap.add_argument("--rmsd-cap-table", default="",
                    help="JSON written by 22_calibrate_rmsd_caps.py; makes the cap a "
                         "function of mutation count")
    ap.add_argument("--core-rmsd-ceiling", type=float, default=1.5)
    ap.add_argument("--interface-rmsd-ceiling", type=float, default=2.0)

    # ---- Tier 1 gates ----
    ap.add_argument("--conservation-freeze", type=float, default=0.90)
    ap.add_argument("--coupling-mask", choices=["off", "closure", "comutate"],
                    default="closure",
                    help="off = blanket freeze of every conserved position (legacy); "
                         "closure = conserved positions that have a structural partner "
                         "stay designable and their partners are opened too; "
                         "comutate = closure, plus Tier 1a rejects a design that moved "
                         "one member of a conserved contact pair and left the other")
    ap.add_argument("--contact-cutoff", type=float, default=8.0,
                    help="Cbeta-Cbeta distance defining a WT contact, in angstrom")
    ap.add_argument("--contact-plddt-min", type=float, default=70.0,
                    help="when the WT structure is predicted rather than "
                         "experimental, ignore contacts unless both residues "
                         "clear this pLDDT; their conserved positions then stay "
                         "frozen")
    ap.add_argument("--min-plddt", type=float, default=0.0,
                    help="if the refold worker returns pLDDT, drop designs below this "
                         "(default 0 disables)")

    # ---- fold worker ----
    ap.add_argument("--refold-workdir", required=True,
                    help="11e_esmfold_worker queue dir; a running worker is required")

    # ---- survivor-conditioned iteration (v2 knob; default OFF) ----
    # When --survivor-topk > 0, after round r we take the top-K survivors so
    # far by score_product; positions where >= --survivor-min-agreement of the
    # top-K agree on a non-WT AA get a per-residue bias of
    # +survivor-bias-strength log-odds toward that AA in round r+1's MPNN call.
    # This turns the plain iid loop into a soft-hill-climb (an actual MPNN-in-
    # the-loop feedback, matching what the script's docstring already claims).
    ap.add_argument("--survivor-topk", type=int, default=0,
                    help="top-K survivors used to bias later rounds; 0=disable "
                         "(default keeps legacy iid behavior)")
    ap.add_argument("--survivor-min-agreement", type=int, default=2,
                    help="min top-K survivors that must share a non-WT AA at a "
                         "position for that position to be biased next round")
    ap.add_argument("--survivor-bias-strength", type=float, default=5.0,
                    help="per-residue bias log-odds applied to consensus AA")
    ap.add_argument("--survivor-warmup-rounds", type=int, default=2,
                    help="draw survivors from all rounds >= this; earlier "
                         "rounds are iid warmup so we have a survivor pool")

    # ---- misc ----
    ap.add_argument("--coupling-mode", default="both",
                    choices=["none", "contact", "span", "both"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--seed", type=int, default=1466)
    args = ap.parse_args()

    seq = args.seq.strip().upper()
    L = len(seq)
    rng = np.random.default_rng(args.seed)
    workdir = Path(args.workdir); workdir.mkdir(parents=True, exist_ok=True)

    # Resolve the WT structure before anything consumes it: MPNN backbone,
    # Tier-1a RMSD reference, and the coupling-mask contact graph all read it.
    args.wt_pdb, wt_struct_prov = ensure_wt_structure(
        seq, args.wt_pdb, args.refold_workdir)

    import torch
    torch.manual_seed(args.seed)

    # ---- Stage A: conservation + active-site (reuse Pipeline A helpers) ----
    cons, n_hits = run_msa_conservation(seq, args.uniref_db, str(workdir))
    transferred = []
    transfer_meta = {}
    if args.transfer_json and Path(args.transfer_json).exists():
        tj = json.loads(Path(args.transfer_json).read_text())
        transferred = tj.get("transferred", [])
        transfer_meta = {k: tj.get(k) for k in ("by_source", "n_foldseek_hits", "n_mmseqs_hits")}
    frozen, active_1b, assigned, src_counts = detect_active_site(
        seq, cons, freeze_thresh=args.conservation_freeze, transferred=transferred
    )
    print(f"[11B] active-site: {len(active_1b)} frozen (assigned={assigned}); "
          f"sources {src_counts}", flush=True)

    # ---- Stage A': interfaces (§16b) ----
    from eptrans.interfaces import (
        parse_explicit_constraints,
        resolve_interfaces_from_complex,
    )
    interfaces: dict[str, dict] = {}
    added_frozen: set[int] = set()
    if args.additional_constraints:
        explicit_toks, nl_phrases = [], []
        for tok in args.additional_constraints.split(","):
            t = tok.strip()
            if not t: continue
            (explicit_toks if any(ch.isdigit() for ch in t) else nl_phrases).append(t)
        if explicit_toks:
            for p in parse_explicit_constraints(",".join(explicit_toks), L):
                frozen[p - 1] = True; added_frozen.add(p)
        if nl_phrases:
            if not args.complex_cif:
                raise SystemExit("[11B] NL --additional-constraints requires --complex-cif")
            interfaces = resolve_interfaces_from_complex(
                args.complex_cif, design_chain=args.design_chain,
                phrases=nl_phrases, contact_cutoff=args.interface_contact_cutoff,
            )
            for label, info in interfaces.items():
                for p in info["positions"]:
                    if 1 <= p <= L:
                        frozen[p - 1] = True; added_frozen.add(p)
                print(f"[11B] interface {label}: {info['n_contacts']} residues", flush=True)
        if args.interfaces_json_out:
            Path(args.interfaces_json_out).write_text(json.dumps({
                "design_chain": args.design_chain,
                "interfaces": interfaces,
                "n_frozen_added": len(added_frozen),
                "contact_cutoff_A": args.interface_contact_cutoff,
            }, indent=2))

    # ---- Coupling-aware handling of conserved positions ------------------------
    #
    # A blanket freeze of every conserved position forbids exactly the move class
    # that adaptation most often uses: *paired* substitutions.  Salt bridges,
    # aromatic stacking pairs and engineered disulfides are two-residue features,
    # and a coevolving pair is by definition one where the compensating double
    # mutant is viable although either single mutant is not.  A per-position mask
    # cannot express that, and freezing the pair also forecloses strengthening an
    # existing contact (D->E on one side, K->R on the other).
    #
    # So conservation decides *how* a position may move, not whether:
    #
    #   conserved, no structural partner  -> frozen.  A lone conserved residue has
    #       nothing to compensate it, so any substitution there is unpaired by
    #       construction and the old reasoning still holds.
    #   conserved, in a WT Cbeta-Cbeta contact -> designable, and its partners are
    #       opened too (contact closure), so LigandMPNN resamples the pair jointly
    #       instead of being handed one half pinned at wild type.
    #
    # Active-site and interface freezes are hard and always win over closure.
    hard_frozen = frozen.copy()
    cons_hi = {i for i in range(L) if cons[i] >= args.conservation_freeze}
    comutate_with: dict[int, set[int]] = {}
    contact_g = {}
    if args.coupling_mask != "off":
        _pred = wt_struct_prov["source"] == "predicted"
        _pdb_txt = Path(args.wt_pdb).read_text()
        _pl = _plddt_from_pdb(_pdb_txt, args.design_chain) if _pred else None
        contact_g = contact_graph_from_pdb(
            _pdb_txt, chain=args.design_chain, cutoff=args.contact_cutoff,
            plddt=_pl, plddt_min=args.contact_plddt_min if _pred else None)
        if not contact_g:
            # No coupling information -> blanket conservation freeze, which is
            # the conservative direction, but say so rather than degrade quietly.
            print("[11B] WARNING: empty WT contact graph; coupling mask inert, "
                  "falling back to blanket conservation freeze", flush=True)
        for i in cons_hi:
            partners = {p - 1 for p in contact_g.get(i + 1, ()) if 0 <= p - 1 < L}
            if partners:
                comutate_with[i] = partners
    for i in cons_hi:
        if i not in comutate_with:
            frozen[i] = True
    for partners in comutate_with.values():
        for p in partners:
            if not hard_frozen[p]:
                frozen[p] = False
    print(f"[11B] coupling_mask={args.coupling_mask}: {len(cons_hi)} conserved "
          f"positions, {len(comutate_with)} of them coupled (left designable), "
          f"{len(cons_hi) - len(comutate_with)} frozen as uncoupled", flush=True)
    fixed_positions_1b = sorted({int(i) + 1 for i in np.where(frozen)[0]})
    print(f"[11B] fixed_residues for MPNN: {len(fixed_positions_1b)}/{L} positions "
          f"(active-site + interfaces + conservation>={args.conservation_freeze})",
          flush=True)

    # ---- Stage B: PLM (adapter + head) for oracle ----
    from transformers import AutoTokenizer, EsmForMaskedLM
    from eptrans.modeling.model import ESM2_CHECKPOINTS, DEFAULT_BACKBONE
    ckpt = ESM2_CHECKPOINTS.get(args.backbone_size, ESM2_CHECKPOINTS[DEFAULT_BACKBONE])
    tok = AutoTokenizer.from_pretrained(ckpt)
    print(f"[11B] loading {ckpt} + adapter ...", flush=True)
    base = EsmForMaskedLM.from_pretrained(ckpt)
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, args.mlm_adapter, adapter_name="mlm")
    model.set_adapter("mlm")
    model.eval().to(args.device)
    hidden = base.config.hidden_size
    head = torch.nn.Sequential(torch.nn.Linear(hidden, 512), torch.nn.GELU(),
                               torch.nn.Dropout(0.1), torch.nn.Linear(512, 1))
    _ckpt = torch.load(args.head, map_location=args.device)
    _sd = _ckpt["state_dict"] if isinstance(_ckpt, dict) and "state_dict" in _ckpt else _ckpt
    head.load_state_dict(_sd)
    head.eval().to(args.device)

    score_classifier, seq_loglik, coupling_preservation, wt_pairs = build_oracle(
        model, tok, head, args.device, seq
    )

    wt_score = score_classifier(seq)
    wt_ll = seq_loglik(seq)
    print(f"[11B] WT {args.phenotype} classifier={wt_score:.4f} seqLL={wt_ll:.4f} "
          f"wt_contact_pairs={len(wt_pairs)}", flush=True)

    # ---- Stage C: bias_AA + refold worker ----
    bias_vec = load_bias_AA(args.bias_aa_json, args.phenotype)

    protected_sets: dict[str, list[int]] = {}
    protected_caps: dict[str, float] = {}
    if active_1b:
        protected_sets["active_site"] = active_1b
        protected_caps["active_site"] = args.core_rmsd_cap
    for label, info in interfaces.items():
        protected_sets[label] = info["positions"]
        protected_caps[label] = args.interface_rmsd_cap

    # Adaptive, clamped caps.  The calibration table maps mutation count -> the
    # upper quantile of RMSD observed for structurally *neutral* variants at that
    # count, so the gate asks "is this further than a conservative variant of the
    # same size would land?" rather than comparing against a round number.  The
    # ceiling clamps it so a wide null can never open the gate indefinitely.
    _cap_table = (json.loads(Path(args.rmsd_cap_table).read_text()).get("caps", {})
                  if args.rmsd_cap_table else {})
    _ceiling = {lbl: (args.core_rmsd_ceiling if lbl == "active_site"
                      else args.interface_rmsd_ceiling) for lbl in protected_sets}

    def cap_for(label: str, n_mut: int) -> float:
        base = protected_caps[label]
        tb = _cap_table.get("active_site" if label == "active_site" else "interface")
        if tb:
            ks = sorted(int(k) for k in tb)
            kk = max([x for x in ks if x <= n_mut], default=ks[0])
            base = float(tb[str(kk)])
        return min(base, _ceiling[label])

    refolder = RefoldClient(args.refold_workdir, args.wt_pdb)
    if not refolder.wait_ready():
        raise SystemExit(
            "[11B] refold worker never became READY; Pipeline B requires in-loop folds"
        )
    print(f"[11B] refold worker ready; {len(protected_sets)} protected sets: "
          + ", ".join(f"{k}({len(v)}) cap={protected_caps[k]}A"
                      for k, v in protected_sets.items()), flush=True)

    # ---- Stage D: MPNN-in-the-loop ----
    temps = [float(x) for x in args.temperatures.split(",")]
    if not temps: temps = [0.05, 0.10, 0.20]
    def temp_for_round(r: int) -> float:
        return temps[r % len(temps)]

    def n_mut(s: str) -> int:
        return sum(1 for a, b in zip(seq, s) if a != b)

    # Round-local MPNN workdir; MPNN writes seqs/*.fa there
    mpnn_root = workdir / "_mpnn"
    mpnn_root.mkdir(parents=True, exist_ok=True)

    def build_survivor_bias(
        survivors: list[dict], K: int, min_agree: int, strength: float
    ) -> tuple[dict, dict]:
        """Return (per_residue_bias_dict, diagnostic).

        per_residue_bias_dict: {"<chain><resid>": {AA: strength}} for positions
        where >= min_agree of the top-K survivors agree on a NON-WT AA (WT-only
        columns are left unbiased since MPNN already sees WT via backbone).
        """
        if not survivors or K <= 0:
            return {}, {"topk_used": 0, "biased_positions": 0}
        top = sorted(survivors, key=lambda d: d["score_product"], reverse=True)[:K]
        bias_pr: dict = {}
        biased = []
        for i in range(L):
            wt_aa = seq[i]
            # tally non-WT AAs at this column across top-K
            tally: dict[str, int] = {}
            for s in top:
                aa = s["sequence"][i]
                if aa != wt_aa:
                    tally[aa] = tally.get(aa, 0) + 1
            if not tally:
                continue
            best_aa, best_n = max(tally.items(), key=lambda kv: kv[1])
            if best_n >= min_agree:
                key = f"{args.design_chain}{i+1}"  # 1-based
                bias_pr[key] = {best_aa: float(strength)}
                biased.append({"pos": i + 1, "wt": wt_aa, "aa": best_aa,
                               "agree": best_n, "of": len(top)})
        return bias_pr, {"topk_used": len(top),
                         "biased_positions": len(biased),
                         "examples": biased[:15]}

    all_survivors: list[dict] = []
    round_stats: list[dict] = []
    for r in range(args.n_rounds):
        T = temp_for_round(r)
        round_dir = mpnn_root / f"r{r:02d}"

        # Build survivor-conditioned per-residue bias for rounds past warmup.
        per_res_bias: dict = {}
        bias_diag: dict = {"topk_used": 0, "biased_positions": 0}
        if (args.survivor_topk > 0
                and r >= args.survivor_warmup_rounds
                and len(all_survivors) > 0):
            per_res_bias, bias_diag = build_survivor_bias(
                all_survivors,
                K=args.survivor_topk,
                min_agree=args.survivor_min_agreement,
                strength=args.survivor_bias_strength,
            )
            print(f"[11B] round {r} survivor-bias: topK={bias_diag['topk_used']} "
                  f"biased_positions={bias_diag['biased_positions']}", flush=True)

        recs, _tail = mpnn_propose(
            mpnn_repo=args.mpnn_repo, mpnn_env=args.mpnn_env,
            wt_pdb=args.wt_pdb, design_chain=args.design_chain,
            fixed_positions_1b=fixed_positions_1b, bias_AA_vec=bias_vec,
            temperature=T, n_designs=args.batch_per_round,
            workdir=round_dir, seed=args.seed + r,
            model_type=args.mpnn_model,
            bias_AA_per_residue=(per_res_bias or None),
        )
        r_survivors, r_rejected = [], []
        for k, rec in enumerate(recs):
            s = rec["sequence"].upper()
            if len(s) != L:
                r_rejected.append({"round": r, "k": k, "reason": "length_mismatch",
                                   "len": len(s)})
                continue
            # ---- Tier 1a: conservation guard (also enforced by --fixed_residues, but
            # keep as double-check in case MPNN respects fixed_residues loosely).
            muts = {i for i in range(L) if s[i] != seq[i]}
            bad_cons = [i + 1 for i in muts if i in cons_hi and i not in comutate_with]
            if bad_cons:
                r_rejected.append({"round": r, "k": k, "reason": "conservation",
                                   "positions": bad_cons[:10]})
                continue
            if args.coupling_mask == "comutate":
                # A conserved residue in a contact may move, but not alone: at least
                # one of its structural partners must move with it.  This is the
                # half-mutation guard -- breaking one side of a salt bridge and
                # leaving the other is the failure mode the blanket freeze used to
                # prevent by forbidding the move entirely.
                half = [i + 1 for i in muts
                        if i in comutate_with and not (comutate_with[i] & muts)]
                if half:
                    r_rejected.append({"round": r, "k": k, "reason": "half_mutation",
                                       "positions": half[:10]})
                    continue
            # ---- Tier 1b: refold + per-set RMSDs (caps adapt to mutation count)
            caps_now = {lbl: cap_for(lbl, len(muts)) for lbl in protected_sets}
            rmsds = refolder.refold_rmsd_multi(s, protected_sets)
            fails = [(lbl, r_v, caps_now[lbl])
                     for lbl, r_v in rmsds.items()
                     if r_v is None or r_v > caps_now[lbl]]
            if fails:
                r_rejected.append({"round": r, "k": k, "reason": "rmsd",
                                   "fails": [(lbl, (None if v is None else round(v,3)),
                                              cap) for lbl, v, cap in fails]})
                continue
            # ---- Tier 2: soft scores (all normalized-ish [0,1])
            p_clf = score_classifier(s)
            ll = seq_loglik(s)
            ll_ratio = float(np.exp(ll - wt_ll))
            p_cpl = coupling_preservation(s)
            product = p_clf * ll_ratio * p_cpl
            r_survivors.append(dict(
                round=r, k=k, temperature=T, sequence=s,
                mpnn_overall_confidence=rec.get("overall_confidence"),
                mpnn_ligand_confidence=rec.get("ligand_confidence"),
                subscores={
                    "p_classifier": round(p_clf, 4),
                    "seqLL_design": round(ll, 4),
                    "seqLL_wt": round(wt_ll, 4),
                    "seqLL_ratio": round(ll_ratio, 4),
                    "p_coupling": round(p_cpl, 4),
                },
                score_product=round(product, 6),
                rmsds={k: (None if v is None else round(v, 3)) for k, v in rmsds.items()},
                n_mutations=n_mut(s),
            ))
        all_survivors.extend(r_survivors)
        round_stats.append(dict(
            round=r, temperature=T,
            proposed=len(recs), survivors=len(r_survivors),
            rejected=len(r_rejected), reject_breakdown=r_rejected,
            survivor_bias=bias_diag,
        ))
        print(f"[11B] round {r} T={T:.3f}: {len(r_survivors)}/{len(recs)} survived "
              f"(cum {len(all_survivors)})", flush=True)

    # ---- Stage E: rank + pick top-N
    all_survivors.sort(key=lambda d: d["score_product"], reverse=True)
    top = all_survivors[: args.n_designs]

    # Persist design PDBs so 11d_assemble.structure_file="{did}.pdb" resolves
    # in the webapp. The in-loop gate discards each fold's PDB after computing
    # RMSDs; re-fold each pick once more (ESMFold is deterministic for a given
    # sequence). Output dir mirrors Pipeline A: <jobdir>/structures/.
    struct_dir = Path(args.out).resolve().parent / "structures"
    struct_dir.mkdir(parents=True, exist_ok=True)

    designs = []
    for lvl, d in enumerate(top):
        muts = [dict(pos=i + 1, wt=a, mut=b)
                for i, (a, b) in enumerate(zip(seq, d["sequence"])) if a != b]
        did = f"{args.phenotype[:4]}_B{lvl+1}"
        # Fold once more and write structures/<did>.pdb.
        pdb_txt = refolder.refold_pdb(d["sequence"])
        if pdb_txt:
            (struct_dir / f"{did}.pdb").write_text(pdb_txt)
            print(f"[11B] wrote {struct_dir/f'{did}.pdb'} ({len(pdb_txt)} bytes)",
                  flush=True)
        else:
            print(f"[11B] WARN: refold_pdb({did}) returned None; "
                  f"structure not persisted", flush=True)
        designs.append(dict(
            design_id=did, sequence=d["sequence"], level=lvl,
            round=d["round"], temperature=d["temperature"],
            classifier_score=d["subscores"]["p_classifier"],
            biophysical_score=round(biophysical_score(d["sequence"], args.phenotype), 4),
            n_mutations=d["n_mutations"], mutations=muts,
            score_product=d["score_product"],
            subscores=d["subscores"],
            rmsds=d["rmsds"],
            mpnn_overall_confidence=d.get("mpnn_overall_confidence"),
            mpnn_ligand_confidence=d.get("mpnn_ligand_confidence"),
        ))
        print(f"[11B] pick {did}: prod={d['score_product']:.4f} "
              f"clf={d['subscores']['p_classifier']:.3f} "
              f"llR={d['subscores']['seqLL_ratio']:.3f} "
              f"cpl={d['subscores']['p_coupling']:.3f} muts={len(muts)}",
              flush=True)

    out = dict(
        pipeline="B",
        wt_sequence=seq, phenotype=args.phenotype,
        wt_classifier_score=round(wt_score, 4),
        wt_seq_loglik=round(wt_ll, 4),
        conservation=[round(float(c), 3) for c in cons],
        active_site=active_1b, active_site_assigned=assigned,
        active_site_sources=src_counts, active_site_transfer=transfer_meta,
        n_msa_hits=int(n_hits),
        interfaces=interfaces,
        interface_rmsd_cap=float(args.interface_rmsd_cap),
        core_rmsd_cap=float(args.core_rmsd_cap),
        conservation_freeze=float(args.conservation_freeze),
        mpnn_model=args.mpnn_model,
        mpnn_temperatures=temps,
        n_rounds=args.n_rounds,
        batch_per_round=args.batch_per_round,
        n_designs=args.n_designs,
        n_survivors_total=len(all_survivors),
        round_stats=round_stats,
        bias_aa=bias_vec,
        wt_structure_provenance=wt_struct_prov,
        coupling_mask=args.coupling_mask,
        coupling_contact_graph={
            "n_residues_with_contacts": len(contact_g),
            "contact_cutoff_A": float(args.contact_cutoff),
            "source": wt_struct_prov["source"],
        },
        wt_contact_pairs=len(wt_pairs),
        designs=designs,
    )
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[11B] wrote {args.out} ({len(designs)} designs, "
          f"{len(all_survivors)} survivors from {sum(r['proposed'] for r in round_stats)} proposals)",
          flush=True)


if __name__ == "__main__":
    main()
