#!/usr/bin/env python
"""Stage 1-5 core of the generation pipeline (runs in the eptrans_ml env).

Chains, for ONE input enzyme + ONE phenotype:
  Stage 1  MSA           mmseqs easy-search vs UniRef30 (falls back to uniform
                          conservation if too few hits -- the spec's degraded path)
  Stage 2  conservation  sequence-weighted per-column conservation from the MSA
  Stage 3  active-site   frozen[] = high-conservation columns U detected
                          multicopper-oxidase Cu-site motif residues (His/Cys)
  Stage 4  masked-gen    ESM-2 3B + MLM adapter, conservation-gated masking units
                          (masking.py), Gibbs passes. NOTE: this first end-to-end
                          version uses SINGLETON mask units (build_mask_units with
                          no contact_pairs) — the contact-pair COUPLED masking of the
                          production spec (decode coupled residues jointly) is the
                          documented next enhancement: it needs the WT ESM-2 contact
                          map fed as contact_pairs, computed once per job.
  Stage 5  scoring       per-phenotype cached head (directional signal) +
                          non-learnable biophysical proxy (anti-gaming gate)

Writes candidates.json: {wt_sequence, phenotype, conservation, active_site (1-based),
active_site_assigned, designs:[{design_id, sequence, level, classifier_score,
biophysical_score, n_mutations, mutations:[...], gibbs_trace:[...]}]}. The MPNN gate
(Stage 6a) and ESMFold refold + catalytic-RMSD (Stage 6b/c) run in their own envs
downstream and merge into results.json.

For a first end-to-end test the MPNN gate runs as a FINAL filter (not the periodic
in-loop audit of the production spec) -- the in-loop audit crosses conda envs every
K steps and is a downstream optimization.
"""
import sys, os, json, argparse, subprocess, tempfile, time, uuid
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---- in-loop refold client (talks to 11e_esmfold_worker via a shared queue dir) ----
def _ca_coords(pdb_text):
    out = {}
    for ln in pdb_text.splitlines():
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
            out[int(ln[22:26])] = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    return out


def _kabsch_rmsd(P, Q):
    if len(P) < 3:
        return None
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    H = Pc.T @ Qc
    V, S, Wt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Wt.T @ V.T))
    R = Wt.T @ np.diag([1, 1, d]) @ V.T
    return float(np.sqrt(((Pc @ R.T - Qc) ** 2).sum(1).mean()))


def _rmsd_over(wt_ca, dca, positions):
    idx = [p for p in positions if p in wt_ca and p in dca]
    if len(idx) < 3:
        return None
    return _kabsch_rmsd(np.array([dca[p] for p in idx]), np.array([wt_ca[p] for p in idx]))


class RefoldClient:
    """Hands a sequence to the persistent ESMFold worker and returns active-site RMSD
    to the WT structure. Returns None on timeout/failure so the loop degrades to
    no-refold rather than crashing."""

    def __init__(self, workdir, wt_pdb_path, timeout=300.0, poll=0.5):
        self.wd = Path(workdir)
        self.req = self.wd / "requests"; self.resp = self.wd / "responses"
        self.timeout = timeout; self.poll = poll
        self.wt_ca = _ca_coords(Path(wt_pdb_path).read_text())

    def wait_ready(self, timeout=1200.0):
        t0 = time.time()
        while not (self.wd / "READY").exists():
            if time.time() - t0 > timeout:
                return False
            time.sleep(self.poll)
        return True

    def refold_rmsd(self, seq, positions):
        """Fold `seq`, return CA-RMSD over `positions` (1-based) to WT, or None."""
        rid = uuid.uuid4().hex[:12]
        tmp = self.req / f"{rid}.fasta.tmp"
        tmp.write_text(seq)
        tmp.rename(self.req / f"{rid}.fasta")   # atomic submit
        out = self.resp / f"{rid}.pdb"; err = self.resp / f"{rid}.err"
        t0 = time.time()
        while True:
            if out.exists():
                pdb = out.read_text(); out.unlink(missing_ok=True)
                return _rmsd_over(self.wt_ca, _ca_coords(pdb), positions)
            if err.exists():
                err.unlink(missing_ok=True)
                return None
            if time.time() - t0 > self.timeout:
                return None
            time.sleep(self.poll)

# ---- biophysical proxies (non-learnable anti-gaming layer, modeling_design.md S2) ----
def biophysical_score(seq, phenotype):
    L = len(seq)
    if L == 0:
        return 0.0
    aa = {c: seq.count(c) / L for c in set(seq)}
    charged = sum(aa.get(c, 0) for c in "DEKR")
    acidic = aa.get("D", 0) + aa.get("E", 0)
    basic = aa.get("K", 0) + aa.get("R", 0)
    ivywrel = sum(aa.get(c, 0) for c in "IVYWREL")  # thermostability signature
    if phenotype in ("thermophile", "hyperthermophile"):
        return float(ivywrel + 0.5 * charged)          # charged surface + IVYWREL
    if phenotype == "halophile":
        return float(acidic)                            # acidic-surface bias
    if phenotype == "acidophile":
        return float(acidic - basic)                    # net negative at low pH
    if phenotype == "alkaliphile":
        return float(basic - acidic)
    return float(charged)


# ---- Stage 1-2: MSA + sequence-weighted conservation ----
def henikoff_weights(msa_rows, L):
    """Henikoff & Henikoff (1994) position-based sequence weights.

    msa_rows: list of dict {query_col(0-based) -> residue char} — one per sequence
    (query included as row 0), residues on the query coordinate frame.

    For each column c: k_c = number of DISTINCT residue types present; for a sequence
    with residue r at c, n_{c,r} = number of sequences sharing r. Its per-column weight
    is 1/(k_c * n_{c,r}) — so a residue that is rare in a well-diversified column earns
    more weight than one shared by a large (over-sampled) group. A sequence's weight is
    the mean of its per-column weights over the columns it covers (mean, not sum, so
    partial-coverage local hits are not penalised for length). Weights are renormalised
    to mean 1 so downstream 'effective counts' stay interpretable.

    This corrects phylogenetic/taxonomic OVER-SAMPLING that a raw count (or a pident
    discount) does not: an over-represented genus contributes many sequences that share
    the same residue, inflating n_{c,r} and thus each of their weights shrinks.
    """
    # per-column composition
    comp = [dict() for _ in range(L)]
    for row in msa_rows:
        for c, r in row.items():
            comp[c][r] = comp[c].get(r, 0) + 1
    kc = [len(comp[c]) for c in range(L)]
    # per-sequence weight = mean over covered columns of 1/(k_c * n_{c,r})
    weights = np.ones(len(msa_rows), dtype=float)
    for si, row in enumerate(msa_rows):
        if not row:
            weights[si] = 0.0
            continue
        acc = 0.0
        for c, r in row.items():
            n_cr = comp[c][r]
            k = kc[c] or 1
            acc += 1.0 / (k * n_cr)
        weights[si] = acc / len(row)
    m = weights[weights > 0].mean() if np.any(weights > 0) else 1.0
    if m > 0:
        weights = weights / m
    return weights


def run_msa_conservation(seq, uniref_db, workdir, min_hits=25, max_hits=2000):
    """mmseqs easy-search -> per-query-column conservation in [0,1].

    Returns (conservation[L], n_effective_hits). Builds an implied MSA on the query
    coordinate frame from the pairwise alignments (qaln/taln), computes Henikoff
    position-based sequence weights (taxonomy/over-sampling correction), then reports
    each column's weighted modal-residue fraction as conservation. Uniform (all 0)
    fallback if too few hits.
    """
    L = len(seq)
    qf = Path(workdir) / "query.fasta"
    qf.write_text(f">query\n{seq}\n")
    m8 = Path(workdir) / "hits.m8"
    tmp = Path(workdir) / "mmseqs_tmp"
    # --split-memory-limit caps the target-DB footprint (pages large DBs through RAM
    # in chunks) so a single-query search against a big DB does NOT OOM — the failure
    # mode that killed the uniprot_kb (90GB) attempt. Target should be a CLUSTERED DB
    # (UniRef50) not the full uniprot_kb: same conservation signal, ~4x smaller, and
    # non-redundant. mmseqs accepts a FASTA target directly (builds a temp DB).
    mem_cap = os.environ.get("MMSEQS_MEM_LIMIT", "80G")
    cmd = ["mmseqs", "easy-search", str(qf), uniref_db, str(m8), str(tmp),
           "--format-output", "query,target,pident,qstart,qend,qaln,taln",
           "-s", "5.7", "--max-seqs", str(max_hits), "-e", "1e-3",
           "--split-memory-limit", mem_cap,
           "--threads", str(os.cpu_count() or 8)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"[11] MSA step failed/timeout ({type(e).__name__}); uniform-conservation fallback",
              flush=True)
        return np.zeros(L, dtype=float), 0

    # parse pairwise alignments -> implied MSA rows on the query coordinate frame.
    # Row 0 is the query itself (always fully present) so every column has >=1 member.
    query_row = {i: seq[i] for i in range(L)}
    msa_rows = [query_row]
    for line in m8.read_text().splitlines()[:max_hits]:
        p = line.split("\t")
        if len(p) < 7:
            continue
        _, _, _pid, qstart, _qend, qaln, taln = p[:7]
        row = {}
        qpos = int(qstart) - 1                             # 0-based query col
        for qc, tc in zip(qaln, taln):
            if qc == "-":
                continue                                   # insertion vs query: no column
            if 0 <= qpos < L and tc != "-":
                row[qpos] = tc
            qpos += 1
        if row:
            msa_rows.append(row)

    n_hits = len(msa_rows) - 1                              # exclude the query row
    if n_hits < min_hits:
        print(f"[11] only {n_hits} MSA hits (<{min_hits}); uniform-conservation fallback",
              flush=True)
        return np.zeros(L, dtype=float), n_hits

    # Henikoff position-based sequence weights (over-sampling / taxonomy correction)
    weights = henikoff_weights(msa_rows, L)

    # weighted modal-residue fraction per column == conservation in [0,1]
    cons = np.zeros(L, dtype=float)
    for i in range(L):
        wsum = 0.0
        per_res = {}
        for si, row in enumerate(msa_rows):
            r = row.get(i)
            if r is None:
                continue
            w = weights[si]
            per_res[r] = per_res.get(r, 0.0) + w
            wsum += w
        cons[i] = (max(per_res.values()) / wsum) if wsum > 0 else 0.0
    print(f"[11] MSA conservation from {n_hits} hits (Henikoff-weighted, "
          f"eff_seqs={weights.sum():.1f}); mean cons={cons.mean():.3f}", flush=True)
    return cons, n_hits


# ---- Stage 3: active-site (multicopper-oxidase Cu-site motifs + high conservation) ----
def detect_active_site(seq, conservation, freeze_thresh=0.90, transferred=None):
    """frozen[] boolean + 1-based active-site list, from three UNIONED sources:
      1. transferred[] -- catalytic/functional-site positions carried over from
         M-CSA / Swiss-Prot via foldseek+mmseqs homology (Stage 3, 11a_annotate.py).
         This is the PRIMARY, general source when annotation hits exist.
      2. very-high-conservation columns (>= freeze_thresh) from the MSA.
      3. multicopper-oxidase Cu-binding motif residues (His/Cys/Met in HxHG/HCH...) --
         a laccase-specific backstop so a copper enzyme is never left unprotected
         even when annotation/conservation are thin.
    Reports which sources fired so the UI/notebook can distinguish a real transfer
    from the heuristic fallback."""
    import re
    L = len(seq)
    frozen = np.zeros(L, dtype=bool)
    active = set()
    src_counts = {"transferred": 0, "conservation": 0, "motif": 0}
    # 1. homology-transferred positions (1-based -> 0-based)
    for p in (transferred or []):
        if 1 <= p <= L:
            frozen[p - 1] = True
            active.add(p - 1)
            src_counts["transferred"] += 1
    # 2. high-conservation columns
    for i in range(L):
        if conservation[i] >= freeze_thresh:
            if not frozen[i]:
                src_counts["conservation"] += 1
            frozen[i] = True
            active.add(i)
    # 3. multicopper-oxidase copper ligands (His/Cys/Met in local His/Cys clusters)
    for m in re.finditer(r"H.{0,3}H|HCH|H.H.{2,4}H|C.{2,4}H", seq):
        for j in range(m.start(), m.end()):
            if seq[j] in "HCM":
                if not frozen[j]:
                    src_counts["motif"] += 1
                frozen[j] = True
                active.add(j)
    active_1based = sorted(p + 1 for p in active)
    assigned = len(active_1based) > 0
    return frozen, active_1based, assigned, src_counts


# ---- Stage 4-5: Gibbs masked-gen + scoring ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="input enzyme sequence (mature chain)")
    ap.add_argument("--phenotype", required=True)
    ap.add_argument("--out", required=True, help="output candidates.json path")
    ap.add_argument("--mlm-adapter", required=True)
    ap.add_argument("--head", required=True, help="cached head_best.pt")
    ap.add_argument("--uniref-db", required=True)
    ap.add_argument("--transfer-json", default="",
                    help="active_site_transfer.json from 11a_annotate.py (optional)")
    ap.add_argument("--backbone-size", default="3B")
    ap.add_argument("--n-designs", type=int, default=3)
    ap.add_argument("--gibbs-iters", type=int, default=24)
    ap.add_argument("--max-mut-frac", type=float, default=0.25,
                    help="hard mutation budget as a fraction of length; a proposal that "
                         "would push the total mutation count past this is rejected even "
                         "if it improves the classifier score (anti-fold-collapse)")
    ap.add_argument("--refold-workdir", default="",
                    help="if set, enable in-loop periodic ESMFold refolds via the worker "
                         "queue at this dir (11e_esmfold_worker.py must be running against it)")
    ap.add_argument("--wt-pdb", default="", help="cached WT structure for refold RMSD")
    ap.add_argument("--refold-every", type=int, default=4,
                    help="refold + RMSD-check every N Gibbs passes (0 disables)")
    ap.add_argument("--refold-rmsd-cap", type=float, default=2.0,
                    help="active-site CA-RMSD (A) above which a checkpoint is rejected -> "
                         "roll back to last passing sequence and reduce mask_rate")
    ap.add_argument("--refold-backoff", type=float, default=0.5,
                    help="multiply mask_rate by this after a rollback (more conservative)")
    ap.add_argument("--refold-max-fails", type=int, default=3,
                    help="stop this design after this many consecutive failed refolds")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--seed", type=int, default=1466)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, EsmForMaskedLM
    from eptrans.modeling.model import build_lora_backbone, ESM2_CHECKPOINTS, DEFAULT_BACKBONE
    from eptrans.modeling import masking as MK

    seq = args.seq.strip().upper()
    L = len(seq)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    Path(args.workdir).mkdir(parents=True, exist_ok=True)

    # Stages 1-3
    cons, n_hits = run_msa_conservation(seq, args.uniref_db, args.workdir)
    transferred = []
    transfer_meta = {}
    if args.transfer_json and Path(args.transfer_json).exists():
        tj = json.loads(Path(args.transfer_json).read_text())
        transferred = tj.get("transferred", [])
        transfer_meta = {k: tj.get(k) for k in ("by_source", "n_foldseek_hits", "n_mmseqs_hits")}
        print(f"[11] loaded {len(transferred)} transferred active-site positions "
              f"(foldseek_hits={tj.get('n_foldseek_hits')}, mmseqs_hits={tj.get('n_mmseqs_hits')})",
              flush=True)
    frozen, active_1b, assigned, src_counts = detect_active_site(seq, cons, transferred=transferred)
    print(f"[11] active-site: {len(active_1b)} frozen residues (assigned={assigned}); "
          f"sources {src_counts}", flush=True)

    # Model: EsmForMaskedLM (3B) + trained MLM LoRA adapter, for BOTH proposer logits
    # and encoder-pooled scoring (one model load; .esm gives encoder hidden states).
    ckpt = ESM2_CHECKPOINTS.get(args.backbone_size, ESM2_CHECKPOINTS[DEFAULT_BACKBONE])
    tok = AutoTokenizer.from_pretrained(ckpt)
    print(f"[11] loading {ckpt} + adapter ...", flush=True)
    base = EsmForMaskedLM.from_pretrained(ckpt)
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, args.mlm_adapter, adapter_name="mlm")
    model.set_adapter("mlm")
    model.eval().to(args.device)
    hidden = base.config.hidden_size

    # cached head (matches 10_train_cached_probe: Linear(hidden,512)->GELU->Dropout->Linear(512,1))
    head = torch.nn.Sequential(torch.nn.Linear(hidden, 512), torch.nn.GELU(),
                               torch.nn.Dropout(0.1), torch.nn.Linear(512, 1))
    head.load_state_dict(torch.load(args.head, map_location=args.device))
    head.eval().to(args.device)

    mask_id = tok.mask_token_id
    special_ids = set(tok.all_special_ids)

    def encode(s):
        return tok(s, return_tensors="pt", add_special_tokens=True).to(args.device)

    @torch.no_grad()
    def score(s):
        enc = encode(s)
        out = model.base_model.model.esm(enc["input_ids"], attention_mask=enc["attention_mask"])
        h = out.last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
        logit = head(pooled).squeeze(-1)
        return torch.sigmoid(logit).item()

    @torch.no_grad()
    def mlm_fill(s, mask_pos0):
        """Mask the given 0-based residue positions, return refilled sequence
        (sampled from the adapted-MLM logits at those positions)."""
        enc = encode(s)
        ids = enc["input_ids"].clone()
        # token index = residue index + 1 (CLS at 0)
        for p in mask_pos0:
            ids[0, p + 1] = mask_id
        out = model(ids, attention_mask=enc["attention_mask"])
        logits = out.logits[0]
        s_list = list(s)
        for p in mask_pos0:
            probs = torch.softmax(logits[p + 1], dim=-1)
            # restrict to canonical AAs
            tid = int(torch.multinomial(probs, 1))
            aa = tok.convert_ids_to_tokens(tid)
            if len(aa) == 1 and aa in "ACDEFGHIKLMNPQRSTVWY":
                s_list[p] = aa
        return "".join(s_list)

    # special mask (CLS/EOS handled by residue indexing; only frozen matters here)
    special = np.zeros(L, dtype=bool)

    # per-design aggressiveness schedule (inline: conservative->aggressive)
    def schedule(level, n):
        # level in [0, n-1]; interpolate mask_rate/gamma/target_mut_frac
        t = 0.0 if n == 1 else level / (n - 1)
        return dict(mask_rate=0.05 + 0.15 * t, gamma=2.5 - 1.5 * t,
                    target_mut_frac=0.05 + 0.25 * t)

    wt_score = score(seq)
    print(f"[11] WT {args.phenotype} score={wt_score:.4f}", flush=True)

    def n_mut(s):
        return sum(1 for a, b in zip(seq, s) if a != b)

    # hard mutation budget (residues), shared cap across levels; the per-level schedule
    # still ramps how AGGRESSIVELY each design approaches it via mask_rate/target_mut_frac.
    mut_budget = max(1, int(round(args.max_mut_frac * L)))

    # optional in-loop refold client (structural rollback). RMSD is measured over the
    # active-site residues (same set the final Stage-6b gate uses).
    refolder = None
    if args.refold_workdir and args.wt_pdb and args.refold_every > 0 and active_1b:
        refolder = RefoldClient(args.refold_workdir, args.wt_pdb)
        if refolder.wait_ready():
            print(f"[11] refold worker ready; checking every {args.refold_every} passes "
                  f"(cap {args.refold_rmsd_cap} A over {len(active_1b)} AS residues)", flush=True)
        else:
            print("[11] WARN refold worker never became READY -> disabling in-loop refold", flush=True)
            refolder = None

    designs = []
    for lvl in range(args.n_designs):
        sch = schedule(lvl, args.n_designs)
        cur = seq
        cur_score = wt_score
        trace = [(0, cur_score)]
        mask_rate = sch["mask_rate"]        # mutable: shrinks on rollback
        last_good = seq                      # last refold-passing sequence
        last_good_score = wt_score
        consec_fails = 0
        n_refolds = n_rollbacks = 0
        for it in range(args.gibbs_iters):
            units = MK.build_mask_units(L, special=special, frozen=frozen)  # singleton units
            mask = MK.sample_mask_units(cons, units, mask_rate=mask_rate,
                                        gamma=sch["gamma"], rng=rng)
            mask_pos0 = np.where(mask)[0].tolist()
            if not mask_pos0:
                break
            prop = mlm_fill(cur, mask_pos0)
            ps = score(prop)
            # accept if the phenotype score improves AND the mutation budget is respected;
            # a score-improving move that overruns the budget is rejected (fold-collapse
            # from over-mutation is the failure mode the RMSD gate later catches — this
            # keeps designs inside the budget so they rarely reach the gate at all).
            if ps >= cur_score and n_mut(prop) <= mut_budget:
                cur, cur_score = prop, ps
            trace.append((it + 1, cur_score))
            # ---- periodic structural checkpoint: refold + active-site RMSD ----
            if refolder is not None and (it + 1) % args.refold_every == 0 and cur != last_good:
                rmsd = refolder.refold_rmsd(cur, active_1b)
                n_refolds += 1
                if rmsd is not None and rmsd <= args.refold_rmsd_cap:
                    last_good, last_good_score, consec_fails = cur, cur_score, 0
                    print(f"[11]   {args.phenotype[:4]}_{lvl+1} pass {it+1}: refold OK "
                          f"rmsd={rmsd:.2f} A (checkpoint)", flush=True)
                else:
                    # roll back to the last passing sequence, take smaller steps next
                    consec_fails += 1; n_rollbacks += 1
                    cur, cur_score = last_good, last_good_score
                    mask_rate = max(0.01, mask_rate * args.refold_backoff)
                    print(f"[11]   {args.phenotype[:4]}_{lvl+1} pass {it+1}: refold FAIL "
                          f"rmsd={rmsd} > {args.refold_rmsd_cap} -> rollback, "
                          f"mask_rate->{mask_rate:.3f} (fail {consec_fails})", flush=True)
                    if consec_fails >= args.refold_max_fails:
                        print(f"[11]   {args.phenotype[:4]}_{lvl+1}: {consec_fails} consecutive "
                              f"refold fails -> stop, keep last_good", flush=True)
                        break
        # if refolding was on, return a structurally-validated sequence. `cur` may hold
        # improvements made since the last periodic checkpoint (or, if gibbs_iters <
        # refold_every, may never have been checkpointed at all) -- do ONE final refold
        # so we don't silently discard good work OR collapse to WT. Keep `cur` iff it
        # passes; otherwise fall back to last_good (WT only if nothing ever passed).
        if refolder is not None and cur != last_good:
            final_rmsd = refolder.refold_rmsd(cur, active_1b)
            n_refolds += 1
            if final_rmsd is not None and final_rmsd <= args.refold_rmsd_cap:
                last_good, last_good_score = cur, cur_score
                print(f"[11]   {args.phenotype[:4]}_{lvl+1} final refold OK "
                      f"rmsd={final_rmsd:.2f} A", flush=True)
            else:
                print(f"[11]   {args.phenotype[:4]}_{lvl+1} final refold FAIL "
                      f"rmsd={final_rmsd} -> keep last_good", flush=True)
        if refolder is not None:
            cur, cur_score = last_good, last_good_score
        muts = [dict(pos=i + 1, wt=a, mut=b) for i, (a, b) in enumerate(zip(seq, cur)) if a != b]
        did = f"{args.phenotype[:4]}_{lvl+1}"
        designs.append(dict(design_id=did, sequence=cur, level=lvl,
                            classifier_score=round(cur_score, 4),
                            biophysical_score=round(biophysical_score(cur, args.phenotype), 4),
                            n_mutations=len(muts), mutations=muts,
                            n_refolds=n_refolds, n_rollbacks=n_rollbacks,
                            gibbs_trace=[[int(a), round(float(b), 4)] for a, b in trace]))
        print(f"[11] design {did}: score {wt_score:.3f}->{cur_score:.3f}, {len(muts)} muts, "
              f"{n_refolds} refolds / {n_rollbacks} rollbacks", flush=True)

    out = dict(wt_sequence=seq, phenotype=args.phenotype,
               wt_classifier_score=round(wt_score, 4),
               conservation=[round(float(c), 3) for c in cons],
               active_site=active_1b, active_site_assigned=assigned,
               active_site_sources=src_counts, active_site_transfer=transfer_meta,
               n_msa_hits=int(n_hits), designs=designs)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[11] wrote {args.out} ({len(designs)} designs)", flush=True)


if __name__ == "__main__":
    main()
