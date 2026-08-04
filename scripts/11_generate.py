#!/usr/bin/env python
"""Stage 1-5 core of the generation pipeline (runs in the eptrans_ml env).

Chains, for ONE input enzyme + ONE phenotype:
  Stage 1  MSA           mmseqs easy-search vs UniRef30 (falls back to uniform
                          conservation if too few hits -- the spec's degraded path)
  Stage 2  conservation  sequence-weighted per-column conservation from the MSA
  Stage 3  active-site   frozen[] = high-conservation columns U detected
                          multicopper-oxidase Cu-site motif residues (His/Cys)
  Stage 4  masked-gen    ESM-2 3B + MLM adapter, conservation-gated masking units
                          (masking.py), Metropolis-annealed mask-and-fill passes.
                          Masking uses COUPLED units (contact pairs + spans) to
                          match the trained adapter, whose production run used
                          coupling_mode='both'; the WT ESM-2 contact map is
                          computed ONCE per job (1 extra forward pass against 144
                          at defaults, ~1.4% wall -- estimated, not benchmarked) since the
                          sequence is fixed and the RMSD gate keeps the fold, so
                          the coupling topology is invariant along a trajectory.
  Stage 5  scoring       per-phenotype cached head (directional signal) +
                          non-learnable biophysical proxy (anti-gaming gate)

Search behaviour (Stage 4). Proposals are stochastic (which units get masked, and
what the MLM fills them with); ACCEPTANCE is Metropolis with a geometric
annealing schedule --mh-t0 -> --mh-t1. A worsening proposal is accepted with
probability exp(dScore / T), so the chain can cross a score barrier instead of
being trapped at the first local optimum. --mh-t0 0 reproduces the earlier strict
hill-climbing exactly. Because the chain may end below its peak, the BEST-scoring
sequence of the trajectory is returned, not the last one visited.

Three escape/robustness mechanisms, distinguished because they serve different
objectives:
  * Metropolis acceptance      escapes SCORE local minima
  * refold rollback + backoff  enforces the FOLD constraint (RMSD over the
                               active site); reverts to last_good and shrinks
                               mask_rate, which --refold-recover then restores
                               after sustained success so one structural failure
                               does not permanently throttle exploration
  * mutation budget            hard constraint, checked before acceptance

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
    # ---- Metropolis acceptance (escape score local minima) ----
    # The prior loop accepted a proposal only if `ps >= cur_score`, i.e. strict
    # hill-climbing: the score could never decrease, so a design that reached a
    # local optimum could only drift laterally across exact ties. Metropolis
    # accepts a worsening move with probability exp(dScore / T), annealing T from
    # --mh-t0 to --mh-t1 geometrically over the passes. T=0 reproduces the old
    # monotone behaviour exactly (kept as the reproducibility escape hatch).
    ap.add_argument("--mh-t0", type=float, default=0.05,
                    help="initial Metropolis temperature (0 = old hill-climbing)")
    ap.add_argument("--mh-t1", type=float, default=0.005,
                    help="final Metropolis temperature (annealed geometrically)")
    # ---- mask_rate recovery (undo the one-way refold ratchet) ----
    # --refold-backoff only ever SHRANK mask_rate; a design failing one RMSD check
    # ran every remaining pass at half step size, so structural backtracking
    # silently made score exploration more conservative for the rest of the
    # trajectory. Recover geometrically after sustained success.
    ap.add_argument("--refold-recover", type=float, default=1.25,
                    help="multiply mask_rate by this after --refold-recover-after "
                         "consecutive passing checkpoints (1.0 = no recovery)")
    ap.add_argument("--refold-recover-after", type=int, default=2,
                    help="consecutive passing checkpoints before recovery kicks in")
    # ---- coupled (contact-pair) masking: match the TRAINING distribution ----
    # The production MLM adapter was trained with coupling_mode='both'
    # (contact_threshold 0.5, contact_min_sep 6, top_k 128 -- see labnotebook
    # "Coupling-aware masking"). Generation masked SINGLETONS only, so inference
    # masking did not match training masking. Computing the WT contact map ONCE
    # per job (the sequence is fixed; only substitutions change) closes that gap.
    ap.add_argument("--coupling-mode", default="both",
                    choices=["none", "contact", "span", "both"],
                    help="mask unit construction; 'both' matches the trained adapter")
    ap.add_argument("--contact-threshold", type=float, default=0.5)
    ap.add_argument("--contact-min-sep", type=int, default=6)
    ap.add_argument("--contact-top-k", type=int, default=128)
    ap.add_argument("--span-len", type=int, default=3)
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

    # ---- WT contact pairs + spans, computed ONCE (Stage 4 coupled masking) ----
    # Cost is one 3B contact-head forward pass for the whole job, not per pass:
    # the WT sequence is fixed and substitutions do not move the backbone enough
    # to invalidate the coupling topology (the RMSD gate enforces exactly that).
    # Marginal runtime is therefore 1 extra forward pass against
    # n_designs * gibbs_iters * (1 fill + 1 score) = 144 at defaults, i.e. ~1.4%
    # of wall. NOTE: estimated from the forward-pass count, NOT benchmarked; the
    # first real run should report the measured contact-pass seconds below.
    contact_pairs = None
    spans = None
    if args.coupling_mode in ("contact", "both"):
        t_cp = time.time()
        cm = None
        try:
            from eptrans.modeling.data import _predict_contacts
            cm = _predict_contacts(model, tok, seq)
        except Exception as e:
            print(f"[11] WARN contact head failed ({type(e).__name__}: {e})", flush=True)
        if cm is not None:
            contact_pairs = MK.contact_pairs_from_map(
                cm, threshold=args.contact_threshold,
                min_sep=args.contact_min_sep, top_k=args.contact_top_k)
            print(f"[11] contact pairs: {len(contact_pairs)} "
                  f"(thr {args.contact_threshold}, min_sep {args.contact_min_sep}, "
                  f"top_k {args.contact_top_k}) in {time.time()-t_cp:.1f}s", flush=True)
        else:
            # Do NOT silently fall back to singleton masking: that is the
            # train/inference mismatch this block exists to remove. Say so.
            print("[11] WARN no contact map -> coupled masking DISABLED, "
                  "masking reverts to singletons (does NOT match the trained "
                  "adapter's coupling_mode=both)", flush=True)
    if args.coupling_mode in ("span", "both"):
        spans = MK.make_span_units(L, span_len=args.span_len)
        print(f"[11] span units: {len(spans)} (span_len {args.span_len})", flush=True)

    # Units depend only on L/frozen/coupling, all fixed across passes -> build once
    # instead of rebuilding every iteration as the previous loop did.
    units = MK.build_mask_units(L, special=special, frozen=frozen,
                                contact_pairs=contact_pairs, spans=spans)
    _sizes = [len(u) for u in units]
    print(f"[11] mask units: {len(units)} "
          f"(coupled {sum(1 for s in _sizes if s > 1)}, singleton "
          f"{sum(1 for s in _sizes if s == 1)}, "
          f"max size {max(_sizes) if _sizes else 0})", flush=True)

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
        mask_rate0 = sch["mask_rate"]       # schedule value; the recovery ceiling
        mask_rate = mask_rate0              # mutable: shrinks on rollback, recovers on success
        last_good = seq                      # last refold-passing sequence
        last_good_score = wt_score
        consec_fails = 0
        consec_passes = 0
        n_refolds = n_rollbacks = n_recover = 0
        # Metropolis temperature, annealed geometrically t0 -> t1 over the passes.
        T = args.mh_t0
        t_decay = (1.0 if args.mh_t0 <= 0 or args.gibbs_iters <= 1
                   else (args.mh_t1 / args.mh_t0) ** (1.0 / (args.gibbs_iters - 1)))
        # best-so-far tracking: with Metropolis the chain can END below its peak,
        # so the peak must be remembered explicitly or annealing loses work the
        # old monotone loop kept by construction.
        best, best_score = cur, cur_score
        n_downhill = 0
        for it in range(args.gibbs_iters):
            mask = MK.sample_mask_units(cons, units, mask_rate=mask_rate,
                                        gamma=sch["gamma"], rng=rng)
            mask_pos0 = np.where(mask)[0].tolist()
            if not mask_pos0:
                break
            prop = mlm_fill(cur, mask_pos0)
            ps = score(prop)
            # Mutation budget is a HARD constraint, checked before acceptance: a
            # score-improving move that overruns it is still rejected, because
            # fold collapse from over-mutation is the failure mode the RMSD gate
            # catches downstream and we would rather never reach that gate.
            if n_mut(prop) <= mut_budget:
                d = ps - cur_score
                if d >= 0:
                    cur, cur_score = prop, ps
                elif T > 0.0 and rng.random() < np.exp(d / T):
                    # Metropolis: accept a WORSE proposal to cross a barrier.
                    cur, cur_score = prop, ps
                    n_downhill += 1
                if cur_score > best_score:
                    best, best_score = cur, cur_score
            trace.append((it + 1, cur_score))
            T *= t_decay
            # ---- periodic structural checkpoint: refold + active-site RMSD ----
            if refolder is not None and (it + 1) % args.refold_every == 0 and cur != last_good:
                rmsd = refolder.refold_rmsd(cur, active_1b)
                n_refolds += 1
                if rmsd is not None and rmsd <= args.refold_rmsd_cap:
                    last_good, last_good_score, consec_fails = cur, cur_score, 0
                    consec_passes += 1
                    msg = ""
                    # Recover step size after sustained structural success, capped at
                    # the level's scheduled value so recovery can never make a design
                    # more aggressive than its aggressiveness level allows.
                    if (args.refold_recover > 1.0 and mask_rate < mask_rate0
                            and consec_passes >= args.refold_recover_after):
                        new_mr = min(mask_rate0, mask_rate * args.refold_recover)
                        if new_mr > mask_rate:
                            msg = f" -> mask_rate {mask_rate:.3f}->{new_mr:.3f} (recover)"
                            mask_rate = new_mr
                            n_recover += 1
                        consec_passes = 0
                    print(f"[11]   {args.phenotype[:4]}_{lvl+1} pass {it+1}: refold OK "
                          f"rmsd={rmsd:.2f} A (checkpoint){msg}", flush=True)
                else:
                    # roll back to the last passing sequence, take smaller steps next
                    consec_fails += 1; n_rollbacks += 1
                    cur, cur_score = last_good, last_good_score
                    consec_passes = 0
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
        # Metropolis may leave the chain below its peak. Return the BEST scoring
        # sequence seen, not the last one visited -- otherwise annealing would
        # discard work the old monotone loop retained by construction. Only
        # promote if it respects the budget (it always does: best is only ever
        # assigned inside the budget-checked branch).
        if best_score > cur_score:
            print(f"[11]   {args.phenotype[:4]}_{lvl+1}: returning best-so-far "
                  f"{best_score:.4f} over chain end {cur_score:.4f} "
                  f"({n_downhill} downhill moves accepted)", flush=True)
            cur, cur_score = best, best_score

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
                            n_downhill_accepted=n_downhill, n_mask_rate_recoveries=n_recover,
                            mh_t0=args.mh_t0, mh_t1=args.mh_t1,
                            coupling_mode=args.coupling_mode,
                            n_contact_pairs=(len(contact_pairs) if contact_pairs else 0),
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
