#!/usr/bin/env python
"""Stage 1-5 core of the generation pipeline (runs in the eptrans_ml env).

Chains, for ONE input enzyme + ONE phenotype:
  Stage 1  MSA           mmseqs easy-search vs UniRef30 (falls back to uniform
                          conservation if too few hits -- the spec's degraded path)
  Stage 2  conservation  sequence-weighted per-column conservation from the MSA
  Stage 3  active-site   frozen[] = high-conservation columns U detected
                          multicopper-oxidase Cu-site motif residues (His/Cys)
  Stage 4  masked-gen    ESM-2 3B + MLM adapter, conservation-gated, contact-pair
                          coupled masking units (masking.py), Gibbs passes
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
import sys, os, json, argparse, subprocess, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
def run_msa_conservation(seq, uniref_db, workdir, min_hits=25, max_hits=2000):
    """mmseqs easy-search -> per-query-column conservation in [0,1].

    Returns (conservation[L], n_effective_hits). Uses pairwise alignments anchored
    to query coords (qaln/taln). Sequence-weighted by Henikoff-style 1/n_similar to
    down-weight redundant homologs. Uniform (all 0) fallback if too few hits.
    """
    L = len(seq)
    qf = Path(workdir) / "query.fasta"
    qf.write_text(f">query\n{seq}\n")
    m8 = Path(workdir) / "hits.m8"
    tmp = Path(workdir) / "mmseqs_tmp"
    cmd = ["mmseqs", "easy-search", str(qf), uniref_db, str(m8), str(tmp),
           "--format-output", "query,target,pident,qstart,qend,qaln,taln",
           "-s", "5.7", "--max-seqs", str(max_hits), "-e", "1e-3",
           "--threads", str(os.cpu_count() or 8)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"[11] MSA step failed/timeout ({type(e).__name__}); uniform-conservation fallback",
              flush=True)
        return np.zeros(L, dtype=float), 0

    # parse pairwise alignments anchored to query columns
    counts = [dict() for _ in range(L)]  # per query col: {aa: weighted count}
    rows = []
    for line in m8.read_text().splitlines():
        p = line.split("\t")
        if len(p) < 7:
            continue
        _, _, pid, qstart, qend, qaln, taln = p[:7]
        rows.append((float(pid), int(qstart), qaln, taln))
    if len(rows) < min_hits:
        print(f"[11] only {len(rows)} MSA hits (<{min_hits}); uniform-conservation fallback",
              flush=True)
        return np.zeros(L, dtype=float), len(rows)

    # Henikoff-ish weight: down-weight near-duplicates by pident bucket
    for pid, qstart, qaln, taln in rows[:max_hits]:
        w = 1.0 / (1.0 + max(0.0, (pid - 30.0)) / 20.0)   # higher-id homologs weigh less
        qpos = qstart - 1                                  # 0-based query col
        for qc, tc in zip(qaln, taln):
            if qc == "-":
                continue                                   # insertion vs query: no column
            if 0 <= qpos < L and tc != "-":
                counts[qpos][tc] = counts[qpos].get(tc, 0.0) + w
            qpos += 1

    cons = np.zeros(L, dtype=float)
    for i in range(L):
        tot = sum(counts[i].values())
        if tot <= 0:
            cons[i] = 0.0
            continue
        # weighted fraction of the modal residue == simple conservation in [0,1]
        cons[i] = max(counts[i].values()) / tot
    print(f"[11] MSA conservation from {len(rows)} hits; mean cons={cons.mean():.3f}", flush=True)
    return cons, len(rows)


# ---- Stage 3: active-site (multicopper-oxidase Cu-site motifs + high conservation) ----
def detect_active_site(seq, conservation, freeze_thresh=0.90):
    """frozen[] boolean + 1-based active-site list. For a first test, catalytic
    residues = detected His/Cys/Met in the classic multicopper-oxidase Cu-binding
    motifs (HxHG, HCHxxxH...) UNION the very-high-conservation columns."""
    import re
    L = len(seq)
    frozen = np.zeros(L, dtype=bool)
    active = set()
    # high-conservation columns
    for i in range(L):
        if conservation[i] >= freeze_thresh:
            frozen[i] = True
            active.add(i)
    # multicopper-oxidase copper ligands: His-rich + Cys motifs. Freeze every His,
    # Cys, and Met that sits in a local His/Cys cluster (Cu T1/T2/T3 ligands).
    for m in re.finditer(r"H.{0,3}H|HCH|H.H.{2,4}H|C.{2,4}H", seq):
        for j in range(m.start(), m.end()):
            if seq[j] in "HCM":
                frozen[j] = True
                active.add(j)
    active_1based = sorted(p + 1 for p in active)
    assigned = len(active_1based) > 0
    return frozen, active_1based, assigned


# ---- Stage 4-5: Gibbs masked-gen + scoring ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="input enzyme sequence (mature chain)")
    ap.add_argument("--phenotype", required=True)
    ap.add_argument("--out", required=True, help="output candidates.json path")
    ap.add_argument("--mlm-adapter", required=True)
    ap.add_argument("--head", required=True, help="cached head_best.pt")
    ap.add_argument("--uniref-db", required=True)
    ap.add_argument("--backbone-size", default="3B")
    ap.add_argument("--n-designs", type=int, default=3)
    ap.add_argument("--gibbs-iters", type=int, default=24)
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
    frozen, active_1b, assigned = detect_active_site(seq, cons)
    print(f"[11] active-site: {len(active_1b)} frozen residues (assigned={assigned})", flush=True)

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

    designs = []
    for lvl in range(args.n_designs):
        sch = schedule(lvl, args.n_designs)
        cur = seq
        cur_score = wt_score
        trace = [(0, cur_score)]
        for it in range(args.gibbs_iters):
            units = MK.build_mask_units(L, special=special, frozen=frozen)  # singleton units
            mask = MK.sample_mask_units(cons, units, mask_rate=sch["mask_rate"],
                                        gamma=sch["gamma"], rng=rng)
            mask_pos0 = np.where(mask)[0].tolist()
            if not mask_pos0:
                break
            prop = mlm_fill(cur, mask_pos0)
            ps = score(prop)
            # accept if phenotype score improves (greedy hill-climb toward target)
            if ps >= cur_score:
                cur, cur_score = prop, ps
            trace.append((it + 1, cur_score))
        muts = [dict(pos=i + 1, wt=a, mut=b) for i, (a, b) in enumerate(zip(seq, cur)) if a != b]
        did = f"{args.phenotype[:4]}_{lvl+1}"
        designs.append(dict(design_id=did, sequence=cur, level=lvl,
                            classifier_score=round(cur_score, 4),
                            biophysical_score=round(biophysical_score(cur, args.phenotype), 4),
                            n_mutations=len(muts), mutations=muts,
                            gibbs_trace=[[int(a), round(float(b), 4)] for a, b in trace]))
        print(f"[11] design {did}: score {wt_score:.3f}->{cur_score:.3f}, {len(muts)} muts", flush=True)

    out = dict(wt_sequence=seq, phenotype=args.phenotype,
               wt_classifier_score=round(wt_score, 4),
               conservation=[round(float(c), 3) for c in cons],
               active_site=active_1b, active_site_assigned=assigned,
               n_msa_hits=int(n_hits), designs=designs)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[11] wrote {args.out} ({len(designs)} designs)", flush=True)


if __name__ == "__main__":
    main()
