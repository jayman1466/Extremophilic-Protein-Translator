# Generation pipeline architecture — A vs B

Head-to-head diagrams of the two generative pipelines that share Stages 0–5 and
diverge at Stage 6 (design loop). Both target IS621 (8WT6 chain A, 306 aa) and
consume the same mhk32 adapter + per-phenotype `head_best.pt` classifier.

- **Pipeline A** — `scripts/11_generate.py`: MLM-in-the-loop Gibbs+MH sampler
  with periodic refold + RMSD rollback (single hard fold gate; classifier score
  is the acceptance signal).
- **Pipeline B** — `scripts/11_generate_pipelineB.py`: LigandMPNN-in-the-loop
  proposer with per-round temperatures, per-candidate refold + RMSD hard gate,
  and a multiplicative soft score (classifier × seqLL ratio × coupling).

Companion sbatch wrappers:
`scripts/slurm/11_generate_is621.sbatch` (A) and
`scripts/slurm/11_generate_is621_pipelineB.sbatch` (B), both `time=48:00:00`.

---

## Shared front matter (Stages 0–5)

```mermaid
flowchart TD
    A0["Stage 0: parse args<br/>load WT PDB + seq<br/>load MSA (a3m)"]
    A1["Stage 1: MSA conservation<br/>run_msa_conservation(seq, uniref_db)<br/>→ cons[L], n_hits"]
    A2["Stage 2: active-site transfer<br/>detect_active_site(seq, transfer_json)<br/>Swiss-Prot + M-CSA consensus (Otsu)<br/>→ active_1b, assigned, src_counts"]
    A3["Stage 3: interface detection<br/>(if --complex-cif) interfaces.py<br/>cutoff=--interface-contact-cutoff Å<br/>→ interfaces{label: positions}"]
    A4["Stage 4: build frozen[]<br/>active site ∪ interfaces ∪<br/>(cons ≥ conservation_freeze)"]
    A5["Stage 5: load PLM stack<br/>ESM-2 3B + mhk32 MLM adapter<br/>head_best.pt state_dict unwrap"]
    A6["Stage 5b: RefoldClient<br/>launch ESMFold worker<br/>wait_ready()"]

    A0 --> A1 --> A2 --> A3 --> A4 --> A5 --> A6
    A6 -->|Pipeline A| BA["Stage 6A"]
    A6 -->|Pipeline B| BB["Stage 6B"]

    classDef shared fill:#f5f5f5,stroke:#333,stroke-width:1px,color:#111;
    class A0,A1,A2,A3,A4,A5,A6 shared;
```

Notes:
- Conservation threshold defaults to 0.90 in both pipelines
  (`--conservation-freeze 0.90`).
- Active-site RMSD cap = `--core-rmsd-cap` (default 1.0 Å in both).
- Interface RMSD cap = `--interface-rmsd-cap` (default 1.5 Å in both).
- `head_best.pt` load path unwraps the checkpoint dict (`state_dict` key) —
  fix applied in both scripts (2026-08-19).

---

## Pipeline A — Gibbs + Metropolis-Hastings

`scripts/11_generate.py` — MLM-in-the-loop proposer, contact/span-coupled
masking, classifier-score acceptance with periodic refold rollback.

```mermaid
flowchart TD
    A6["Stage 6A entry:<br/>build mask units<br/>(contacts + spans + singletons)<br/>coupling_mode = both"]
    LEVELS["for lvl in range(n_designs):<br/>schedule(lvl) →<br/>mask_rate ∈ [0.05, 0.20]<br/>gamma ∈ [2.5, 1.0]<br/>target_mut_frac ∈ [0.05, 0.30]"]
    INIT["cur = WT; best = WT<br/>T = mh_t0 (0.05)<br/>t_decay = (mh_t1 / mh_t0)^(1/(iters-1))<br/>mh_t1 = 0.005"]
    GIBBS{"for it in range(gibbs_iters)"}
    MASK["sample_mask_units(cons, units,<br/>mask_rate, gamma)"]
    PROP["prop = mlm_fill(cur, mask_pos)"]
    SCORE["ps = score(prop)<br/>= classifier head(mean-pool)"]
    BUDGET{"n_mut(prop) ≤ mut_budget?<br/>(max_mut_frac × L)"}
    MH{"Δ = ps − cur_score<br/>Δ ≥ 0 or U(0,1) < exp(Δ/T)?"}
    ACCEPT["cur ← prop<br/>if ps > best: best ← prop<br/>T *= t_decay"]
    REFOLD{"it % refold_every == 0<br/>AND cur != last_good?"}
    RMSD["refold_rmsd_multi(cur, protected_sets)<br/>protected: {active_site, iface_*}<br/>caps: core=1.0Å, iface=1.5Å"]
    PASS{"all sets ≤ cap?"}
    ROLLBACK["cur ← last_good<br/>shrink mask_rate<br/>consec_fails += 1"]
    KEEP["last_good ← cur<br/>consec_passes += 1<br/>recover mask_rate → mask_rate0"]

    A6 --> LEVELS --> INIT --> GIBBS
    GIBBS --> MASK --> PROP --> SCORE --> BUDGET
    BUDGET -- no --> GIBBS
    BUDGET -- yes --> MH
    MH --> ACCEPT --> REFOLD
    REFOLD -- no --> GIBBS
    REFOLD -- yes --> RMSD --> PASS
    PASS -- no --> ROLLBACK --> GIBBS
    PASS -- yes --> KEEP --> GIBBS

    WRITE["Stage 7A:<br/>designs[] ← best per level<br/>candidates_&lt;pheno&gt;.json<br/>fields: design_id, sequence,<br/>classifier_score, biophysical_score,<br/>mutations, trace, n_refolds,<br/>n_rollbacks, per-set rmsds"]
    GIBBS -.->|iters done| WRITE

    classDef stage fill:#fef3c7,stroke:#a16207,color:#111;
    classDef gate fill:#fee2e2,stroke:#b91c1c,color:#111;
    classDef ok fill:#d1fae5,stroke:#065f46,color:#111;
    class A6,LEVELS,INIT,MASK,PROP,SCORE,ACCEPT stage;
    class BUDGET,MH,REFOLD,PASS,RMSD gate;
    class KEEP,WRITE ok;
    class ROLLBACK gate;
```

**Key parameters (defaults):**
- `--n-designs 3` (aggressiveness levels)
- `--gibbs-iters 24`
- `--mh-t0 0.05 --mh-t1 0.005` (geometric anneal per level)
- `--refold-every 4`
- `--mask-rate` schedule 0.05 → 0.20 across levels
- `--max-mut-frac 0.30` (hard cap on residues changed)
- `coupling_mode=both` — contact-pair + span-length units

**Acceptance geometry:** score is directional (classifier alone), fold is
gated periodically (every 4 iters); a run can *end below its peak* under MH,
so `best` is tracked separately and returned as the design.

---

## Pipeline B — MPNN-in-the-loop with hard/soft two-tier gate

`scripts/11_generate_pipelineB.py` — LigandMPNN proposer over `n_rounds`
rounds at cycling temperatures, per-candidate refold + RMSD *hard* gate,
then a multiplicative *soft* score product for ranking.

```mermaid
flowchart TD
    B6["Stage 6B entry:<br/>fixed_positions_1b =<br/>active_site ∪ interface ∪<br/>(cons ≥ conservation_freeze)<br/>bias_vec = load_bias_AA(json, pheno)<br/>temps = [0.05, 0.10, 0.20]"]
    ROUND{"for r in range(n_rounds=4)"}
    MPNN["mpnn_propose(<br/>mpnn_repo, mpnn_env=ligandmpnn,<br/>wt_pdb, design_chain,<br/>fixed_positions_1b, bias_AA_vec,<br/>T=temps[r % 3],<br/>n_designs=batch_per_round=6,<br/>model=ligand_mpnn)<br/>subprocess: conda run -n ligandmpnn"]
    LOOPK{"for k in range(batch_per_round)"}
    LEN{"len(s) == L?"}
    T1A{"Tier 1a<br/>bad_cons (cons ≥ freeze<br/>positions mutated) empty?"}
    T1B["Tier 1b<br/>refold_rmsd_multi(s, protected_sets)<br/>caps: core=1.0Å, iface=1.5Å"]
    T1BQ{"all sets ≤ cap?"}
    T2["Tier 2 soft scores:<br/>p_clf = score_classifier(s)<br/>ll = seq_loglik(s)<br/>ll_ratio = exp(ll − wt_ll)<br/>p_cpl = coupling_preservation(s)<br/>score_product = p_clf × ll_ratio × p_cpl"]
    KEEP["survivors[r].append(<br/>sequence, T, subscores,<br/>score_product, rmsds,<br/>mpnn_overall_confidence,<br/>n_mutations)"]
    REJ["rejected[r].append(reason)<br/>reason ∈ {length_mismatch,<br/>conservation, rmsd}"]
    ALL["all_survivors.extend(survivors[r])<br/>round_stats.append(counts)"]

    B6 --> ROUND --> MPNN --> LOOPK --> LEN
    LEN -- no --> REJ
    LEN -- yes --> T1A
    T1A -- no --> REJ
    T1A -- yes --> T1B --> T1BQ
    T1BQ -- no --> REJ
    T1BQ -- yes --> T2 --> KEEP
    KEEP --> LOOPK
    REJ --> LOOPK
    LOOPK -.->|batch done| ALL --> ROUND

    RANK["Stage 6B rank:<br/>all_survivors.sort(<br/>key=score_product, desc)<br/>top = all_survivors[:n_designs]"]
    WRITE["Stage 7B:<br/>candidates_&lt;pheno&gt;.json<br/>pipeline=B<br/>extras: wt_seq_loglik,<br/>mpnn_model, mpnn_temperatures,<br/>n_rounds, batch_per_round,<br/>n_survivors_total,<br/>round_stats, bias_aa,<br/>wt_contact_pairs<br/>designs[*].subscores,<br/>designs[*].score_product,<br/>designs[*].rmsds,<br/>designs[*].mpnn_overall_confidence"]
    ROUND -.->|rounds done| RANK --> WRITE

    classDef stage fill:#e0e7ff,stroke:#3730a3,color:#111;
    classDef gate fill:#fee2e2,stroke:#b91c1c,color:#111;
    classDef ok fill:#d1fae5,stroke:#065f46,color:#111;
    class B6,MPNN,T2,ALL,RANK stage;
    class LEN,T1A,T1B,T1BQ gate;
    class KEEP,WRITE ok;
    class REJ gate;
```

**Key parameters (defaults):**
- `--n-rounds 4 --batch-per-round 6` → up to 24 candidates/run
- `--temperatures 0.05,0.10,0.20` (cycled by round mod len)
- `--n-designs 3` (top-K after ranking survivors)
- `--conservation-freeze 0.90`
- `--core-rmsd-cap 1.0 --interface-rmsd-cap 1.5`
- `--mpnn-model ligand_mpnn` (LigandMPNN v1)
- `--bias-aa-json data/bias_aa_by_phenotype.json`

**Acceptance geometry:** every proposal is folded and gated (no periodic
skipping), and the ranker is a *product* of three sub-scores, so a design
must be plausible under the classifier, close to WT under the language
model, AND preserve WT contact patterns to rank at all.

---

## What changed vs `docs/generation_pipeline.md`

The older `generation_pipeline.md` predates the mhk32 adapter refactor and
describes MPNN as an out-of-loop gate on top of Gibbs. This document reflects
the current scripts:

| Aspect | Old doc | Pipeline A (now) | Pipeline B (now) |
|---|---|---|---|
| Proposer | ESM-2 + adapter | ESM-2 + mhk32 adapter | LigandMPNN (subprocess) |
| Acceptance | classifier score | classifier + MH anneal | product (clf × llR × cpl) |
| Fold gate | out-of-loop MPNN + refold | in-loop refold every 4 iters | in-loop refold per candidate |
| Rollback | discard-only | shrink mask_rate + revert | reject-only (no revert) |
| Composition bias | – | – | per-phenotype bias_AA vector |
| Interfaces | active-site only | active + per-interface RMSD | active + per-interface RMSD |

## Progress monitoring rubric (all 4+4 jobs, 48 h walltime)

- Pipeline A logs: `is621_gen_<JOBID>.log`, line format
  `[gen] iter=<i>/24 accept=<n> ...` — expect ~24 iters × 3 levels per pheno.
- Pipeline B logs: `is621_genB_<JOBID>.log`, line format
  `[genB] === round <r>/4 batch=6 T=<T> ===` — expect 4 rounds × 6 candidates.
- `squeue -u $USER -o '%.10i %.16j %.10L'` for remaining walltime.
- If a first Pipeline A iter > 20 min or a first Pipeline B round > 30 min,
  48 h may still be tight; log the rate and check back at ~1 h.

## Currently deployed job IDs (2026-08-19)

| Job | Script | Phenotype |
|---|---|---|
| 1178426 | 11_generate.py (A) | thermophile |
| 1178427 | 11_generate.py (A) | halophile |
| 1178428 | 11_generate.py (A) | hyperthermophile |
| 1178429 | 11_generate.py (A) | alkaliphile |
| 1178430 | 11_generate_pipelineB.py (B) | thermophile |
| 1178431 | 11_generate_pipelineB.py (B) | halophile |
| 1178432 | 11_generate_pipelineB.py (B) | hyperthermophile |
| 1178433 | 11_generate_pipelineB.py (B) | alkaliphile |

All 8 running on `gpu_h200 node-224-2t-8gpu-1` at 48 h walltime.
