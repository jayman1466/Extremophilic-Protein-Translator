# Generative pipeline: extremophilic enzyme redesign

**Scope.** The end-to-end runtime pipeline that turns one input enzyme sequence
into ranked, phenotype-steered, fold-preserving designs. This document fills in
the stages the webapp catalog (`webapp/pipeline_options.py`) advertises and the
§16 masked-generation design sketches, wiring them to the symbols that already
exist in the repo (`src/eptrans/modeling/masking.py`, `webapp/aggressiveness.py`,
the Stage-2 cached heads, the coupling-aware contact machinery).

Pipeline (this is the order the SLURM job executes, and the order below):

```
input enzyme sequence
   │
   ├─▶ (1) MSA generation ─────────────▶ homolog alignment
   │                                        │
   ├─▶ (2) conservation weighting ◀─────────┘  →  c_i per column
   │                                        │
   ├─▶ (3) active-site annotation ──────────┴──▶  frozen[] (immutable set)
   │                                                │
   ▼                                                ▼
(4) masked generation (Gibbs + contact-pair, conservation-gated)
   │        proposer = ESM-2 3B + extremophilic MLM adapter
   ▼
(5) scoring  — per-phenotype cached head (directional signal)
   │
   ▼
(6) MPNN structural gate  →  refold  →  catalytic-RMSD gate
   │            (accept / reject each Gibbs move)
   ▼
ranked designs (5 spanning aggressiveness levels)
```

The governing division of labor (from `modeling_design.md` §8): **PLM proposes →
classifier scores → MPNN gates**, active site protected both preventively
(frozen, never masked) and verificationally (catalytic-atom RMSD gate).

---

## Stage 1 — MSA generation

**Purpose.** Produce a homolog alignment for the input enzyme. Its *only* job in
this pipeline is to supply per-column conservation (Stage 2); it is not used for
folding here (folding is single-sequence ESMFold in Stage 6). A column conserved
across hundreds of homologs is functional regardless of what any annotation
database says — so the MSA is an orthogonal, annotation-free source of the
immutable set.

**Method.** `MMseqs2` against `UniRef30` (webapp default `msa=mmseqs_uniref30`;
deeper `ColabFold envDB` is the advertised upgrade path). The ColabFold MSA
protocol (`mmseqs search` → `expandaln` → `result2msa`) is the reference
implementation; run it as a remote CPU job (MSA generation is embarrassingly
CPU-bound, not GPU work — keep it off the GPU partition).

**Inputs → outputs.**
- in: mature-chain query sequence (signal peptide already cleaved per
  `modeling_design.md` §3).
- out: `msa.a3m` (aligned homologs, query as row 0).

**Parameters.**
- coverage ≥ 0.5, sequence-identity floor ~0.30 (keep the alignment in the
  functional-homolog regime; avoid drift into unrelated folds).
- depth cap ~10k sequences (conservation estimates saturate well before this;
  the cap bounds runtime and the Stage-2 weighting cost).

**Failure / degraded path.** Shallow MSA (few homologs — orphan enzyme). Below a
depth floor (~25 effective sequences), conservation is unreliable: fall back to
annotation-only freezing (Stage 3) and set a uniform-prior conservation vector
(`c_i = 0` everywhere non-frozen, i.e. `gamma`-gating becomes inactive and the
generator relies on the adapter + annotation freeze alone). This is the
"structure-poor / homolog-poor input" branch flagged as open in §16.

---

## Stage 2 — conservation weighting

**Purpose.** Convert the MSA into a per-residue conservation score `c_i ∈ [0,1]`
that (a) contributes to the immutable set (very-high-conservation columns are
frozen) and (b) *targets* where the generator is allowed to mutate (the
`gamma`-gated mask weight `(1 - c_i)^gamma`, already implemented in
`masking.mask_weights`).

**Method — sequence-weighted column conservation.** Naive column entropy
over-counts redundant near-duplicate homologs (a MSA dominated by one clade
looks "conserved" everywhere). Use **position-specific sequence weighting**
(Henikoff or the `1/n_clusters`-style down-weighting) so each *independent*
homolog contributes ~equally:

1. Cluster/weight rows: assign each aligned sequence a weight `w_s` inversely
   proportional to its redundancy (Henikoff weights, or cluster at ~62% id and
   weight by `1/cluster_size`).
2. Per column `i`, compute a weighted conservation. Two reasonable estimators;
   use the first, keep the second as a diagnostic:
   - **weighted relative entropy** to a background distribution `q`:
     `c_i = Σ_a p_i(a) log(p_i(a)/q(a))` normalised to `[0,1]` by the max over
     columns (information-content form; rewards columns that are both
     low-entropy *and* deviate from background).
   - weighted Shannon: `c_i = 1 - H_w(col_i)/log20`.
3. Down-weight gappy columns: multiply `c_i` by `(1 - gap_frac_i)` so a column
   that is mostly gaps in the query's homologs isn't spuriously "conserved."

**Inputs → outputs.**
- in: `msa.a3m`, query length `L`.
- out: `conservation.npy` — `float array (L,)`, aligned to query residue coords.

**Where it plugs in.**
- **freeze rule:** `c_i ≥ conservation_freeze_thresh` (default 0.90) →
  contributes to `frozen[]` (Stage 3 union). This is the "conservation as a
  **hard mask**, not a soft PLM preference" decision from `modeling_design.md`
  §2 Oracle 2.
- **targeting:** the full `conservation` vector is passed to
  `masking.sample_mask_units(conservation, units, mask_rate, gamma)` — moderate
  conservation biases the sampler away from a column without hard-freezing it,
  and `gamma` (from the aggressiveness schedule) controls how sharply.

---

## Stage 3 — active-site annotation (the immutable set)

**Purpose.** Assemble `frozen[]`, the boolean length-`L` array of positions the
generator may **never** mask. This is the preventive half of active-site
protection.

**Sources (union across all selected; webapp `active_site` section, `multi=True`).**
- **M-CSA** — curated catalytic residues for the enzyme (or its closest MSA
  homolog with an M-CSA entry), mapped onto query coords.
- **UniProt / Swiss-Prot** `ACT_SITE` + `BINDING` features (via
  `mcp-protein-annotation`).
- **InterPro / Pfam** active-site and binding-site positional features.
- **Ligand/cofactor-contacting residues** — if a holo structure or a docked
  cofactor is available, residues within a contact radius of the ligand.
- **High-conservation columns** from Stage 2 (`c_i ≥ conservation_freeze_thresh`).

**Coordinate mapping.** Each database hit is on some reference sequence; map to
query residue index through the MSA (query row 0) or a pairwise alignment. Record
provenance per frozen position (which source(s) flagged it) for the results
report — a residue frozen by *both* M-CSA and conservation is higher-confidence
than one frozen by a single Pfam feature.

**Inputs → outputs.**
- in: query sequence, `conservation.npy`, annotation-DB responses.
- out: `frozen.npy` — `bool (L,)`; `frozen_provenance.json` — per-position source
  list.

**Interaction with masking.** `frozen[]` is passed as the `frozen=` argument to
`masking.mask_weights` / `build_mask_units`, which hard-zeroes those positions'
mask weight and excludes them from every masking unit — the `gamma→∞` limit. It
is invariant across all aggressiveness levels (`aggressiveness.py` docstring:
"the active-site freeze … are invariant across all levels").

---

## Stage 4 — masked generation (Gibbs + contact-pair, conservation-gated)

**Purpose.** The proposer. Iteratively rewrite mutable positions toward the
target phenotype using the fine-tuned MLM, decoding coupled positions jointly.

**Proposer model.** ESM-2 3B + the project's **extremophilic MLM LoRA adapter**
(`esm2_3b_extremo`), loaded exactly as the classifier branch does
(`load_mlm_adapter_into_classifier` remap, or the MLM head form for logits). The
adapter is what biases fills toward extremophile-like residues; the base model
without it is the ablation (advertised, disabled by default).

**One Gibbs pass.**
1. **Build masking units** — `masking.build_mask_units(L, special, frozen,
   contact_pairs, spans)`. Contact pairs come from the input's ESM-2 contact map
   (`contact_pairs_from_map`, the SAME contact head used in coupling-aware
   training), so coupled residues form a joint unit; the remainder are span or
   singleton units.
2. **Select positions to mask** — `masking.sample_mask_units(conservation,
   units, mask_rate, gamma, rng)`. `mask_rate` and `gamma` come from the
   aggressiveness schedule (`aggressiveness.schedule(level)`); frozen/special
   positions have zero weight by construction. Masking whole units means a
   contact pair is masked and refilled *together*.
3. **Predict + decode jointly** — run the masked sequence through the adapted
   MLM; for each masked unit, decode its positions from the model's joint
   distribution over that unit (a coupled pair's two residues are sampled from
   the pair's joint, not two independent argmaxes) so a new salt bridge / coupled
   substitution can co-emerge — the specific limitation that plain masked-fill
   (`modeling_design.md` §4 option 2) suffers.
4. **Propose → gate → accept/reject** — the refilled sequence is a *proposal*,
   not an accepted step. It goes to Stage 5 (score) and Stage 6 (structural
   gate); a move is accepted only if it passes the gate and improves (or a
   Metropolis criterion on) the phenotype score. Rejected → revert those units,
   continue.

**Steering.** The directional signal is the **per-phenotype cached head** (Stage
5), not the bulk MLM logits. Contrastive delta-logit steering
(`logit_adapted − logit_base`) is deferred/validate-first per §16 — it is muddy
for pH (acido/alkali cancel in the pooled Stage-1 adapter); temp/salinity are
more coherent. Default = classifier-guided acceptance.

**Convergence (`aggressiveness.gibbs_stop_rule()`).** Stop at whichever fires
first:
- classifier-score plateau (Δ < `PLATEAU_TOL` over `PLATEAU_WINDOW` passes),
- mutation budget reached (`target_mut_frac` of mutable residues changed),
- acceptance collapse (accepted-move fraction < `ACCEPT_COLLAPSE`),
- hard cap (`MAX_GIBBS_ITERS`).

**Aggressiveness span.** A job produces N designs (default 5) spanning
conservative→aggressive via `aggressiveness.span_levels(N)` /
`resolve(N, override)`; each level is a coordinated `(target_mut_frac, mask_rate,
gamma)` triple. The frozen set and RMSD gate are identical across levels — only
the mutable-surface budget moves.

**Inputs → outputs.**
- in: query seq, `conservation.npy`, `frozen.npy`, contact map, phenotype,
  schedule.
- out: per-level accepted design sequence + its Gibbs trajectory (score per pass,
  mutations introduced) → `design_<level>.json`.

---

## Stage 5 — scoring (per-phenotype directional signal)

**Purpose.** At each Gibbs step, tell the sampler whether the proposal moved
*toward* the target phenotype. This is the objective the sampler climbs.

**Method.** The **Stage-2 cached head** for the selected phenotype
(`models/cached_probes/clf_<pheno>_cached/head_best.pt`), applied to the frozen
MLM-adapted backbone embedding of the proposal:
1. embed the candidate through the frozen adapted backbone (masked mean-pool,
   the exact representation the head was trained on — see `09_embed_secretome`),
2. `score = sigmoid(head(embedding))` — the phenotype logit.

Using the cached head keeps scoring cheap (one forward pass through the frozen
backbone per candidate; no head retraining) and *coherent* — the `(adapter,
head)` pair is matched by construction, since the adapter is fixed. Validated
signal: hyperthermophile AUPRC 0.898 / pair-AUC 0.924, thermophile 0.862 /
0.905 (pair-AUC ≫ 0.5 ⇒ the head tracks thermoadaptation, not taxonomy).

**Non-learnable secondary score (anti-gaming).** Alongside the learned head,
compute the cheap biophysical layer (`modeling_design.md` §2): thermophile →
charged-surface / salt-bridge count; halophile → acidic-surface (Asp+Glu)
fraction; acido/alkaliphile → net surface charge at target pH. The sampler's
objective is the learned score **gated by** biophysical agreement — a proposal
that raises the classifier but moves the biophysical proxy the wrong way is
treated as adversarial and rejected. This is the "keep the non-learnable layer
in the loop" defense against oracle exploitation.

**Inputs → outputs.**
- in: candidate sequence, phenotype, cached head.
- out: `{clf_score, biophysical_score, accept_signal}` per candidate.

---

## Stage 6 — MPNN structural gate → refold → catalytic-RMSD gate

**Purpose.** The verificational half of active-site protection and the fold-
integrity oracle. Cheap structural screen first, then a folding confirmation.

**6a — MPNN structural audit (periodic, not every step).** The inverse-folding
likelihood of a proposed sequence on the fixed WT backbone screens structurally
implausible designs without folding anything: a sequence MPNN finds unlikely on
the native backbone is likely destabilising. But a full MPNN pass per Gibbs
proposal is too expensive (up to `MAX_GIBBS_ITERS` × N-levels passes), and MPNN
only at the very end lets a trajectory drift far into implausible territory
before it's caught. Use a **two-tier gate**:

- **Every step — cheap accept/reject.** The per-step objective is the phenotype
  score (Stage 5) gated by the non-learnable biophysical layer — both already
  computed each step. This catches obvious-bad moves at zero extra cost.
- **Every K accepted moves — MPNN audit.** Run MPNN on the accumulated design;
  if its margin vs. WT has fallen below `MPNN_THRESH`, **roll back to the last
  MPNN-passing checkpoint** and resume with a fresh RNG seed. This bounds
  structural drift to K accepted moves at `1/K` the MPNN cost. `K` is set by the
  aggressiveness schedule and scales *inversely* with boldness (conservative
  ~every 8 moves, aggressive ~every 3), so the audit fires per roughly-constant
  *mutation count* rather than per step — a bold run that mutates fast is checked
  more often. `K` lives in `aggressiveness._SCHEDULE` alongside
  `mask_rate`/`gamma`/`target_mut_frac`.

**MPNN variant — auto-selected on ligand presence, not a manual toggle.**
- **Holo input (ligand / cofactor / metal present in the folded structure) →
  LigandMPNN** (`ligandmpnn` skill, Dauparas et al. 2023). It conditions the
  sequence on the ligand context, so a pocket-lining or metal-coordinating
  residue is scored *in the presence of* the cofactor. ProteinMPNN is
  ligand-blind (backbone only) and can favour a residue that is geometrically
  fine but destroys coordination — wrong for metalloenzymes and cofactor-
  dependent enzymes, which is much of this project's target set.
- **Apo input (no ligand resolved, or sequence-only → single-sequence ESMFold) →
  ProteinMPNN** (`proteinmpnn` skill). With no ligand context LigandMPNN degrades
  to ~ProteinMPNN anyway, so there's nothing to gain from the heavier path.
- **Selection is automatic** from whether the WT structure carries relevant
  heteroatoms/metals — the user does not have to know whether their enzyme is a
  metalloenzyme to get the right gate. (`SolubleMPNN` is deliberately *not* used:
  it biases toward cytosolic solubility, the opposite of this project's secreted-
  protein context.)
- **Consequence — the ligand choice is one decision seen from both ends.** When
  LigandMPNN gates, Stage 3 must include the ligand-coordinating shell in
  `frozen[]` (already listed there) and Stage 6c must measure **coordination
  geometry** (metal–ligand / cofactor-contact distances), not just backbone RMSD.

**6b — refold survivors.** Fold each surviving accepted design with **ESMFold**
(single-sequence; `esmfold2` skill). Fold WT with the *same* method so the RMSD
reflects the sequence change, not cross-method bias (webapp `fold` help text).
Require high pLDDT and low global backbone RMSD to the WT fold (Oracle 3).

**6c — catalytic-geometry gate.** The decisive functional check: superpose
design onto WT and measure **catalytic-atom RMSD** over the `frozen[]` active-site
residues specifically (not just global backbone). A design that keeps global
fold but distorts the catalytic geometry fails. When the input is holo (LigandMPNN
path), this gate additionally checks **coordination geometry** — metal–ligand and
key cofactor-contact distances against WT — since a preserved catalytic-residue
RMSD does not by itself guarantee the cofactor still coordinates. This is the
verificational guarantee that complements the preventive freeze.

**Gate ordering (cheap→expensive).** MPNN (per proposal) → global pLDDT/RMSD
(per accepted design) → catalytic-RMSD (per folded design). Most rejections
happen at the cheapest stage.

**Inputs → outputs.**
- in: accepted design sequences, WT backbone (folded once up front).
- out: `design_<level>.pdb`, `{mpnn_margin, plddt, backbone_rmsd,
  catalytic_rmsd, pass}` per design.

---

## Gibbs loop assembly (stages 4–6 together)

```
fold WT once (ESMFold) ; compute WT contact map, MSA, conservation, frozen[]
for level in span_levels(N):                       # aggressiveness 1..5
    sched = schedule(level)                         # mask_rate, gamma, target_mut_frac
    mpnn = LigandMPNN if wt_has_ligand else ProteinMPNN     # auto-selected (6a)
    seq = query ; checkpoint = query ; n_accept = 0
    for it in range(MAX_GIBBS_ITERS):
        units = build_mask_units(L, special, frozen, contact_pairs, spans)
        mask  = sample_mask_units(conservation, units, sched.mask_rate, sched.gamma)
        prop  = adapted_mlm.fill_joint(seq, mask, units)     # coupled decode
        # --- cheap per-step accept/reject (stage 5) ---
        s = clf_score(prop, phenotype) gated by biophysical(prop)
        if accept(s, prev_s):                                  # score-driven accept
            seq, prev_s = prop, s ; n_accept += 1
            # --- periodic MPNN structural audit (6a), every K accepted moves ---
            if n_accept % sched.mpnn_every == 0:
                if mpnn.margin(seq, wt_backbone) < MPNN_THRESH:
                    seq, prev_s = checkpoint, clf_score(checkpoint, phenotype)  # roll back
                    reseed(rng)
                else:
                    checkpoint = seq                            # advance last-good
        if gibbs_stop_rule fires (plateau / budget / collapse / cap): break
    # --- confirm the level's final design ---
    pdb = esmfold(seq)
    if plddt ok and backbone_rmsd ok and catalytic_rmsd(pdb, wt) ok:
        emit design_<level>
```

Key invariants:
- `frozen[]` and the catalytic-geometry gate are identical across all levels;
  only `sched` (the mutable-surface budget *and* the MPNN audit cadence
  `mpnn_every`) varies. (`aggressiveness.py` contract.)
- the MPNN gate is **periodic in-loop** (every `mpnn_every` accepted moves, with
  rollback to the last MPNN-passing checkpoint) plus a full gate on the final
  design — not per-proposal. Cheap per-step accept/reject is the phenotype +
  biophysical score.
- MPNN variant is auto-selected per input: LigandMPNN for holo (ligand/cofactor/
  metal), ProteinMPNN for apo.
- coupled positions are always masked and refilled as a unit (never split), so
  the contact-pair coupling learned in Stage-1/Stage-2 training is honoured at
  generation time.
- scoring is the cached head (fixed adapter → coherent `(adapter, head)`); the
  bulk delta-logit contrastive term stays off until per-phenotype validation
  (§16 deferred item).

---

## What is committed vs. what this spec adds

**Already in the repo (referenced above, not re-implemented):**
- `masking.py` — `mask_weights`, `sample_mask_positions`,
  `contact_pairs_from_map`, `make_span_units`, `build_mask_units`,
  `sample_mask_units` (the conservation-gated, contact-pair-aware masking core).
- `aggressiveness.py` — level→schedule map, `span_levels`, `resolve`,
  `gibbs_stop_rule` (the aggressiveness axis + convergence criteria).
- Stage-2 cached heads (`models/cached_probes/clf_<pheno>_cached/head_best.pt`)
  and the frozen-backbone embedding path (`09_embed_secretome`,
  `10_train_cached_probe`) — the scoring oracle.
- `webapp/pipeline_options.py`, `aggressiveness.py`, `backends.py` — the
  selection catalog + the (stubbed) SLURM submit/poll the runtime plugs into.

**This spec fills in (the runtime pipeline, currently stubbed in
`SlurmBackend.submit`):**
- Stage 1 MSA generation (MMseqs2/ColabFold protocol, CPU job).
- Stage 2 sequence-weighted conservation estimator → `conservation.npy`.
- Stage 3 multi-source active-site union → `frozen.npy` + provenance.
- Stage 4 the Gibbs driver: joint coupled decode from the adapted MLM, wired to
  the existing masking units + schedule + stop rule.
- Stage 5 cached-head scoring + non-learnable biophysical gate.
- Stage 6 periodic in-loop MPNN audit (every `mpnn_every` accepted moves, with
  checkpoint rollback) → ESMFold refold → catalytic-geometry gate, ordered
  cheap→expensive; MPNN variant auto-selected (LigandMPNN holo / ProteinMPNN apo).
  Requires adding an `mpnn_every` field to `aggressiveness._SCHEDULE` (inverse to
  boldness: ~8 conservative → ~3 aggressive).
- The assembled per-level Gibbs loop and its result artifacts
  (`design_<level>.{json,pdb}`, trajectories) that `poll()` harvests.

**Open items carried from §16 (decide before/at implementation):**
- Contrastive delta-logit steering — validate per-phenotype (esp. pH
  anti-correlation) before enabling; default is classifier-guided.
- Standalone masked-gen vs. MPNN-coupled — this spec treats MPNN as a periodic
  in-loop audit (tightly coupled path); the MLM-standalone degraded path drops 6a
  and relies on the refold + catalytic-geometry gate only.
- MPNN audit cadence `mpnn_every` (K) — the ~8→3 inverse-to-boldness schedule is
  a starting point; tune once real trajectories show how fast drift accumulates
  per accepted move at each level.
- Contact-precompute batching — the WT contact map is one-off per job so it is
  not the bottleneck here; batching matters for the training-time precompute, not
  generation.
