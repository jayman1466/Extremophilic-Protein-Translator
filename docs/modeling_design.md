# Modeling design: extremophilic enzyme translator

**Goal.** Input: an enzyme sequence. Output: a sequence with the *same enzymatic
activity* but a *more extremophilic scaffold* (thermo-/halo-/acido-/alkaliphilic).

This is a constrained, multi-objective sequence-design problem: harden the
scaffold while preserving the fold and the catalytic machinery.

---

## 0. The reframing that governs everything

The goal is about **protein-level** extremophilicity; the dataset is labeled by
**organism-level** environment. These differ — a protein from a hyperthermophile
is not automatically thermostable (chaperones, cytoplasmic context can prop up a
marginally stable protein).

**Why the secretome focus resolves this.** Secreted proteins fold and function
*outside* the cell, with no chaperone or cytoplasmic buffering. A thermophile's
secreted enzyme must be *intrinsically* extremophilic. So the secretome is the
compartment where "from an extremophile" ≈ "is itself extremophilic." This is the
strongest single de-risking decision in the project.

**Consequence for training:** use the **mature chain** (signal peptide cleaved),
not the precursor.

---

## 1. Architecture — hybrid (structure constrains, sequence models drive)

Two families:

- **Sequence-space (PLM-centric):** fine-tune a protein LM on extremophile
  secreted proteins; mutate the target by masked-fill. Cheap, fold-blind.
- **Structure-space (MPNN-centric):** fold the enzyme, inverse-fold the backbone
  with the active site pinned. Guarantees geometry, but pulls toward generic
  PDB-average sequences.

**Recommended hybrid loop:**
1. Fold input enzyme (ESMFold2 fast; AF2+MSA for higher confidence).
2. Freeze the immutable set (catalytic + ligand-contacting residues).
3. Propose mutations outside that set (MPNN redesign and/or PLM masked-fill).
4. Score each candidate on three oracles (Section 2); accept/reject in a
   directed-evolution loop.
5. Periodically re-fold accepted candidates to confirm scaffold integrity.

---

## 2. The three oracles (this is where the work is)

### Oracle 1 — extremophilic phenotype
Learned classifier (extremophile vs matched mesophile), **plus** a
non-learnable biophysical layer.

- **Failure mode: phylogenetic shortcut** — classifier learns clade, not trait.
- **Mitigations (already in the pipeline):**
  - Phylogenetically-matched mesophile **outgroups** — force the model onto the
    trait, not the clade. (This is *why* the outgroup design exists.)
  - **Cluster-based splits** (mmseqs ~30–50% id) — no train/test homolog leak.
- **Biophysical layer (cheap, non-gameable secondary score):** thermophile →
  more charged surface / salt bridges; halophile → strongly acidic surface
  (Asp/Glu excess); acido-/alkaliphile → pH-tuned surface charge. If classifier
  saliency doesn't recover these, it is cheating.

### Oracle 2 — enzymatic function retention
Conserve the structural correlates of activity (can't assay in silico).

- **Active-site ID:** Pfam/InterPro active-site features (protein-annotation
  connector), M-CSA / UniProt ACT_SITE for the specific enzyme, **plus** homolog
  MSA conservation as an orthogonal signal (a column conserved across hundreds of
  homologs is functional regardless of annotation). Use conservation as a **hard
  mask**, not a soft PLM preference.
- **Geometry conservation:** MPNN/LigandMPNN enforce at design time; re-fold and
  check active-site-atom RMSD + pocket shape after.

### Oracle 3 — fold integrity
Self-consistency: fold the design (ESMFold2/AF2), require high pLDDT and low
backbone RMSD to the original. Standard de-novo validation; catches
confident-but-wrong sequence-model output.

> **Oracle 2 is elaborated in detail in Section 10** (function-retention
> scoring): active-site ladder, MSA conservation, Foldseek structural transfer,
> and the gated composite score.

---

## 3. Using the SignalP output for training

- **Train on the mature chain** (residues `cs_after+1…end`); drop the signal
  peptide — it is cleaved and is noise for scaffold work.
- **`label_confidence` → sample weighting** in the loss (high > medium).
- **`signalp_class` / `anchoring`** as optional conditioning (soluble vs
  membrane-anchored have different surface constraints) — or separate models.

---

## 4. Generation engine — options ranked

1. **MPNN redesign** (recommended workhorse): coordinated surface redesign with
   active site pinned; bias toward extremophilic composition. LigandMPNN if a
   cofactor/metal is present.
2. **Masked-fill with fine-tuned PLM** (good v1): simple, local edits; struggles
   with coordinated multi-residue changes (e.g. a new salt bridge = 2 coupled
   mutations).
3. **Diffusion (RFdiffusion-style):** held in reserve — wrong tool for
   *preserving* a fold; risks function. (No skill installed; would need setup.)

---

## 5. Phased, de-risked roadmap

Scoring functions are needed by every generation approach and are independently
validatable — build them first.

1. **Phenotype classifier + biophysical layer**, validated on cluster-split
   matched pairs → calibrated score with known FP behavior.
2. **Function-retention scorer** (DB active-site mask + homolog conservation +
   geometry check), validated on families where the right residues are known.
3. **PLM fine-tune** on mature extremophile chains, confidence-weighted; verify
   its pseudo-likelihood separates extremophile/mesophile better than base.
4. **Generation loop** (MPNN + masked-fill hybrid) with oracles as accept/reject.
5. **Validation set:** mesophilic enzymes with known thermostable homologs — check
   the pipeline rediscovers real stabilizing mutations.

**Cross-cutting risk — adversarial oracle exploitation:** generator fools the
learned classifier without actually improving stability. Defenses: ensemble phenotype
scorers; keep the non-learnable biophysical layer in the loop; gate every accept
through structure self-consistency (hard to game).

---

## 6. Tooling available in this environment (verified in catalog)

- **PLMs / embeddings / mutation scoring:** `fair-esm2` (ESM-2), `esmfold2`
  (also exposes ESMC 300M/600M/6B masked-LM logits + mutation scoring).
- **Inverse folding / redesign:** `proteinmpnn`, `ligandmpnn`, `solublempnn`.
- **Structure prediction (self-consistency):** `esmfold2`, `alphafold2`, `boltz`,
  `chai1`.
- **Annotation:** `mcp-protein-annotation` (InterPro/Pfam domains, active sites).
- **Not installed:** ESM3, Profluent E1, RFdiffusion — would need setup if chosen.

---

## 7. Resolved decisions

- **Base PLM:** start with **ESMC** (or ESM-2) — installed, best-in-class for the
  *scorer/embedding* role we need. ESM3 / Profluent E1 offer no expected gain here
  because their advantage is *generation*, which MPNN now handles; they are gated
  + need setup. **Revisit only if the generation strategy changes** to PLM-driven.
- **Phenotype model:** **per-phenotype** classifiers (one per class), matching the
  overlapping natural-size dataset.
- **Generator:** **MPNN** (ProteinMPNN / LigandMPNN when cofactor/metal present).
- **Output:** **5 designs per query** spanning a conservative→aggressive ladder
  (Section 8), each folded + scored on all three oracles, presented as a
  tradeoff table.

## 8. Roles in the MPNN + PLM loop (the key architecture point)

MPNN and the PLM do **not** compete — different stages:

1. **MPNN proposes** — inverse-folds the fixed backbone, active site + ligand
   contacts pinned, emits many candidate sequences.
2. **PLM scores + steers** (three jobs, none generative):
   - **embedding backbone for the phenotype classifier** (Oracle 1) — primary job;
   - **re-rank / filter MPNN output** by fine-tuned-on-extremophiles
     pseudo-likelihood — "looks like a natural extremophilic protein"; this is
     where fine-tuning pays off;
   - **mutation-effect guard** — masked-LM scores at conserved positions flag
     family-intolerable mutations (orthogonal to the active-site mask, feeds
     Oracle 2).
3. **Fold to verify** (Oracle 3) → accept.

Slogan: **MPNN is the hand, the PLM is the taste.**

## 9. The 5-design aggressiveness ladder

"Aggressiveness" = a stability↔function tradeoff sweep, not one knob. Three
composable knobs, safest→boldest:

1. **Mutable region:** surface-only → surface+second-shell → all non-active-site.
   (Surface is where thermo/halo signatures live and is lowest fold-risk.)
2. **MPNN sampling temperature:** low (near wild-type) → high (more mutations).
3. **Extremophilic-bias strength:** how hard the PLM re-ranker pushes toward
   extremophilic statistics vs staying close to the input.

Deliverable: 5 designs along a conservative→aggressive path, each folded + scored,
shown as a table (e.g. design 1: +2 stability / 3 mut / RMSD 0.4 Å; design 5: +8
stability / 25 mut / higher drift). Show the **frontier**, not 5 samples at one
setting. Optionally collapse to a single "aggressiveness" summary or a calibrated
predicted-ΔTm-equivalent.

---

## 10. Function-retention scoring (Oracle 2, detailed)

Function is the one property we **cannot simulate** and have **no labeled
training set** for. Principle: build several *interpretable, orthogonal* proxies
with **hard gates**, not one learned/blended score, and calibrate on enzymes
where the answer is known. The three factors fail in uncorrelated ways, which is
why combining them works.

### Factor 1 — MSA conservation (universal coverage → the backbone term)
- Every enzyme has homologs, so this always fires; it covers what M-CSA misses.
- **Weight sequences before scoring** (Henikoff, or cluster @~62% + weight by
  cluster) so redundant near-duplicate homologs don't dominate.
- Score per column as sequence-weighted information content, or better an
  **evolutionary rate** (Rate4Site-style — accounts for phylogeny; slow columns
  = functionally constrained).
- **Soft penalty**, proportional to conservation, for mutating a column.

### Factor 2 — active-site identification + catalytic-atom RMSD
Active-site ID is a **confidence-tiered ladder** (take best available, record
which tier fired):
1. Direct **M-CSA** entry (highest confidence; ~1k curated — low recall)
2. M-CSA **homolog transfer** (align to nearest reference, map catalytic columns)
3. **Swiss-Prot** `ACT_SITE`/`BINDING`/`METAL` features (curated, broad)
4. **InterPro/Pfam** active-site position annotations
5. **Foldseek** structural-homolog transfer (the recall booster — see below)
6. Fallback: conservation peaks ∩ folded-structure ligand-pocket residues
   (*predicted* active site, flagged low-confidence)

RMSD monitoring: after redesign, fold + superpose on wild-type, measure RMSD over
**catalytic side-chain functional atoms** (Ser-OG, His-NE2, …), not just Cα —
reactive-atom geometry is what activity depends on.

### Factor 3 — ProteinMPNN geometry (reframed)
MPNN gives a **sequence↔backbone compatibility** score (NLL of sequence given
backbone) = "does this sequence plausibly fold into this shape." That is a
*fold-integrity* signal more than a *function* one. Its function contribution is
at **design time**: fix catalytic residues in the MPNN run (never mutated) and
read per-residue confidence around the pocket. Score feeds Oracle 3.

### The missing piece — Foldseek (structural homology)
Fold is far more conserved than sequence: enzymes at ~15% identity (invisible to
sequence search) can have superimposable active sites. When sequence search finds
no M-CSA/Swiss-Prot match, **Foldseek** vs PDB + AlphaFold DB finds a structural
homolog *with annotated catalytic residues* → transfer by superposition. This is
what turns the ladder from "well-studied enzymes only" into "most enzymes."
Make it first-class, not an afterthought.

### Composite — gates, not a blended number
No functional ground truth → a weighted sum is a trap (guessed weights; generator
exploits the softest term).
- **Hard gates (binary, reject on fail):** catalytic residue *identity* unchanged;
  catalytic side-chain-atom RMSD < ~1–1.5 Å.
- **Soft score (for gate-passers):** **product** of sub-scores in [0,1]
  (conservation term × fold-compatibility term) — a product means a near-zero on
  any factor kills the design; a sum would let a great fold paper over a destroyed
  residue. Carry the active-site-ID tier alongside as a trust flag.
- **Calibrate thresholds on the validation set** (mesophilic enzymes w/ known
  thermostable homologs): known stabilizing mutations must *pass*, known
  activity-killing mutations must *fail*. Only honest way to set the RMSD cutoff
  and conservation scale without functional labels.

### MSA tooling
- **MMseqs2 profile pipeline** (ColabFold-style iterative profile search): near
  HHblits sensitivity at a fraction of runtime; faster than jackhmmer/HHblits.
- **Search DB = UniRef30 (or UniRef90)**, not raw NCBI-nr — pre-clustered →
  deeper, less-redundant MSAs faster (redundancy removal is what conservation
  weighting wants anyway). Existing NCBI-nr mmseqs2 DB = fine fallback.
- **Foldseek** for the structural arm (own DB, structure input).
- DIAMOND ultra-sensitive is an option for raw speed, but MMseqs2 profile search
  is the better sensitivity/speed point for MSA building.

### Local databases to download (biotite)
| database | purpose | rough size |
|---|---|---|
| UniRef30 (or UniRef90) | fast MSA (MMseqs2 profile) | ~50–100 GB |
| M-CSA (entries + homolog lists) | curated catalytic residues | ~MB |
| Swiss-Prot (reviewed UniProt + features) | ACT_SITE/BINDING/METAL | ~1 GB |
| Pfam HMMs + InterPro | domain + active-site positions | ~10–20 GB |
| Foldseek PDB DB | structural active-site transfer | ~10 GB |
| Foldseek AlphaFold DB (optional) | deeper structural homologs | 100s GB (subset) |

Existing NCBI-nr mmseqs2 DB on biotite = fallback MSA source; add UniRef30 as
primary (clustering makes both search and conservation weighting cleaner).

---

## 11. Fine-tuning strategy (ESM + ProteinMPNN)

Fine-tuning serves two *uses* — masked fill-in (generation) and
classifying/scoring — but that is **one shared foundation with two heads**, not
two separate fine-tunes.

### Unifying architecture: one adapted backbone, two heads, one split
1. **Shared foundation — domain-adaptive continued pretraining.** ESM-2, MLM
   objective, continued on the extremophile secreted-protein corpus →
   likelihood surface shifts toward extremophilic statistics. **Unsupervised**
   (sequences only). Powers *both* uses; the single highest-value step.
2. **Head 1 — generation.** The MLM head is the generator: mask non-active-site
   positions, fill in → extremophile-flavoured. Same head's pseudo-likelihood =
   the "looks like a natural extremophile" re-ranker for MPNN output.
3. **Head 2 — scoring.** Classifier head on the same adapted backbone,
   supervised extremophile-vs-matched-mesophile = per-phenotype Oracle 1.

One backbone, two heads, **one consistent cluster split** underneath.

### ESM fine-tuning specifics
- **Parameter-efficient (LoRA/adapters), NOT full fine-tune.** Full FT on ~10^4
  sequences overfits AND causes catastrophic forgetting of the base model's
  structural knowledge — the very knowledge that protects enzyme function. LoRA
  (rank 8–16 on q/v proj) freezes the base, trains ~0.1% of params, one GPU,
  and keeps **multiple phenotype adapters swappable on one backbone** (no 5×650M
  copies).
- **Per-phenotype via adapters:**
  - generation: per-phenotype MLM adapter → *directed* fill-in. Even
    hyperthermophile (216 genomes × few-hundred secreted ≈ 10^4 seqs) is enough
    for LoRA (too little to train from scratch).
  - scoring: one shared adapted backbone + 5 light classifier heads (small
    classes borrow representation; boundaries stay independent).
- **Starting hyperparameters:**
  - backbone ESM-2 **3B** (t36) — chosen (resources available); LoRA keeps it
    single-GPU tractable (bf16 + gradient checkpointing, ~0.1% trainable).
  - LoRA rank 8–16, α 16–32, dropout 0.05, target q_proj/v_proj.
  - continued MLM 15% mask, LR 1e-4 + warmup, few epochs, early-stop on held-out
    pseudo-perplexity.
  - classifier head: mean-pool → 2-layer MLP (or attention pool); LR 1e-3 head /
    1e-5 adapter; early-stop on val AUPRC.
  - context: mature chains mostly fit ESM-2's **1022-residue** limit; truncate /
    sliding-window the few that don't.
- **Tooling note (revises earlier ESMC lean):** ESMC is stronger per-param for
  *zero-shot inference*, but for a *fine-tuning*-heavy workflow ESM-2's ecosystem
  maturity (LoRA recipes, HF integration) wins. Fine-tune on **ESM-2 650M**; keep
  ESMC as an optional *frozen* inference scorer in the ensemble.

### Leakage discipline (where projects like this quietly fail)
1. **One global cluster split** (mmseqs 30–50% id, whole clusters → folds), made
   once, read by every use (MLM adaptation, classifier, generation eval).
2. **Domain-adaptive MLM sees TRAIN clusters only.** Most-overlooked leak: if the
   backbone does continued MLM over sequences later in the classifier *test*
   fold, held-out AUPRC is optimistic. Adaptation corpus = train-only.
3. **Co-assign matched pairs.** Outgroups are close by construction; for each
   test-fold extremophile put its mesophile outgroup in test too, else the
   matched-pair contrast splits across folds and weakens the signal.
- **Label noise:** label = genome phenotype stamped on each secreted protein
  (noisy). Mitigate with `label_confidence` as sample weight + matched-mesophile
  contrast (learn the delta, not clade).

### ProteinMPNN: bias it, don't fine-tune it (at first)
- Weight fine-tuning is the weakest-justified piece: needs **structures we don't
  have** (would fold the whole secretome → predicted backbones, risks teaching
  folding artifacts), and **erodes MPNN's core competence** (geometry fidelity is
  why we use it).
- **Cheap safe alternative — sampling bias:** MPNN accepts per-position aa bias +
  sampling temperature. Inject an extremophilic composition bias from dataset
  statistics (charged/salt-bridge residues for thermo; acidic surface for halo)
  at sampling time, no retraining. ESM re-ranker does the fine-grained steering.
- So **fine-tuning is mostly an ESM story** (both uses). MPNN steering = bias +
  re-rank. MPNN weight FT = later experiment only if bias underperforms, and
  needs a folded-structure training set first.

### Order of operations
1. Global cluster split (mmseqs, pairs co-assigned).
2. Domain-adaptive continued MLM (LoRA, per-phenotype/conditional) on TRAIN-only.
3. Branch adapted backbone: MLM head → mask-fill + re-rank; + per-phenotype
   classifier heads (supervised, `label_confidence`-weighted).
4. Validate each head: classifier by cluster-held-out AUPRC + saliency recovering
   biophysical signatures; MLM by pseudo-perplexity drop + fill composition.
5. Wire into MPNN-generate → ESM-rerank → fold-verify loop.

---

## 12. Loss functions

**Three scoring objects; only two are training losses.** The generation-time
composite (Section 10) is an *inference gate*, NOT a differentiable loss — keep
it separate from the two head losses below.

### Loss 1 — domain-adaptive MLM (shared backbone)
Masked cross-entropy over masked positions, sequence-confidence weighted:

    L_MLM = -(1/|M|) Σ_{i∈M} w_seq · log p_θ(x_i | x_\M)

- BERT masking 15% (80% mask / 10% random / 10% keep).
- **w_seq = label_confidence** — cleanest genomes drive the adaptation.
- Optional forgetting-guard `+ β·KL(p_θ ‖ p_base)` toward base ESM-2. Tension:
  adaptation *wants* drift → start β=0 (esp. the generation adapter), add only if
  fills look unnatural.

### Loss 2 — per-phenotype classifier (matched-pair aware)
Combine a pointwise and a pairwise term.

Pointwise — weighted BCE (imbalance via pos_weight or focal (1-p_t)^γ):

    L_BCE = -Σ_i w_i [ y_i log σ(s_i) + (1-y_i) log(1-σ(s_i)) ],  w_i = label_confidence

Pairwise — margin ranking on each extremophile e vs its matched outgroup m:

    L_pair = Σ_{(e,m)} max(0, δ - (s_e - s_m))

Forces score(extremophile) > score(matched mesophile): the two differ mainly in
trait not clade, so this is the loss-function embodiment of the outgroup design —
pushes the classifier onto the phenotype *delta*, not clade features.

Combined:  **L_cls = L_BCE + λ · L_pair**  (pointwise = calibration, pairwise =
matched contrast; tune λ on val).

- **Calibrate** output probabilities (post-hoc temperature scaling on val) so the
  scores are trustworthy when gating generated sequences.
- **Model-select / early-stop on AUPRC** (per phenotype), not accuracy — heavy
  imbalance.

### Object 3 — generation-time composite (inference gate, not a loss)
MPNN proposes → score+gate, no backprop. Hard gates (catalytic identity +
side-chain RMSD) reject outright; then **product** of soft sub-scores
(calibrated classifier prob × ESM pseudo-likelihood × fold-integrity). Product,
not sum — any near-zero kills the design. Classifier factor = calibrated prob
from Loss 2.
