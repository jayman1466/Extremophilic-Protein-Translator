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
