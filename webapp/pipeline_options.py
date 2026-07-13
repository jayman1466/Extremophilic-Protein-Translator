"""Catalog of selectable pipeline components, grouped by role.

Data-driven so new models/databases are added by appending a dict entry — the
form and the results page read this catalog, nothing is hard-coded in templates.
Each SECTION has: key, label, help, multi (bool), and options [(value, label,
default_selected, enabled)]. `enabled=False` renders a disabled "coming soon"
choice so the UI advertises the extension point without offering a broken path.
"""

PHENOTYPES = [
    ("acidophile", "Acidophile (low pH)"),
    ("alkaliphile", "Alkaliphile (high pH)"),
    ("halophile", "Halophile (high salinity)"),
    ("thermophile", "Thermophile (\u226550 \u00b0C)"),
    ("hyperthermophile", "Hyperthermophile (\u226580 \u00b0C)"),
]

# Each section: (key, label, help, multi, [(value, label, default, enabled), ...])
SECTIONS = [
    ("mlm", "Generator \u2014 Masked Language Model",
     "Proposes substitutions on the mutable positions. The extremophilic adapter "
     "is the LoRA fine-tune from this project.", False,
     [("esm2_3b_extremo", "ESM-2 3B + extremophilic adapter", True, True),
      ("esm2_3b_base", "ESM-2 3B (base, no adapter)", False, False)]),

    ("fold", "Structure Prediction (folding)",
     "Folds both wild-type and designs. Folding both with the SAME method makes "
     "the active-site RMSD reflect the sequence change, not cross-method bias.", False,
     [("esmfold", "ESMFold", True, True),
      ("colabfold_wt", "ColabFold/AF2 for wild-type (ESMFold for designs)", False, False)]),

    ("msa", "MSA / Conservation",
     "Builds the alignment for sequence-weighted conservation scoring "
     "(directs where the generator is allowed to mutate).", False,
     [("mmseqs_uniref30", "MMseqs2 \u2014 UniRef30", True, True),
      ("mmseqs_colabfold_envdb", "MMseqs2 \u2014 ColabFold envDB (deeper)", False, False)]),

    ("active_site", "Active-Site Annotation",
     "Identifies catalytic / functional residues to freeze. Sequence- and "
     "annotation-based sources; combined across all selected.", True,
     [("mcsa", "M-CSA (curated catalytic residues)", True, True),
      ("interpro", "InterPro", True, True),
      ("pfam", "Pfam", True, True),
      ("swissprot", "Swiss-Prot (ACT_SITE/BINDING features)", True, True)]),

    ("foldseek", "Structural Homology (Foldseek)",
     "Structural search \u2014 transfers active-site geometry from homologs and "
     "scores fold-compatibility of designs.", True,
     [("pdb100", "PDB100", True, True),
      ("alphafold", "AlphaFold DB", True, True)]),

    ("gate", "Structural Gate",
     "Rejects designs whose backbone the inverse-folding model finds implausible; "
     "survivors are refolded and gated on catalytic-atom RMSD.", False,
     [("ligandmpnn", "LigandMPNN (cofactor-aware)", True, True),
      ("proteinmpnn", "ProteinMPNN", False, True)]),

    ("scoring", "Scoring",
     "Ranks designs by the per-phenotype classifier (this project's Stage-2 heads).",
     False,
     [("clf_per_phenotype", "Per-phenotype classifiers (ESM-2 3B adapter)", True, True)]),
]


def default_selection():
    """Return {section_key: value or [values]} of the enabled defaults."""
    out = {}
    for key, _label, _help, multi, opts in SECTIONS:
        defaults = [v for v, _l, d, en in opts if d and en]
        out[key] = defaults if multi else (defaults[0] if defaults else None)
    return out


def validate_selection(selection: dict) -> list[str]:
    """Return a list of error strings for an invalid selection (empty = OK)."""
    errs = []
    valid = {k: {v for v, *_ in opts} for k, _l, _h, _m, opts in SECTIONS}
    enabled = {k: {v for v, _l, _d, en in opts if en} for k, _l, _h, _m, opts in SECTIONS}
    for key, _label, _help, multi, _opts in SECTIONS:
        got = selection.get(key)
        if multi:
            got = got or []
            if not got:
                errs.append(f"{key}: select at least one option")
            for g in got:
                if g not in valid[key]:
                    errs.append(f"{key}: unknown option {g!r}")
                elif g not in enabled[key]:
                    errs.append(f"{key}: {g!r} is not yet available")
        else:
            if got is None:
                errs.append(f"{key}: a selection is required")
            elif got not in valid[key]:
                errs.append(f"{key}: unknown option {got!r}")
            elif got not in enabled[key]:
                errs.append(f"{key}: {got!r} is not yet available")
    return errs
