"""Environmental binning.

Two independent evidence sources are combined into a final extremophile label:

1. **Metadata proxy** - keyword matching against ``ncbi_isolation_source``
   (and, more weakly, ``ncbi_organism_name``). This module.
2. **GenomeSPOT predictions** - genomic optima for temperature / pH / salinity.
   Predicted-class derivation + the combination rule are also here.

Extremophile classes: thermophile, hyperthermophile, psychrophile, acidophile,
alkaliphile, halophile. A genome can carry more than one class
(e.g. a haloalkaliphile from a soda lake).

Design notes
------------
* The keyword dictionary is intentionally explicit and auditable - every match
  retains the substring that triggered it (``*_evidence`` columns), so calls can
  be reviewed and the dictionary refined.
* Matching is done on lower-cased text with word-ish boundaries to avoid
  substring false positives ("salt" in "basalt", "acid" in "placid").
* Ambiguous sources are handled deliberately: "soda lake" -> halophile +
  alkaliphile; "cold seep" is NOT flagged psychrophile (deep-sea seeps sit at
  ambient ~2-4 C but host organisms across the thermal range).
* Organism-name signals (Thermo-, Halo-, ...) are collected separately and
  down-weighted, because genus names correlate with clade and would leak
  phylogeny into the label.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import load_config

# Canonical class names.
CLASSES = ["thermophile", "hyperthermophile", "psychrophile",
           "acidophile", "alkaliphile", "halophile"]

# --------------------------------------------------------------------------
# Isolation-source keyword dictionary.
# Each entry: class -> list of regex fragments (matched with word boundaries,
# case-insensitive). Grounded in the observed r232 isolation_source vocabulary.
# --------------------------------------------------------------------------
ISOLATION_KEYWORDS: dict[str, list[str]] = {
    "thermophile": [
        r"hydrothermal", r"hot spring", r"hotspring", r"geothermal",
        r"black smoker", r"solfatara", r"fumarole", r"thermal spring",
        r"thermal vent", r"hot water", r"thermal pool", r"volcanic",
        r"deep-sea vent", r"deep sea vent", r"sulfide chimney", r"chimney",
        r"steam vent", r"boiling", r"geyser",
    ],
    # Hyperthermophile is a stronger/narrower thermal signal from metadata;
    # these are also thermophile-positive (handled in normalization).
    "hyperthermophile": [
        r"black smoker", r"sulfide chimney", r"deep-sea hydrothermal",
        r"deep sea hydrothermal",
    ],
    "psychrophile": [
        r"permafrost", r"glacier", r"glacial", r"ice core", r"sea ice",
        r"polar", r"antarctic", r"arctic", r"cryo", r"snow", r"ice sheet",
        r"subglacial", r"cold desert", r"tundra", r"ice shelf",
        # Deep-sea cold habitats (added 2026-07-30). Below the permanent
        # thermocline the water column sits at 1-4 C worldwide, so depth is a
        # reliable proxy for cold EXCEPT where hydrothermal input overrides it
        # (see EXCLUSIONS). Terms are deliberately depth-specific: bare
        # "marine sediment" is NOT included because it matches coastal and
        # shallow samples that are not cold (it would have added 678 genomes
        # in r232, including a thermophilic enrichment culture).
        r"hadal", r"abyssal", r"abyssopelagic", r"bathypelagic",
        r"deep-sea sediment", r"deep sea sediment",
        r"deep-sea water", r"deep sea water",
        r"mariana trench", r"trench sediment",
        r"ocean crust", r"seafloor sediment",
    ],
    "acidophile": [
        r"acid mine", r"acid drainage", r"\bamd\b", r"acidic", r"acid lake",
        r"acid hot spring", r"acid spring", r"sulfuric", r"acidophilic",
        r"low ph", r"acid soil", r"bioleaching",
    ],
    "alkaliphile": [
        r"soda lake", r"alkaline lake", r"alkaline spring", r"alkaline hot",
        r"alkaline", r"caustic", r"high ph", r"natron", r"lye",
        r"serpentiniz",  # serpentinization -> high pH
    ],
    "halophile": [
        r"hypersaline", r"saltern", r"salt lake", r"salt pan", r"salt flat",
        r"brine", r"soda lake",  # soda lakes are also hypersaline
        r"solar salt", r"salt mine", r"salt marsh sediment", r"halite",
        r"saline lake", r"dead sea", r"great salt lake", r"sabkha",
        r"salt crust", r"evaporitic", r"salted",
    ],
}

# Organism-name morpheme signals (weaker; clade-correlated). Word-start-ish.
ORGANISM_KEYWORDS: dict[str, list[str]] = {
    "thermophile": [r"thermo", r"thermus", r"pyro", r"caldi", r"geothermo"],
    "hyperthermophile": [r"pyro", r"pyrococcus", r"pyrolobus", r"hyperthermus"],
    "psychrophile": [r"psychro", r"cryo", r"gelid", r"frigo", r"glacii"],
    "acidophile": [r"acidi", r"acido", r"ferroplasma", r"thermoplasma",
                   r"sulfolobus", r"picrophilus"],
    "alkaliphile": [r"alkali", r"natrono", r"natri", r"alkalibacter",
                    r"alkaliphilus"],
    "halophile": [r"halo", r"halobacter", r"haloarcula", r"salini",
                  r"salinibacter", r"halorubrum", r"natrono"],
}

# Sources that must NOT trigger a class even though a fragment appears.
# (fragment, class) pairs that are known false positives.
EXCLUSIONS: list[tuple[str, str]] = [
    (r"salt marsh(?! sediment)", "halophile"),   # tidal salt marsh: not hypersaline
    (r"basalt", "halophile"),                     # 'salt' inside basalt
    (r"cold seep", "psychrophile"),               # ambient deep-sea, not psychrophilic
    (r"cold-adapted enrichment", "psychrophile"),
    # Hydrothermal input overrides the depth-implies-cold inference: a hadal
    # vent sample is thermophilic, not psychrophilic. These guard the deep-sea
    # keywords added above.
    (r"hydrothermal", "psychrophile"),
    (r"black smoker", "psychrophile"),
    (r"chimney", "psychrophile"),
    (r"hot vent", "psychrophile"),
    (r"warm vent", "psychrophile"),
    # Culture-derived samples describe the culture, not the habitat.
    (r"thermophilic", "psychrophile"),
    (r"enrichment culture", "psychrophile"),
    # Depth qualifiers that contradict the deep-sea terms.
    (r"shallow", "psychrophile"),
    (r"coastal", "psychrophile"),
]


def _compile(patterns: list[str]) -> list[re.Pattern]:
    # \b works for alnum boundaries; fragments with spaces/hyphens are fine as-is.
    return [re.compile(rf"(?<![a-z]){p}", re.IGNORECASE) for p in patterns]


_ISO_RE = {cls: _compile(pats) for cls, pats in ISOLATION_KEYWORDS.items()}
_ORG_RE = {cls: _compile(pats) for cls, pats in ORGANISM_KEYWORDS.items()}
_EXCL_RE = [(re.compile(pat, re.IGNORECASE), cls) for pat, cls in EXCLUSIONS]


@dataclass
class MetadataFlag:
    """Metadata-derived class hits for one genome, with evidence."""
    iso_classes: set[str] = field(default_factory=set)
    iso_evidence: dict[str, str] = field(default_factory=dict)
    org_classes: set[str] = field(default_factory=set)
    org_evidence: dict[str, str] = field(default_factory=dict)


def flag_text(text: str, table: dict[str, list[re.Pattern]]) -> tuple[set[str], dict[str, str]]:
    """Return (classes, {class: matched_substring}) for one text against a regex table."""
    classes: set[str] = set()
    evidence: dict[str, str] = {}
    if not isinstance(text, str) or not text.strip():
        return classes, evidence
    low = text.lower()
    for cls, regexes in table.items():
        for rgx in regexes:
            m = rgx.search(low)
            if m:
                classes.add(cls)
                evidence[cls] = m.group(0)
                break
    return classes, evidence


def _apply_exclusions(text: str, classes: set[str], evidence: dict[str, str]) -> None:
    if not isinstance(text, str):
        return
    for rgx, cls in _EXCL_RE:
        if cls in classes and rgx.search(text.lower()):
            classes.discard(cls)
            evidence.pop(cls, None)


def flag_genome(isolation_source: str, organism_name: str = "") -> MetadataFlag:
    """Flag a single genome from its isolation source (+ optional organism name)."""
    iso_c, iso_e = flag_text(isolation_source, _ISO_RE)
    _apply_exclusions(isolation_source, iso_c, iso_e)
    org_c, org_e = flag_text(organism_name, _ORG_RE)
    # hyperthermophile implies thermophile.
    if "hyperthermophile" in iso_c:
        iso_c.add("thermophile")
    if "hyperthermophile" in org_c:
        org_c.add("thermophile")
    return MetadataFlag(iso_c, iso_e, org_c, org_e)


def flag_dataframe(
    df: pd.DataFrame,
    isolation_col: str = "ncbi_isolation_source",
    organism_col: str = "ncbi_organism_name",
) -> pd.DataFrame:
    """Add metadata-flag columns to a representatives DataFrame.

    Adds, per class: ``meta_iso_<class>`` (bool), ``meta_iso_<class>_evidence``
    (str), ``meta_org_<class>`` (bool). Also ``meta_iso_any``/``meta_org_any``
    and ``meta_iso_classes`` (semicolon-joined) convenience columns.
    """
    out = df.copy()
    org_series = out[organism_col] if organism_col in out.columns else pd.Series([""] * len(out))

    records = [flag_genome(iso, org) for iso, org in zip(out[isolation_col], org_series)]

    for cls in CLASSES:
        out[f"meta_iso_{cls}"] = [cls in r.iso_classes for r in records]
        out[f"meta_iso_{cls}_evidence"] = [r.iso_evidence.get(cls, "") for r in records]
        out[f"meta_org_{cls}"] = [cls in r.org_classes for r in records]

    out["meta_iso_classes"] = [";".join(sorted(r.iso_classes)) for r in records]
    out["meta_org_classes"] = [";".join(sorted(r.org_classes)) for r in records]
    out["meta_iso_any"] = [bool(r.iso_classes) for r in records]
    out["meta_org_any"] = [bool(r.org_classes) for r in records]
    return out


# ==========================================================================
# GenomeSPOT prediction -> class derivation, and the combination rule.
# (Implemented alongside metadata flagging; used after predictions exist.)
# ==========================================================================
def predicted_classes(
    temp_opt: float | None,
    ph_opt: float | None,
    salinity_opt: float | None,
    cfg=None,
) -> set[str]:
    """Derive extremophile classes from GenomeSPOT predicted optima + config thresholds."""
    cfg = cfg or load_config()
    th = cfg.get_path("thresholds")
    classes: set[str] = set()

    def ok(x):
        return x is not None and not (isinstance(x, float) and np.isnan(x))

    if ok(temp_opt):
        if temp_opt >= th["temperature"]["hyperthermophile_min_opt"]:
            classes.update({"thermophile", "hyperthermophile"})
        elif temp_opt >= th["temperature"]["thermophile_min_opt"]:
            classes.add("thermophile")
        elif temp_opt <= th["temperature"]["psychrophile_max_opt"]:
            classes.add("psychrophile")
    if ok(ph_opt):
        if ph_opt <= th["ph"]["acidophile_max_opt"]:
            classes.add("acidophile")
        elif ph_opt >= th["ph"]["alkaliphile_min_opt"]:
            classes.add("alkaliphile")
    if ok(salinity_opt):
        if salinity_opt >= th["salinity"]["halophile_min_opt"]:
            classes.add("halophile")
    return classes


def is_confident_mesophile(
    temp_opt, ph_opt, salinity_opt, cfg=None
) -> bool:
    """True if predicted optima all fall inside the mesophile envelope."""
    cfg = cfg or load_config()
    m = cfg.get_path("thresholds.mesophile")

    def within(x, lo, hi):
        return x is not None and not (isinstance(x, float) and np.isnan(x)) and lo <= x <= hi

    return (
        within(temp_opt, m["temp_min_opt"], m["temp_max_opt"])
        and within(ph_opt, m["ph_min_opt"], m["ph_max_opt"])
        and (salinity_opt is not None
             and not (isinstance(salinity_opt, float) and np.isnan(salinity_opt))
             and salinity_opt <= m["salinity_max_opt"])
    )


def combine_label(meta_classes: set[str], pred_classes: set[str],
                  pred_available: bool) -> tuple[str, str]:
    """Reconcile metadata + prediction into (final_class, confidence).

    final_class is ';'-joined (may be multi-class or '' / 'mesophile').
    confidence in {high, medium, low, none}:
      * high   - metadata and prediction agree on >=1 class
      * medium - prediction only (prediction available, no metadata agreement)
      * low    - metadata only, or metadata/prediction conflict
      * none   - no evidence either way
    """
    if not meta_classes and not pred_classes:
        return ("", "none")
    agree = meta_classes & pred_classes
    if agree:
        return (";".join(sorted(agree)), "high")
    if pred_available and pred_classes:
        return (";".join(sorted(pred_classes)), "medium")
    if meta_classes:
        return (";".join(sorted(meta_classes)), "low")
    return ("", "none")


if __name__ == "__main__":
    tests = [
        ("hypersaline soda lake sediment", ""),
        ("marine hydrothermal vent", ""),
        ("acid mine drainage sediment", ""),
        ("permafrost active layer soil", ""),
        ("alkaline hot spring water", ""),
        ("salt marsh", ""),          # excluded from halophile
        ("cold seep", ""),           # excluded from psychrophile
        ("soil", ""),                # nothing
        ("deep-sea hydrothermal sulfide chimney", ""),
    ]
    for iso, org in tests:
        f = flag_genome(iso, org)
        print(f"{iso!r:45s} -> iso={sorted(f.iso_classes)} evidence={f.iso_evidence}")
