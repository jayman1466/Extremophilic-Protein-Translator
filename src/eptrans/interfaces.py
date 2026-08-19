"""Protected-set (active-site + partner-interface) resolution for the
generative pipeline.

Two entry points map to the two --additional-constraints modes:

  parse_explicit_constraints(spec, seq_len)
      Parse a comma-separated list of residue positions/ranges (e.g.
      "A:45,A:88-92,120") into a single flat set of 1-based positions on the
      DESIGN CHAIN. Chain prefixes ("A:") are stripped -- there is only one
      design chain in a generation job. Ranges are inclusive.

  resolve_interfaces_from_complex(cif_path, design_chain, phrases,
                                  contact_cutoff=4.5, cb_cutoff=8.0)
      Resolve a natural-language phrase list (e.g. ["tetramer interfaces",
      "protein-bRNA interface", "protein-donor interface",
      "protein-target interface"]) to a DICT of named protected sets on the
      design chain, using two auditable stages:

        1. phrase -> partner ENTITIES via keyword matching against the PDB's
           entity descriptions retrieved from the mmCIF header (LLM optional;
           deterministic fallback used here so the pipeline has no runtime LLM
           dependency for the reproducible IS621 run). One phrase can resolve
           to multiple partner chains (e.g. "tetramer interfaces" -> the other
           three protein chains, kept as SEPARATE sub-interfaces so a A:B
           failure is distinct from A:D).

        2. entity -> RESIDUES by geometry: design-chain residues with any
           heavy atom within `contact_cutoff` of any partner heavy atom
           (default 4.5 A). A CB<=`cb_cutoff` fallback (default 8 A) is
           computed too and returned for auditability but NOT used for the
           frozen set -- the frozen set is the 4.5-A contact ring only.

The reference-frame rule ("look at the face, don't co-fold") is enforced at
the CALLER: identities are locked from the true complex (this module),
but RMSD drift is measured against the WT-sequence ESMFold monomer (apo
self-consistent baseline). This module does not touch conformation, only
identity + membership.

Output shape (returned by resolve_interfaces_from_complex):
    {
      "iface_A_B":    {"positions": [11,12,...], "partner_chains": ["B"],
                       "phrase": "tetramer interfaces", "n_contacts": 30,
                       "cb_positions": [...]},
      "iface_A_D":    ...,
      "iface_A_bRNA": {"positions":[...], "partner_chains":["E","F"],
                       "phrase":"protein-bRNA interface", ...},
      ...
    }
Empty sets are dropped -- a face with 0 contacts is not a protected face.

Positions are 1-based (matching --transfer-json convention and the
1-based `active_site` published in candidates.json).
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Iterable

# -- keyword table for phrase->entity resolution (deterministic, no LLM dep). --
# Each key is a phrase-fragment; the value is a list of role tags that a partner
# entity's `pdbx_description` must contain (case-insensitive) to match.
# "tetramer" / "homotetramer" is special-cased: it resolves to OTHER protein chains
# of the same entity as the design chain.
_PARTNER_KEYWORDS: dict[str, list[str]] = {
    "tetramer":         [],  # sentinel; handled specially
    "homotetramer":     [],
    "dimer":            [],  # same handling
    "homodimer":        [],
    "trimer":            [],
    "brna":             ["bridge rna", "bridge_rna"],
    "bridge rna":       ["bridge rna", "bridge_rna"],
    "bridge_rna":       ["bridge rna", "bridge_rna"],
    "rna":              ["rna"],
    "donor":            ["donor"],
    "target":           ["target"],
    "dna":              ["dna"],
    "substrate":        ["substrate"],
    "cofactor":         ["cofactor"],
    "guide":            ["guide"],
}


def parse_explicit_constraints(spec: str, seq_len: int) -> list[int]:
    """Parse "A:45,A:88-92,120" -> sorted unique 1-based positions.

    Positions outside [1, seq_len] are silently dropped (with a printed WARN).
    Chain prefixes are stripped: there is one design chain per job.
    """
    if not spec:
        return []
    out: set[int] = set()
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        # strip chain prefix "A:" or "chain A:" etc.
        if ":" in tok:
            tok = tok.split(":", 1)[1].strip()
        m = re.match(r"^(\d+)\s*-\s*(\d+)$", tok)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                a, b = b, a
            for p in range(a, b + 1):
                if 1 <= p <= seq_len:
                    out.add(p)
                else:
                    print(f"[interfaces] WARN dropped out-of-range position {p} (seq_len={seq_len})", flush=True)
        else:
            try:
                p = int(tok)
            except ValueError:
                print(f"[interfaces] WARN unparseable token {tok!r}", flush=True)
                continue
            if 1 <= p <= seq_len:
                out.add(p)
            else:
                print(f"[interfaces] WARN dropped out-of-range position {p} (seq_len={seq_len})", flush=True)
    return sorted(out)


def _load_biopython():
    try:
        from Bio.PDB import MMCIFParser
        from Bio.PDB.NeighborSearch import NeighborSearch
        return MMCIFParser, NeighborSearch
    except ImportError as e:
        raise RuntimeError(
            "resolve_interfaces_from_complex requires biopython "
            "(pip install biopython)"
        ) from e


def _read_entity_map_from_cif(cif_path: str) -> dict[str, dict]:
    """Return {auth_chain_id: {"description": str, "polymer_type": str}}
    for every polymer chain in the mmCIF."""
    # A minimal auth-chain -> entity description scan.
    # We use biopython for structure parsing, but the entity->chain map lives
    # in _entity_poly + _entity + _struct_asym -- easier to parse the header
    # loops directly for the two fields we need.
    text = Path(cif_path).read_text()
    # collect entity id -> description
    ent_desc: dict[str, str] = {}
    # try _entity.pdbx_description (with a _entity loop)
    # crude but sufficient for PDB-issued cif files.
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln == "loop_":
            # collect subsequent _xxx tags then rows
            i += 1
            tags = []
            while i < len(lines) and lines[i].strip().startswith("_"):
                tags.append(lines[i].strip())
                i += 1
            # tags identifies which loop this is
            if any(t == "_entity.id" for t in tags) and any(
                t == "_entity.pdbx_description" for t in tags
            ):
                id_ix = tags.index("_entity.id")
                desc_ix = tags.index("_entity.pdbx_description")
                # rows until the next line starting with _ or #
                while i < len(lines):
                    row = lines[i].strip()
                    if not row or row.startswith("#") or row.startswith("_") or row == "loop_":
                        break
                    # naive whitespace split; description can be quoted
                    row_tokens = _cif_split_row(row)
                    if len(row_tokens) > max(id_ix, desc_ix):
                        ent_desc[row_tokens[id_ix]] = row_tokens[desc_ix].strip("'\"")
                    i += 1
                continue
            elif any(t == "_struct_asym.id" for t in tags) and any(
                t == "_struct_asym.entity_id" for t in tags
            ):
                # struct_asym maps label_asym_id -> entity_id (not what we want; auth_asym differs)
                # we'll fall back to biopython for auth mapping via label
                pass
            elif any(t == "_atom_site.auth_asym_id" for t in tags):
                # skip atom_site (huge)
                while i < len(lines):
                    row = lines[i].strip()
                    if row.startswith("_") or row == "loop_" or row.startswith("#"):
                        break
                    i += 1
                continue
        i += 1
    # Now build auth_asym -> entity_id using biopython (it exposes entity_id per residue via
    # the header). Fallback: parse _atom_site directly for the first-row mapping.
    # Simplest: read _atom_site fields to build the mapping from auth_asym_id -> label_entity_id
    auth_to_entity: dict[str, str] = {}
    i = 0
    in_atom = False
    tags = []
    while i < len(lines):
        ln = lines[i].strip()
        if ln == "loop_":
            j = i + 1
            local_tags = []
            while j < len(lines) and lines[j].strip().startswith("_"):
                local_tags.append(lines[j].strip())
                j += 1
            if any(t.startswith("_atom_site.") for t in local_tags):
                # find indices
                try:
                    auth_ix = local_tags.index("_atom_site.auth_asym_id")
                    ent_ix = local_tags.index("_atom_site.label_entity_id")
                except ValueError:
                    i = j
                    continue
                k = j
                while k < len(lines):
                    row = lines[k].strip()
                    if not row or row.startswith("#") or row.startswith("_") or row == "loop_":
                        break
                    toks = row.split()
                    if len(toks) > max(auth_ix, ent_ix):
                        ac = toks[auth_ix]
                        if ac not in auth_to_entity:
                            auth_to_entity[ac] = toks[ent_ix]
                    k += 1
                i = k
                continue
        i += 1
    # Combine into result
    out: dict[str, dict] = {}
    for auth, ent in auth_to_entity.items():
        out[auth] = {"description": ent_desc.get(ent, ""), "entity_id": ent}
    return out


def _cif_split_row(row: str) -> list[str]:
    """Tokenize a CIF loop row, respecting single/double quotes."""
    out = []
    i = 0
    n = len(row)
    while i < n:
        c = row[i]
        if c == " " or c == "\t":
            i += 1
            continue
        if c in ("'", '"'):
            end = row.find(c, i + 1)
            if end == -1:
                out.append(row[i + 1:])
                return out
            out.append(row[i + 1:end])
            i = end + 1
        else:
            j = i
            while j < n and row[j] not in " \t":
                j += 1
            out.append(row[i:j])
            i = j
    return out


def _phrase_matches_entity(phrase: str, entity_desc: str, is_design_entity: bool) -> bool:
    """Return True if `phrase` (lowercased) picks out this entity.

    Homomeric-interface phrases (tetramer/dimer/trimer) only match OTHER copies
    of the design entity.
    """
    p = phrase.lower()
    d = entity_desc.lower()
    # homomeric hits: partner must be same-entity, but this is checked at chain level
    if any(w in p for w in ("tetramer", "dimer", "trimer", "homotetramer", "homodimer", "hexamer")):
        return is_design_entity
    # keyword match against description
    for kw_key, aliases in _PARTNER_KEYWORDS.items():
        if kw_key in p:
            for alias in aliases:
                if alias in d:
                    return True
            # also match the bare keyword against the description
            if kw_key in d:
                return True
    # last-resort: any word from phrase (>=3 chars) present in description
    for w in re.findall(r"[a-z]+", p):
        if len(w) < 3:
            continue
        if w in d:
            return True
    return False


def _face_label(design_chain: str, partner_chain: str, partner_desc: str) -> str:
    """Stable per-face label: iface_{design}_{partner} for tetramer subs,
    iface_{design}_{descTag} for functional partners."""
    tag = partner_desc.lower()
    for k in ("bridge rna", "bridge_rna"):
        if k in tag:
            return f"iface_{design_chain}_bRNA_{partner_chain}"
    if "target" in tag and "dna" in tag:
        return f"iface_{design_chain}_target_{partner_chain}"
    if "donor" in tag and "dna" in tag:
        return f"iface_{design_chain}_donor_{partner_chain}"
    if "rna" in tag:
        return f"iface_{design_chain}_RNA_{partner_chain}"
    if "dna" in tag:
        return f"iface_{design_chain}_DNA_{partner_chain}"
    # protein-protein: use chain letters
    return f"iface_{design_chain}_{partner_chain}"


def resolve_interfaces_from_complex(
    cif_path: str,
    design_chain: str,
    phrases: Iterable[str],
    contact_cutoff: float = 4.5,
    cb_cutoff: float = 8.0,
    keep_empty: bool = False,
) -> dict[str, dict]:
    """Resolve natural-language partner phrases to per-face protected sets.

    Returns a dict {face_label: {"positions":[1-based ints],
                                 "cb_positions":[1-based ints],
                                 "partner_chains":[str],
                                 "phrase": str,
                                 "n_contacts": int}}.
    Empty faces are dropped unless keep_empty=True.
    """
    MMCIFParser, NeighborSearch = _load_biopython()
    parser = MMCIFParser(QUIET=True)
    struct = parser.get_structure("in", cif_path)
    model = list(struct.get_models())[0]

    entity_map = _read_entity_map_from_cif(cif_path)
    if design_chain not in entity_map:
        raise ValueError(f"design chain {design_chain!r} not found in {cif_path}; "
                         f"present chains: {sorted(entity_map)}")
    design_entity = entity_map[design_chain]["entity_id"]

    # collect chain objects
    chains = {c.id: c for c in model.get_chains()}
    if design_chain not in chains:
        raise ValueError(f"design chain {design_chain!r} not in structure model")

    # heavy atoms of the design chain
    def heavy(chain):
        out = []
        for res in chain:
            if res.id[0] != " ":
                continue
            for atom in res:
                if atom.element == "H":
                    continue
                out.append(atom)
        return out

    design_atoms = heavy(chains[design_chain])
    design_residues = {res.id[1]: res for res in chains[design_chain] if res.id[0] == " "}

    results: dict[str, dict] = {}
    # for each phrase, decide which partner chains it selects
    for phrase in phrases:
        phrase = phrase.strip()
        if not phrase:
            continue
        partner_chains: list[str] = []
        for auth, info in entity_map.items():
            if auth == design_chain:
                continue
            is_design_entity = info["entity_id"] == design_entity
            if _phrase_matches_entity(phrase, info["description"], is_design_entity):
                partner_chains.append(auth)
        if not partner_chains:
            print(f"[interfaces] WARN phrase {phrase!r} matched no partner chains", flush=True)
            continue

        # per-partner-chain, compute contacts to keep sub-face attribution
        for pc in partner_chains:
            if pc not in chains:
                continue
            partner_atoms = heavy(chains[pc])
            if not partner_atoms:
                continue
            ns = NeighborSearch(partner_atoms)
            contact_res: set[int] = set()
            for atom in design_atoms:
                if ns.search(atom.coord, contact_cutoff, level="A"):
                    contact_res.add(atom.get_parent().id[1])
            cb_res: set[int] = set()
            for res in chains[design_chain]:
                if res.id[0] != " ":
                    continue
                if "CB" in res:
                    cb = res["CB"]
                elif "CA" in res:
                    cb = res["CA"]
                else:
                    continue
                if ns.search(cb.coord, cb_cutoff, level="A"):
                    cb_res.add(res.id[1])
            label = _face_label(design_chain, pc, entity_map[pc]["description"])
            if (not contact_res) and not keep_empty:
                print(f"[interfaces] skipping empty face {label} (phrase {phrase!r})", flush=True)
                continue
            # merge if the same label already exists (multiple phrases picking same chain)
            if label in results:
                results[label]["positions"] = sorted(set(results[label]["positions"]) | contact_res)
                results[label]["cb_positions"] = sorted(set(results[label]["cb_positions"]) | cb_res)
                results[label]["phrase"] += f"; {phrase}"
                results[label]["n_contacts"] = len(results[label]["positions"])
            else:
                results[label] = {
                    "positions": sorted(contact_res),
                    "cb_positions": sorted(cb_res),
                    "partner_chains": [pc],
                    "partner_description": entity_map[pc]["description"],
                    "phrase": phrase,
                    "n_contacts": len(contact_res),
                }
    return results
