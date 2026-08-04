"""GTDB r232 metadata indexing and on-disk accessors.

Parses the GTDB bacterial (bac120) and archaeal (ar53) metadata TSVs, filters
to species representatives, extracts environment- and QC-relevant fields, and
provides accessors that map a genome accession to its files on biotite:

  * per-genome proteome  : gtdb/protein_faa_reps/{archaea,bacteria}/<PREFIX>_<acc>_protein.faa.gz
  * genome nucleotide    : resolved via work/genome_index.tsv (bare acc -> abs path)
  * combined proteome    : work/gtdb_reps.faa               (>{GENOME}~{PROTID})
  * per-protein coords   : work/protein_coords.tsv.gz

Genome id conventions
---------------------
GTDB genome ids keep a source prefix: ``GB_`` (GenBank / GCA) or ``RS_``
(RefSeq / GCF), e.g. ``RS_GCF_000005845.2``. The *bare* accession drops the
prefix: ``GCF_000005845.2``. ``genome_index.tsv`` is keyed on the bare form.
"""
from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import load_config

# Columns pulled from the 113-col metadata TSVs. Verified present in r232.
METADATA_FIELDS = [
    "accession",
    "gtdb_taxonomy",
    "ncbi_taxonomy",
    "ncbi_isolation_source",
    "ncbi_organism_name",
    "ncbi_strain_identifiers",
    "ncbi_country",
    "checkm2_completeness",
    "checkm2_contamination",
    "gtdb_representative",
    "gtdb_genome_representative",
]

GTDB_RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]
_RANK_PREFIX = {"d__": "domain", "p__": "phylum", "c__": "class", "o__": "order",
                "f__": "family", "g__": "genus", "s__": "species"}
_PREFIX_RE = re.compile(r"^(GB_|RS_|CU_)")


# --------------------------------------------------------------------------
# Accession helpers
# --------------------------------------------------------------------------
def bare_accession(genome_id: str) -> str:
    """Strip the GTDB source prefix: 'RS_GCF_000005845.2' -> 'GCF_000005845.2'.

    Non-string input (e.g. a NaN from an unmatched outgroup cell) returns "".
    """
    if not isinstance(genome_id, str):
        return ""
    return _PREFIX_RE.sub("", genome_id)


def source_prefix(genome_id: str) -> str | None:
    """Return 'GB_' or 'RS_' if present, else None."""
    m = _PREFIX_RE.match(genome_id)
    return m.group(1) if m else None


def accession_root(accession: str) -> tuple[str, str]:
    """Split a bare accession into (db, numeric): 'GCF_000005845.2' -> ('GCF','000005845')."""
    bare = bare_accession(accession)
    db, rest = bare.split("_", 1)
    numeric = rest.split(".", 1)[0]
    return db, numeric


# --------------------------------------------------------------------------
# Taxonomy parsing
# --------------------------------------------------------------------------
def parse_gtdb_taxonomy(tax_string: str) -> dict[str, str]:
    """Parse 'd__Bacteria;p__...;s__...' into a {rank: name} dict.

    Empty ranks (e.g. 's__') map to ''. Non-string / missing input -> all ''.
    """
    out = {r: "" for r in GTDB_RANKS}
    if not isinstance(tax_string, str):
        return out
    for token in tax_string.split(";"):
        token = token.strip()
        pref = token[:3]
        if pref in _RANK_PREFIX:
            out[_RANK_PREFIX[pref]] = token[3:].strip()
    return out


def expand_taxonomy(df: pd.DataFrame, column: str = "gtdb_taxonomy") -> pd.DataFrame:
    """Add per-rank columns (domain..species) parsed from a taxonomy string column."""
    parsed = df[column].apply(parse_gtdb_taxonomy).apply(pd.Series)
    return pd.concat([df, parsed[GTDB_RANKS]], axis=1)


# --------------------------------------------------------------------------
# Metadata loading
# --------------------------------------------------------------------------
def _read_metadata_tsv(path: str | Path, fields: Iterable[str]) -> pd.DataFrame:
    """Read selected columns from a (gzipped) GTDB metadata TSV."""
    fields = list(fields)
    # Read header to know which requested fields exist (robust to release drift).
    with gzip.open(path, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
    present = [f for f in fields if f in header]
    missing = [f for f in fields if f not in header]
    if missing:
        # Not fatal: keep going with what's present, but make it visible.
        print(f"[gtdb] WARNING: fields absent from {Path(path).name}: {missing}")
    df = pd.read_csv(
        path, sep="\t", usecols=present, dtype=str,
        compression="gzip", na_values=["none", "None", "NA", ""], keep_default_na=True,
    )
    return df


def load_representatives(
    bac_path: str | Path | None = None,
    arc_path: str | Path | None = None,
    fields: Iterable[str] | None = None,
    apply_qc: bool = False,
) -> pd.DataFrame:
    """Load GTDB metadata, filter to species representatives, add domain + taxonomy.

    Parameters
    ----------
    bac_path, arc_path : metadata TSV paths; default to config `biotite.metadata`.
    fields : metadata columns to keep; default `METADATA_FIELDS`.
    apply_qc : if True, drop genomes failing config `qc` completeness/contamination.

    Returns a DataFrame with one row per representative genome, numeric QC
    columns coerced to float, a `bare_accession` column, and expanded taxonomy.
    """
    cfg = load_config()
    bac_path = bac_path or cfg.get_path("biotite.metadata.bacteria")
    arc_path = arc_path or cfg.get_path("biotite.metadata.archaea")
    fields = list(fields) if fields else list(METADATA_FIELDS)

    frames = []
    for dom, path in [("Bacteria", bac_path), ("Archaea", arc_path)]:
        d = _read_metadata_tsv(path, fields)
        d["domain_file"] = dom
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    # Filter to representatives.
    flag_col = cfg.get_path("conventions.representative_flag_col", "gtdb_representative")
    flag_val = cfg.get_path("conventions.representative_flag_value", "t")
    df = df[df[flag_col] == flag_val].copy()

    # Numeric QC.
    for c in ["checkm2_completeness", "checkm2_contamination"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["bare_accession"] = df["accession"].map(bare_accession)
    df = expand_taxonomy(df, "gtdb_taxonomy")

    if apply_qc:
        min_comp = cfg.get_path("qc.min_completeness", 0.0)
        max_cont = cfg.get_path("qc.max_contamination", 100.0)
        before = len(df)
        df = df[
            (df["checkm2_completeness"].fillna(0) >= min_comp)
            & (df["checkm2_contamination"].fillna(100) <= max_cont)
        ].copy()
        print(f"[gtdb] QC: kept {len(df)}/{before} reps "
              f"(completeness>={min_comp}, contamination<={max_cont})")

    df = df.reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# On-disk file accessors
# --------------------------------------------------------------------------
@dataclass
class GenomePaths:
    """Resolved on-disk locations for one genome."""
    genome_id: str          # with prefix, e.g. RS_GCF_000005845.2
    bare: str               # GCF_000005845.2
    proteome_faa: Path | None
    genome_fna: Path | None


class GenomeIndex:
    """Accessor over work/genome_index.tsv (bare_acc -> absolute .fna.gz path)."""

    def __init__(self, index_path: str | Path | None = None):
        cfg = load_config()
        self.index_path = Path(index_path or cfg.get_path("biotite.genome_index"))
        self._map: dict[str, str] | None = None
        self.protein_reps_root = Path(cfg.get_path("biotite.protein_faa_reps"))

    def _load(self) -> dict[str, str]:
        if self._map is None:
            m: dict[str, str] = {}
            with open(self.index_path) as fh:
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 2:
                        m[parts[0]] = parts[1]
            self._map = m
        return self._map

    def genome_fna(self, accession: str) -> Path | None:
        """Absolute path to <acc>_genomic.fna.gz (keyed on bare accession)."""
        bare = bare_accession(accession)
        p = self._load().get(bare)
        return Path(p) if p else None

    def proteome_faa(self, genome_id: str, domain: str | None = None) -> Path | None:
        """Absolute path to <PREFIX>_<acc>_protein.faa.gz.

        The file name retains the genome_id prefix. `domain` ('Archaea'/'Bacteria')
        selects the subdir; if None, both are tried.
        """
        fname = f"{genome_id}_protein.faa.gz"
        subdirs = []
        if domain:
            subdirs = ["archaea" if domain.lower().startswith("a") else "bacteria"]
        else:
            subdirs = ["archaea", "bacteria"]
        for sub in subdirs:
            cand = self.protein_reps_root / sub / fname
            if cand.exists():
                return cand
        # Fall back to the first candidate path even if not present (for reporting).
        return self.protein_reps_root / subdirs[0] / fname

    def resolve(self, genome_id: str, domain: str | None = None) -> GenomePaths:
        bare = bare_accession(genome_id)
        faa = self.proteome_faa(genome_id, domain)
        fna = self.genome_fna(bare)
        return GenomePaths(
            genome_id=genome_id, bare=bare,
            proteome_faa=faa if (faa and faa.exists()) else faa,
            genome_fna=fna,
        )


if __name__ == "__main__":
    import sys
    cfg = load_config()
    print(f"[gtdb] bacteria metadata: {cfg.get_path('biotite.metadata.bacteria')}")
    print(f"[gtdb] archaea  metadata: {cfg.get_path('biotite.metadata.archaea')}")
    # quick taxonomy parse self-test (works offline)
    t = "d__Archaea;p__Methanobacteriota;c__Methanobacteria;o__Methanobacteriales;f__Methanobacteriaceae;g__Methanobrevibacter;s__Methanobrevibacter smithii"
    print(parse_gtdb_taxonomy(t))
    print("bare:", bare_accession("RS_GCF_000005845.2"))
