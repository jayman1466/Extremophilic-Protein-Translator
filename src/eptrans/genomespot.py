"""GenomeSPOT wrapper + prediction parsing.

GenomeSPOT (Barnum et al. 2024, bioRxiv 2024.03.22.586313) predicts oxygen
tolerance and optimum/min/max temperature, pH, and salinity from a genome's
DNA + protein FASTA using amino-acid composition features.

CLI (from the repo):
    python -m genome_spot.genome_spot --models <models_dir> \
        --contigs <genome.fna[.gz]> --proteins <proteins.faa[.gz]> \
        --output-prefix <prefix>

Outputs ``<prefix>.predictions.tsv`` with columns:
    target, value, error, units, is_novel, warning
and (optionally) ``<prefix>.features.json``. A ``.predictions.json`` with the
same content as a nested dict is also produced by the Python API.

Targets (10): temperature_{optimum,min,max}, ph_{optimum,min,max},
salinity_{optimum,min,max}, oxygen.

Interpretation notes (from the GenomeSPOT README):
* value/error: error is a local RMSE for continuous traits; for oxygen it's the
  classification probability (recommend decisions only when p>0.75).
* is_novel: True if genome features are more unusual than 98% of training data
  (~4% of GTDB genomes for O2/temp; ~10-15% for salinity/pH). Treat as low-conf.
* warning: e.g. "min_exceeded"/"max_exceeded" - predicted value hit the sensical
  range boundary and was clamped. Suspect UNLESS it's salinity min/optimum at 0
  (common and benign).
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TARGETS = [
    "temperature_optimum", "temperature_min", "temperature_max",
    "ph_optimum", "ph_min", "ph_max",
    "salinity_optimum", "salinity_min", "salinity_max",
    "oxygen",
]

# Benign warning: salinity min/optimum clamped to 0 is common and not suspect.
_BENIGN_ZERO_WARN = {"salinity_optimum", "salinity_min"}


@dataclass
class GenomeSpotResult:
    """Flat per-genome GenomeSPOT result with values, errors, flags."""
    genome: str
    values: dict[str, float | str]
    errors: dict[str, float]
    is_novel: dict[str, bool]
    warnings: dict[str, str | None]
    ok: bool = True
    error_message: str = ""

    def to_row(self) -> dict:
        """Flatten to a single dict row (one genome), suitable for a DataFrame."""
        row: dict = {"genome": self.genome, "genomespot_ok": self.ok}
        if self.error_message:
            row["genomespot_error"] = self.error_message
        for t in TARGETS:
            row[f"{t}"] = self.values.get(t)
            row[f"{t}__error"] = self.errors.get(t)
            row[f"{t}__is_novel"] = self.is_novel.get(t)
            row[f"{t}__warning"] = self.warnings.get(t)
        # Convenience: suspect flag per trait (warning present & not benign-zero).
        for t in TARGETS:
            w = self.warnings.get(t)
            suspect = bool(w) and not (t in _BENIGN_ZERO_WARN and self.values.get(t) in (0, 0.0, "0"))
            row[f"{t}__suspect"] = suspect
        return row


def _coerce_value(target: str, raw: str | float):
    if target == "oxygen":
        return raw  # "tolerant" / "not tolerant"
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _coerce_bool(raw) -> bool:
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"true", "1", "yes"}


def _coerce_warning(raw) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    return None if s in {"", "None", "nan", "null"} else s


def parse_predictions_tsv(path: str | Path, genome: str | None = None) -> GenomeSpotResult:
    """Parse a GenomeSPOT ``<prefix>.predictions.tsv`` file into a result."""
    path = Path(path)
    genome = genome or path.name.replace(".predictions.tsv", "")
    df = pd.read_csv(path, sep="\t")
    values, errors, novel, warns = {}, {}, {}, {}
    for _, r in df.iterrows():
        t = r["target"]
        values[t] = _coerce_value(t, r["value"])
        errors[t] = _coerce_value("_", r["error"])  # always numeric
        novel[t] = _coerce_bool(r["is_novel"])
        warns[t] = _coerce_warning(r["warning"])
    return GenomeSpotResult(genome, values, errors, novel, warns)


def parse_predictions_json(path: str | Path, genome: str | None = None) -> GenomeSpotResult:
    """Parse a GenomeSPOT ``.predictions.json`` (nested dict) into a result."""
    path = Path(path)
    genome = genome or path.name.replace(".predictions.json", "")
    data = json.load(open(path))
    values, errors, novel, warns = {}, {}, {}, {}
    for t, d in data.items():
        values[t] = _coerce_value(t, d.get("value"))
        errors[t] = _coerce_value("_", d.get("error"))
        novel[t] = _coerce_bool(d.get("is_novel"))
        warns[t] = _coerce_warning(d.get("warning"))
    return GenomeSpotResult(genome, values, errors, novel, warns)


def run_genomespot(
    contigs: str | Path,
    proteins: str | Path,
    models_dir: str | Path,
    output_prefix: str | Path,
    genome: str | None = None,
    python_exe: str | None = None,
    save_features: bool = False,
    timeout: int = 600,
) -> GenomeSpotResult:
    """Run GenomeSPOT on one genome and parse the result.

    Invokes ``python -m genome_spot.genome_spot``. Requires the ``genome_spot``
    package importable by ``python_exe`` (default: current interpreter) and the
    correct scikit-learn (==1.2.2) in that environment.
    """
    python_exe = python_exe or sys.executable
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    genome = genome or output_prefix.name

    cmd = [
        python_exe, "-m", "genome_spot.genome_spot",
        "--models", str(models_dir),
        "--contigs", str(contigs),
        "--proteins", str(proteins),
        "--output-prefix", str(output_prefix),
    ]
    if save_features:
        cmd.append("--save-genome-features")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return _empty_result(genome, ok=False, msg=f"timeout after {timeout}s")
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        return _empty_result(genome, ok=False, msg="; ".join(tail) or f"exit {proc.returncode}")

    tsv = Path(f"{output_prefix}.predictions.tsv")
    if not tsv.exists():
        return _empty_result(genome, ok=False, msg="no predictions.tsv produced")
    res = parse_predictions_tsv(tsv, genome=genome)
    return res


def _empty_result(genome: str, ok: bool, msg: str) -> GenomeSpotResult:
    return GenomeSpotResult(
        genome=genome, values={}, errors={}, is_novel={}, warnings={},
        ok=ok, error_message=msg,
    )


def results_to_frame(results: list[GenomeSpotResult]) -> pd.DataFrame:
    """Collect a list of results into a tidy per-genome DataFrame."""
    return pd.DataFrame([r.to_row() for r in results])


def load_predictions_dir(directory: str | Path) -> pd.DataFrame:
    """Parse every ``*.predictions.tsv`` in a directory into one DataFrame."""
    directory = Path(directory)
    results = [parse_predictions_tsv(p) for p in sorted(directory.glob("*.predictions.tsv"))]
    return results_to_frame(results)


if __name__ == "__main__":
    # Parse the test-genome JSON shipped with GenomeSPOT if present.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="a .predictions.tsv or .predictions.json")
    args = ap.parse_args()
    p = Path(args.path)
    res = (parse_predictions_json(p) if p.suffix == ".json" else parse_predictions_tsv(p))
    row = res.to_row()
    for k, v in row.items():
        print(f"{k:32s} {v}")
