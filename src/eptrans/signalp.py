"""SignalP 6.0 wrapper + secreted-protein extraction.

SignalP 6.0 predicts signal peptides and their type for each protein. We treat a
protein as **secreted / cell-surface exposed** when its predicted class is any
signal-peptide type (i.e. anything other than ``OTHER``):

    SP       Sec/SPI    - standard secretory signal peptide
    LIPO     Sec/SPII   - lipoprotein signal peptide
    TAT      Tat/SPI    - twin-arginine translocation signal
    TATLIPO  Tat/SPII   - Tat lipoprotein signal
    PILIN    Sec/SPIII  - pilin/pseudopilin signal

Output format (from the installed SignalP 6.0 source, ``make_output_files.py``):

``prediction_results.txt`` is tab-separated with two ``#``-prefixed header lines,
then one row per protein. For ``--organism other`` the columns are::

    ID  Prediction  OTHER  SP(Sec/SPI)  LIPO(Sec/SPII)  TAT(Tat/SPI)  \
        TATLIPO(Tat/SPII)  PILIN(Sec/SPIII)  CS Position

``CS Position`` is formatted ``CS pos: <k>-<k+1>. Pr: <p>`` where cleavage is
between residues k and k+1 (1-based); empty when no cleavage site. The mature
(secreted) chain is the sequence from position k+1 to the C-terminus.

CLI (installed as a pyenv shim on the target host)::

    signalp6 --fastafile <faa> --output_dir <dir> --format txt \
             --organism other --mode fast
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# Signal-peptide classes SignalP 6.0 can assign (prokaryote model).
SP_CLASSES = ["SP", "LIPO", "TAT", "TATLIPO", "PILIN"]
ALL_CLASSES = ["OTHER"] + SP_CLASSES

# Probability column names in prediction_results.txt (organism=other).
_PROB_COLS = ["OTHER", "SP(Sec/SPI)", "LIPO(Sec/SPII)", "TAT(Tat/SPI)",
              "TATLIPO(Tat/SPII)", "PILIN(Sec/SPIII)"]

_CS_RE = re.compile(r"CS pos:\s*(\d+)-(\d+)\.\s*Pr:\s*([\d.]+)")


@dataclass
class SignalPrediction:
    protein_id: str
    prediction: str                 # OTHER / SP / LIPO / TAT / TATLIPO / PILIN
    probs: dict = field(default_factory=dict)
    cs_after: int | None = None     # cleavage after this 1-based residue (mature starts at cs_after+1)
    cs_prob: float | None = None

    @property
    def is_secreted(self) -> bool:
        return self.prediction != "OTHER"


def parse_prediction_results(path: str | os.PathLike) -> list[SignalPrediction]:
    """Parse a SignalP 6.0 ``prediction_results.txt`` into SignalPrediction rows."""
    preds: list[SignalPrediction] = []
    header_cols: list[str] | None = None
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                # second comment line carries the column names
                cols = [c.strip() for c in line.lstrip("#").strip().split("\t")]
                if cols and cols[0] == "ID":
                    header_cols = cols
                continue
            fields = line.split("\t")
            pid = fields[0]
            prediction = fields[1] if len(fields) > 1 else "OTHER"
            # probability columns sit between Prediction and CS Position
            probs: dict[str, float] = {}
            # map by header if available, else positional
            if header_cols:
                for cname in _PROB_COLS:
                    if cname in header_cols:
                        j = header_cols.index(cname)
                        if j < len(fields):
                            try:
                                probs[cname] = float(fields[j])
                            except ValueError:
                                pass
            cs_after = cs_prob = None
            m = _CS_RE.search(line)
            if m:
                cs_after = int(m.group(1))
                cs_prob = float(m.group(3))
            preds.append(SignalPrediction(pid, prediction, probs, cs_after, cs_prob))
    return preds


def build_signalp_command(
    fasta: str,
    output_dir: str,
    organism: str = "other",
    mode: str = "fast",
    fmt: str = "txt",
    binary: str = "signalp6",
    extra: list[str] | None = None,
) -> list[str]:
    """Construct the SignalP 6.0 argv."""
    cmd = [binary, "--fastafile", str(fasta), "--output_dir", str(output_dir),
           "--format", fmt, "--organism", organism, "--mode", mode]
    if extra:
        cmd += list(extra)
    return cmd


def run_signalp(
    fasta: str,
    output_dir: str,
    organism: str = "other",
    mode: str = "fast",
    binary: str = "signalp6",
    timeout: int | None = None,
) -> list[SignalPrediction]:
    """Run SignalP 6.0 locally and parse the results.

    Note: on the biotite target SignalP is a pyenv shim and requires
    ``PYENV_ROOT``/``PATH`` export first; for cluster runs prefer submitting a
    job script that does that export (see scripts/05_run_signalp.py). This
    function is for environments where ``signalp6`` is directly runnable.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cmd = build_signalp_command(fasta, output_dir, organism, mode, "txt", binary)
    subprocess.run(cmd, check=True, timeout=timeout,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return parse_prediction_results(Path(output_dir) / "prediction_results.txt")


def iter_fasta(path: str | os.PathLike):
    """Minimal FASTA iterator -> (header_id, full_header, sequence)."""
    import gzip
    opener = gzip.open if str(path).endswith(".gz") else open
    hid = full = None
    seq: list[str] = []
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if hid is not None:
                    yield hid, full, "".join(seq)
                full = line[1:]
                hid = full.split()[0] if full else ""
                seq = []
            else:
                seq.append(line.strip())
    if hid is not None:
        yield hid, full, "".join(seq)


def extract_secreted(
    predictions: list[SignalPrediction],
    fasta_path: str,
    mature: bool = False,
    classes: list[str] | None = None,
) -> list[tuple[str, str, str]]:
    """Extract secreted-protein sequences.

    Args:
        predictions: parsed SignalP predictions.
        fasta_path: the proteome FASTA that was scored (for sequences).
        mature: if True, return the mature chain (after the cleavage site);
            otherwise return the full precursor sequence.
        classes: signal-peptide classes to keep (default: all SP_CLASSES).

    Returns:
        list of (protein_id, prediction_class, sequence).
    """
    keep = set(classes or SP_CLASSES)
    pred_by_id = {p.protein_id: p for p in predictions if p.prediction in keep}
    out: list[tuple[str, str, str]] = []
    for hid, _full, seq in iter_fasta(fasta_path):
        p = pred_by_id.get(hid)
        if p is None:
            continue
        s = seq
        if mature and p.cs_after and 0 < p.cs_after < len(seq):
            s = seq[p.cs_after:]
        out.append((hid, p.prediction, s))
    return out


def summarize(predictions: list[SignalPrediction]) -> dict:
    """Class counts + secreted fraction."""
    from collections import Counter
    c = Counter(p.prediction for p in predictions)
    n = len(predictions)
    n_secreted = sum(v for k, v in c.items() if k != "OTHER")
    return {
        "n_proteins": n,
        "n_secreted": n_secreted,
        "secreted_fraction": round(n_secreted / n, 4) if n else 0.0,
        "by_class": {k: int(c.get(k, 0)) for k in ALL_CLASSES},
    }
