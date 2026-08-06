"""Whole-proteome scope must not depend on SignalP coverage.

Regression test for a silent 90.6% data-loss bug: the FASTA emission loop walked
only the genomes present in the SignalP prediction table, so a whole-scope genome
with no predictions contributed zero sequences. Whole-proteome scope wants every
protein regardless of secretion, so SignalP coverage is irrelevant to it.

The emptiness guard did not catch this: enough genomes overlapped to keep the file
non-empty (685 of 7,320 on the real run), so the corpus was silently truncated.
"""
import gzip, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_whole_scope_genome_without_predictions_still_emitted(tmp_path):
    pr = tmp_path / "prot"
    pr.mkdir()
    for g, n in [("GB_GCA_HASPRED.1", 2), ("GB_GCA_NOPRED.1", 3)]:
        with gzip.open(pr / f"{g}_protein.faa.gz", "wt") as fh:
            for i in range(n):
                fh.write(f">p{i} d\nMKV{'A' * (i + 3)}\n")

    ch = tmp_path / "chunk_0"
    ch.mkdir()
    (ch / "prediction_results.txt").write_text(
        "# SignalP-6.0\n# ID\tPrediction\tOTHER\tSP(Sec/SPI)\tCS Position\n"
        "GB_GCA_HASPRED.1~p0\tSP(Sec/SPI)\t0.01\t0.99\tCS pos: 22-23. Pr: 0.88\n"
        "GB_GCA_HASPRED.1~p1\tOTHER\t0.97\t0.03\t\n")

    ws = tmp_path / "ws.txt"
    ws.write_text("GB_GCA_HASPRED.1\nGB_GCA_NOPRED.1\n")

    whole = tmp_path / "whole.faa"
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts/05_aggregate_signalp.py"),
         "--pred-dirs", str(tmp_path / "chunk_*"),
         "--proteome-root", str(pr),
         "--whole-scope-accessions", str(ws),
         "--faa-secreted", str(tmp_path / "sec.faa"),
         "--faa-whole", str(whole),
         "--out", str(tmp_path / "out.tsv")],
        capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr

    heads = [l[1:] for l in whole.read_text().splitlines() if l.startswith(">")]
    nopred = [h for h in heads if h.startswith("GB_GCA_NOPRED.1~")]
    # all 3 proteins of the un-predicted genome must be present
    assert len(nopred) == 3, f"whole-scope genome without SignalP lost proteins: {heads}"
    assert len(heads) == 5, heads

    # the SECRETED fasta must NOT gain anything from it (it keys off is_secreted)
    sec = [l[1:] for l in (tmp_path / "sec.faa").read_text().splitlines()
           if l.startswith(">")]
    assert sec == ["GB_GCA_HASPRED.1~p0"], sec


def test_whole_scope_tsv_covers_same_proteins_as_fasta(tmp_path):
    """INV-EMIT-A: the TSV must not be a strict subset of the whole-proteome FASTA.

    The FASTA-side fix above made whole-scope genomes contribute every protein to
    wholeproteome.faa (and hence to the id40 clustering). The TSV kept being built
    from SignalP predictions ALONE, so stage 06 -- which reads the TSV, not the
    FASTA -- still saw only the secreted rows. Measured on the 2026-08-05 run:
    GB_GCA_002167555.2 had 51 TSV rows against 1,045 FASTA records and 1,315 clu40
    lines, so its 994 cytoplasmic proteins could never enter the corpus and all 19
    psychrophile pair-extremophiles stayed effectively secreted-scope despite the
    INV-SCOPE-D fix. A protein that is clustered but absent from the TSV is
    unreachable, which is why this is asserted rather than merely logged.
    """
    import csv

    pr = tmp_path / "prot"
    pr.mkdir()
    # HASPRED: scanned by SignalP (1 secreted, 1 not) but has a THIRD protein that
    # SignalP never saw -- the exact shape of the production defect.
    with gzip.open(pr / "GB_GCA_HASPRED.1_protein.faa.gz", "wt") as fh:
        for i in range(3):
            fh.write(f">p{i} d\nMKV{'A' * (i + 3)}\n")
    with gzip.open(pr / "GB_GCA_NOPRED.1_protein.faa.gz", "wt") as fh:
        for i in range(2):
            fh.write(f">q{i} d\nMKW{'C' * (i + 3)}\n")

    ch = tmp_path / "chunk_0"
    ch.mkdir()
    (ch / "prediction_results.txt").write_text(
        "# SignalP-6.0\n# ID\tPrediction\tOTHER\tSP(Sec/SPI)\tCS Position\n"
        "GB_GCA_HASPRED.1~p0\tSP(Sec/SPI)\t0.01\t0.99\tCS pos: 22-23. Pr: 0.88\n"
        "GB_GCA_HASPRED.1~p1\tOTHER\t0.97\t0.03\t\n")

    ws = tmp_path / "ws.txt"
    ws.write_text("GB_GCA_HASPRED.1\nGB_GCA_NOPRED.1\n")

    whole = tmp_path / "whole.faa"
    out = tmp_path / "out.tsv"
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts/05_aggregate_signalp.py"),
         "--pred-dirs", str(tmp_path / "chunk_*"),
         "--proteome-root", str(pr),
         "--whole-scope-accessions", str(ws),
         "--faa-secreted", str(tmp_path / "sec.faa"),
         "--faa-whole", str(whole),
         "--out", str(out)],
        capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr

    rows = list(csv.DictReader(out.open(), delimiter="\t"))
    by_id = {x["tagged_id"]: x for x in rows}
    fasta_ids = {l[1:] for l in whole.read_text().splitlines() if l.startswith(">")}

    # THE INVARIANT: nothing in the whole-proteome FASTA may be missing from the TSV
    assert fasta_ids <= set(by_id), f"clustered but unreachable: {fasta_ids - set(by_id)}"

    # the protein SignalP never saw is present, and marked non-secreted
    assert "GB_GCA_HASPRED.1~p2" in by_id
    assert by_id["GB_GCA_HASPRED.1~p2"]["is_secreted"] in ("False", "false")
    # real SignalP calls must WIN the dedupe, not be overwritten by the OTHER stub
    assert by_id["GB_GCA_HASPRED.1~p0"]["prediction"].startswith("SP")
    assert by_id["GB_GCA_HASPRED.1~p0"]["is_secreted"] in ("True", "true")
    # a genome with no predictions at all contributes all its proteins
    assert {"GB_GCA_NOPRED.1~q0", "GB_GCA_NOPRED.1~q1"} <= set(by_id)
