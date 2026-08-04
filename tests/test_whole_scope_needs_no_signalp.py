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
