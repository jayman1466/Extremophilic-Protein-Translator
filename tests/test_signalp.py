"""Unit tests for eptrans.signalp (parse + secreted extraction).

Fixtures reproduce the exact SignalP 6.0 output format documented in the
installed source (make_output_files.py).
"""
from eptrans.signalp import (
    parse_prediction_results, extract_secreted, summarize, build_signalp_command,
    SP_CLASSES,
)

RESULTS = (
    "# SignalP-6.0\tOrganism: Other\tTimestamp: 20240101000000\n"
    "# ID\tPrediction\tOTHER\tSP(Sec/SPI)\tLIPO(Sec/SPII)\tTAT(Tat/SPI)\tTATLIPO(Tat/SPII)\tPILIN(Sec/SPIII)\tCS Position\n"
    "prot_sp\tSP\t0.010000\t0.980000\t0.005000\t0.003000\t0.001000\t0.001000\tCS pos: 24-25. Pr: 0.9012\n"
    "prot_lipo\tLIPO\t0.020000\t0.010000\t0.950000\t0.010000\t0.005000\t0.005000\tCS pos: 18-19. Pr: 0.8500\n"
    "prot_tat\tTAT\t0.030000\t0.020000\t0.010000\t0.930000\t0.005000\t0.005000\tCS pos: 30-31. Pr: 0.7700\n"
    "prot_other\tOTHER\t0.990000\t0.005000\t0.002000\t0.001000\t0.001000\t0.001000\t\n"
)

FASTA = (
    ">prot_sp some description\n"
    "MKKTLLASLLASGVLAAQAAMADSTQEVKLPPVEVKQ\n"     # 37 aa; CS after 24 -> mature = res 25..37 (13 aa)
    ">prot_lipo\n"
    "MRLLLSVLTTLCLSACSSKPVEEKSGYQ\n"              # 28 aa; CS after 18 -> mature 10 aa
    ">prot_tat\n"
    "MSRRQFLKQSAAALGVTALGTSAFAADTVKAQ\n"          # 32 aa; CS after 30 -> mature 2 aa
    ">prot_other\n"
    "MSKGEELFTGVVPILVELDGDVNGHKF\n"               # cytoplasmic, no SP
)


def _write(tmp_path):
    r = tmp_path / "prediction_results.txt"
    f = tmp_path / "proteome.faa"
    r.write_text(RESULTS)
    f.write_text(FASTA)
    return str(r), str(f)


def test_parse_classes_and_cs(tmp_path):
    rp, _ = _write(tmp_path)
    preds = parse_prediction_results(rp)
    assert len(preds) == 4
    by = {p.protein_id: p for p in preds}
    assert by["prot_sp"].prediction == "SP"
    assert by["prot_sp"].cs_after == 24
    assert abs(by["prot_sp"].cs_prob - 0.9012) < 1e-6
    assert by["prot_sp"].probs["SP(Sec/SPI)"] == 0.98
    assert by["prot_other"].prediction == "OTHER"
    assert by["prot_other"].cs_after is None


def test_is_secreted_flag(tmp_path):
    rp, _ = _write(tmp_path)
    preds = parse_prediction_results(rp)
    secreted = [p.protein_id for p in preds if p.is_secreted]
    assert set(secreted) == {"prot_sp", "prot_lipo", "prot_tat"}


def test_extract_full_precursor(tmp_path):
    rp, fp = _write(tmp_path)
    preds = parse_prediction_results(rp)
    got = extract_secreted(preds, fp, mature=False)
    ids = {g[0] for g in got}
    assert ids == {"prot_sp", "prot_lipo", "prot_tat"}
    seq_by = {g[0]: g[2] for g in got}
    assert len(seq_by["prot_sp"]) == 37  # full precursor


def test_extract_mature_chain(tmp_path):
    rp, fp = _write(tmp_path)
    preds = parse_prediction_results(rp)
    got = extract_secreted(preds, fp, mature=True)
    seq_by = {g[0]: g[2] for g in got}
    # CS after 24 -> mature = residues 25..37 = 13 aa, starts with the residue after cleavage
    full = "MKKTLLASLLASGVLAAQAAMADSTQEVKLPPVEVKQ"
    assert seq_by["prot_sp"] == full[24:]
    assert len(seq_by["prot_sp"]) == 13


def test_extract_class_filter(tmp_path):
    rp, fp = _write(tmp_path)
    preds = parse_prediction_results(rp)
    got = extract_secreted(preds, fp, classes=["LIPO"])
    assert {g[0] for g in got} == {"prot_lipo"}


def test_summarize(tmp_path):
    rp, _ = _write(tmp_path)
    preds = parse_prediction_results(rp)
    s = summarize(preds)
    assert s["n_proteins"] == 4
    assert s["n_secreted"] == 3
    assert s["by_class"]["SP"] == 1
    assert s["by_class"]["OTHER"] == 1
    assert abs(s["secreted_fraction"] - 0.75) < 1e-9


def test_build_command():
    cmd = build_signalp_command("in.faa", "outdir", organism="other", mode="fast")
    assert cmd[0] == "signalp6"
    assert "--fastafile" in cmd and "in.faa" in cmd
    assert "--organism" in cmd and "other" in cmd
    assert "--mode" in cmd and "fast" in cmd
