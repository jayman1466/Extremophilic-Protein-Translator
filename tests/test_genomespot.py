"""Unit tests for eptrans.genomespot parsing (offline)."""
import textwrap

import pytest

from eptrans.genomespot import (
    TARGETS, parse_predictions_tsv, parse_predictions_json, results_to_frame,
    GenomeSpotResult,
)

TSV = textwrap.dedent("""\
target\tvalue\terror\tunits\tis_novel\twarning
oxygen\ttolerant\t0.974\tprobability\tFalse\tNone
ph_max\t8.99\t1.30\tpH\tFalse\tNone
ph_min\t5.44\t0.92\tpH\tFalse\tNone
ph_optimum\t7.07\t0.90\tpH\tFalse\tNone
salinity_max\t3.11\t2.36\t% w/v NaCl\tFalse\tNone
salinity_min\t0\t1.18\t% w/v NaCl\tFalse\tmin_exceeded
salinity_optimum\t0.20\t1.93\t% w/v NaCl\tFalse\tNone
temperature_max\t31.30\t6.19\tC\tFalse\tNone
temperature_min\t5.64\t6.32\tC\tFalse\tNone
temperature_optimum\t22.95\t6.48\tC\tTrue\tNone
""")

JSON = (
    '{"temperature_optimum": {"value": 27.4, "error": 5.7, "is_novel": false, '
    '"warning": null, "units": "C"}, '
    '"salinity_min": {"value": 0, "error": 1.1, "is_novel": false, '
    '"warning": "min_exceeded", "units": "% w/v NaCl"}, '
    '"oxygen": {"value": "tolerant", "error": 0.95, "is_novel": false, '
    '"warning": null, "units": "probability"}}'
)


@pytest.fixture
def tsv_file(tmp_path):
    p = tmp_path / "GCA_000172155.1.predictions.tsv"
    p.write_text(TSV)
    return p


@pytest.fixture
def json_file(tmp_path):
    p = tmp_path / "GCA_000172155.1.predictions.json"
    p.write_text(JSON)
    return p


def test_parse_tsv_basic(tsv_file):
    res = parse_predictions_tsv(tsv_file)
    assert res.genome == "GCA_000172155.1"
    assert res.ok is True
    assert res.values["temperature_optimum"] == pytest.approx(22.95)
    assert res.values["oxygen"] == "tolerant"      # string preserved
    assert res.errors["oxygen"] == pytest.approx(0.974)


def test_tsv_novelty_and_warning(tsv_file):
    res = parse_predictions_tsv(tsv_file)
    assert res.is_novel["temperature_optimum"] is True
    assert res.is_novel["ph_max"] is False
    assert res.warnings["salinity_min"] == "min_exceeded"
    assert res.warnings["ph_max"] is None


def test_benign_zero_salinity_not_suspect(tsv_file):
    row = parse_predictions_tsv(tsv_file).to_row()
    # salinity_min=0 with min_exceeded is benign -> not suspect
    assert row["salinity_min__suspect"] is False
    assert row["salinity_min__warning"] == "min_exceeded"


def test_row_has_all_targets(tsv_file):
    row = parse_predictions_tsv(tsv_file).to_row()
    for t in TARGETS:
        assert t in row
        assert f"{t}__error" in row
        assert f"{t}__is_novel" in row
        assert f"{t}__warning" in row
        assert f"{t}__suspect" in row


def test_parse_json(json_file):
    res = parse_predictions_json(json_file)
    assert res.values["temperature_optimum"] == pytest.approx(27.4)
    assert res.warnings["salinity_min"] == "min_exceeded"
    assert res.values["oxygen"] == "tolerant"


def test_results_to_frame(tsv_file):
    r1 = parse_predictions_tsv(tsv_file)
    r2 = GenomeSpotResult("failed_genome", {}, {}, {}, {}, ok=False, error_message="boom")
    df = results_to_frame([r1, r2])
    assert len(df) == 2
    assert df.loc[df["genome"] == "failed_genome", "genomespot_ok"].iloc[0] == False
    assert "temperature_optimum" in df.columns
