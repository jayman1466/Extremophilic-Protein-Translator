"""Per-sample cap on extremophile selection (MAG pseudoreplication control)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eptrans.selection import select_extremophiles


def _pool(n_per_sample: int, n_samples: int, n_isolates: int = 0) -> pd.DataFrame:
    """MAGs spread over samples, each in its OWN family so the family cap never binds."""
    rows = []
    fam = 0
    for s in range(n_samples):
        for k in range(n_per_sample):
            fam += 1
            rows.append(dict(accession=f"CU_S{s}_bin{k}", final_thermophile=True,
                             final_confidence="high", family=f"f{fam}",
                             source_sample_id=f"SAMPLE{s}"))
    for i in range(n_isolates):
        fam += 1
        rows.append(dict(accession=f"GB_GCA_{i:09d}.1", final_thermophile=True,
                         final_confidence="high", family=f"f{fam}",
                         source_sample_id=np.nan))
    return pd.DataFrame(rows)


def test_cap_limits_genomes_per_sample():
    df = _pool(n_per_sample=10, n_samples=3)
    got = select_extremophiles(df, "thermophile", max_per_lineage=99, max_per_sample=2)
    assert len(got) == 6, f"expected 2 per sample x 3 samples, got {len(got)}"
    assert got.source_sample_id.value_counts().max() == 2


def test_none_is_a_noop():
    df = _pool(n_per_sample=10, n_samples=3)
    uncapped = select_extremophiles(df, "thermophile", max_per_lineage=99, max_per_sample=None)
    assert len(uncapped) == 30


def test_blank_sample_ids_are_exempt_not_pooled():
    """The load-bearing case: NaN means 'isolate, its own sample'.

    If blanks were pooled into one bucket, a cap of 2 would admit only 2 of the
    isolates -- which would silently cap ALL of GTDB, where the column is empty
    for every row.
    """
    df = _pool(n_per_sample=0, n_samples=0, n_isolates=50)
    got = select_extremophiles(df, "thermophile", max_per_lineage=99, max_per_sample=2)
    assert len(got) == 50, f"blank sample ids must be exempt; got {len(got)}"


def test_mixed_pool_caps_mags_only():
    df = _pool(n_per_sample=10, n_samples=2, n_isolates=20)
    got = select_extremophiles(df, "thermophile", max_per_lineage=99, max_per_sample=3)
    is_mag = got.source_sample_id.notna()
    assert int(is_mag.sum()) == 6, "3 per sample x 2 samples"
    assert int((~is_mag).sum()) == 20, "all isolates retained"


def test_cap_is_deterministic():
    df = _pool(n_per_sample=10, n_samples=3)
    a = select_extremophiles(df, "thermophile", max_per_lineage=99, max_per_sample=2, seed=1466)
    b = select_extremophiles(df, "thermophile", max_per_lineage=99, max_per_sample=2, seed=1466)
    assert list(a.accession) == list(b.accession)


def test_family_cap_still_applies_under_sample_cap():
    """Both caps compose: family cap binds even when the sample cap would allow more."""
    rows = [dict(accession=f"CU_S0_bin{k}", final_thermophile=True, final_confidence="high",
                 family="shared_family", source_sample_id="SAMPLE0") for k in range(10)]
    df = pd.DataFrame(rows)
    got = select_extremophiles(df, "thermophile", max_per_lineage=3, max_per_sample=99)
    assert len(got) == 3


@pytest.mark.parametrize("cap,expect", [(1, 3), (2, 6), (5, 15), (99, 30)])
def test_cap_scales(cap, expect):
    df = _pool(n_per_sample=10, n_samples=3)
    got = select_extremophiles(df, "thermophile", max_per_lineage=99, max_per_sample=cap)
    assert len(got) == expect
