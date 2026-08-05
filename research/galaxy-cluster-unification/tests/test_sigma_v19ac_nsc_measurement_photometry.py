from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from download_sigma_v19ac_nsc_measurement_photometry import (
    build_query,
    make_batches,
    parse_payload,
)


def test_batches_preserve_sorted_inputs_without_loss() -> None:
    values = [f"179969_{index}" for index in range(53)]
    batches = make_batches(values, 25)
    assert [len(batch) for batch in batches] == [25, 25, 3]
    assert [value for batch in batches for value in batch] == values


def test_query_is_exact_and_deterministic() -> None:
    query = build_query(["179969_1", "179969_2"])
    assert "FROM nsc_dr2.meas AS m JOIN nsc_dr2.exposure AS e" in query
    assert "m.exposure=e.exposure" in query
    assert "WHERE m.objectid IN ('179969_1','179969_2')" in query
    assert query.endswith("ORDER BY m.objectid,m.filter,m.mjd,m.measid")
    with pytest.raises(ValueError):
        build_query(["unsafe'id"])


def test_payload_schema_must_match_exactly() -> None:
    columns = ["objectid", "measid"]
    rows = parse_payload(b"objectid,measid\n179969_1,m1\n", columns)
    assert rows == [{"objectid": "179969_1", "measid": "m1"}]
    with pytest.raises(RuntimeError):
        parse_payload(b"measid,objectid\nm1,179969_1\n", columns)
