from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from download_sigma_v19ae_fors1_commissioning_frames import (
    QUERY_COLUMNS,
    build_queries,
    parse_metadata,
    safe_dataset_filename,
)


def test_queries_freeze_archive_roles_and_ordering() -> None:
    queries = build_queries()
    assert set(queries) == {"science", "bias", "flat"}
    assert "dp_cat='SCIENCE'" in queries["science"]
    assert "filter_path IN ('B_BESS','R_BESS','I_BESS')" in queries["science"]
    assert "ob_name='ALL-BIAS_fl1x1_10'" in queries["bias"]
    assert "tpl_id='FORS1_img_cal_Twili'" in queries["flat"]
    assert all(query.endswith("ORDER BY dp_id") for query in queries.values())


def test_metadata_requires_exact_schema_unique_sorted_fors_ids() -> None:
    header = ",".join(QUERY_COLUMNS)
    blank = [""] * len(QUERY_COLUMNS)
    first = dict(zip(QUERY_COLUMNS, blank, strict=True))
    second = dict(zip(QUERY_COLUMNS, blank, strict=True))
    first["dp_id"] = "OFORS.1998-01-01T00:00:00.000"
    second["dp_id"] = "OFORS.1998-01-02T00:00:00.000"

    def line(row: dict[str, str]) -> str:
        return ",".join(row[column] for column in QUERY_COLUMNS)

    payload = (header + "\n" + line(first) + "\n" + line(second) + "\n").encode()
    assert [row["dp_id"] for row in parse_metadata(payload)] == [
        first["dp_id"],
        second["dp_id"],
    ]
    with pytest.raises(RuntimeError):
        parse_metadata((header + "\n" + line(second) + "\n" + line(first) + "\n").encode())
    with pytest.raises(RuntimeError):
        parse_metadata((header + "\n" + line(first) + "\n" + line(first) + "\n").encode())
    with pytest.raises(RuntimeError):
        parse_metadata(b"dp_id\nOFORS.1\n")


def test_dataset_filename_is_windows_portable_and_identity_is_preserved_elsewhere() -> None:
    assert (
        safe_dataset_filename("OFORS.1998-12-14T06:05:25.298")
        == "OFORS.1998-12-14T06_05_25.298"
    )
    assert safe_dataset_filename(":::") == "_"
