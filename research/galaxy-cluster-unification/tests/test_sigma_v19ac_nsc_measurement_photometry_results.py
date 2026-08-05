from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19ac_nsc_measurement_photometry.json"
REPORT = ROOT / "results" / "sigma_v19ac_nsc_measurement_photometry" / "provenance.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_v19ac_provenance_and_exact_counts() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["config_sha256"] == sha256(CONFIG)
    assert report["implementation"]["runner_sha256"] == sha256(
        ROOT / report["implementation"]["runner"]
    )
    assert report["requested_object_ids"] == 226
    assert report["returned_object_ids"] == 226
    assert report["measurement_rows"] == 8199
    assert report["instruments"] == {"c4d": 8006, "k4m": 59, "ksb": 134}
    assert report["filters"] == {"Y": 867, "g": 1468, "i": 1687, "r": 2855, "z": 1322}
    assert report["gates"]["all_acquisition_gates_pass"] is True


def test_v19ac_every_batch_and_combined_payload_hash() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert len(report["records"]) == 10
    requested: list[str] = []
    returned: list[str] = []
    for record in report["records"]:
        assert record["http_status"] == 200
        assert sha256(ROOT / record["csv_path"]) == record["csv_sha256"]
        assert sha256(ROOT / record["query_path"]) == record["query_sha256"]
        assert sha256(ROOT / record["form_path"]) == record["form_sha256"]
        requested.extend(record["requested_object_ids"])
        returned.extend(record["returned_object_ids"])
    assert len(requested) == len(set(requested)) == 226
    assert set(returned) == set(requested)
    combined = ROOT / report["outputs"]["combined_measurements"]
    assert sha256(combined) == report["outputs"]["combined_measurements_sha256"]
    assert len(csv_rows(combined)) == 8199


def test_v19ac_rows_are_lossless_and_instrument_counts_recompute() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    rows = csv_rows(ROOT / report["outputs"]["combined_measurements"])
    assert Counter(row["instrument"] for row in rows) == Counter(report["instruments"])
    assert Counter(row["filter"] for row in rows) == Counter(report["filters"])
    assert len({row["objectid"] for row in rows}) == 226
    assert len({row["measid"] for row in rows}) == 8199


def test_v19ac_claim_boundary_remains_acquisition_only() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["aperture_or_exposure_selected"] is False
    assert report["photometric_identity_scored"] is False
    assert report["counterpart_selected"] is False
    assert report["stellar_mass_inferred"] is False
    assert report["mass_current_constructed"] is False
    assert report["lensing_or_halo_payload_opened"] is False
    assert report["gravity_formula_or_parameter_changed"] is False
