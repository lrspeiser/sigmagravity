from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19ad_fixed_aperture_color_commissioning.json"
REPORT = ROOT / "results" / "sigma_v19ad_fixed_aperture_color_commissioning" / "report.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v19ad_records_fail_closed_before_fit() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["config_sha256"] == sha256(CONFIG)
    assert report["implementation"]["runner_sha256"] == sha256(
        ROOT / report["implementation"]["runner"]
    )
    assert report["status"] == "failed_before_fit_incomplete_primary_photometry"
    assert report["gates"]["all_commissioning_gates_pass"] is False
    assert report["gates"]["all_primary_rows_have_griz"] is False
    assert report["primary"]["fit_or_validation_scoring_performed"] is False
    assert report["ambiguous_likelihood_application_authorized"] is False


def test_v19ad_missing_measurement_evidence_is_exact() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["primary"]["missing_rows"] == [
        {
            "object_id": "57",
            "nsc_id": "179969_8549",
            "split": "validation",
            "missing_filters": ["r", "i"],
            "accepted_measurements": {"g": 1, "r": 0, "i": 0, "z": 4},
        }
    ]
    assert report["sensitivity"] == [
        {"aperture_diameter_arcsec": 2, "complete_griz": False},
        {"aperture_diameter_arcsec": 8, "complete_griz": False},
    ]


def test_v19ad_aggregated_output_is_complete_and_hashed() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    path = ROOT / report["outputs"]["aggregated_sample"]
    assert sha256(path) == report["outputs"]["aggregated_sample_sha256"]
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 45
    assert {int(row["aperture_diameter_arcsec"]) for row in rows} == {2, 4, 8}


def test_v19ad_claim_boundary_remains_closed() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["ambiguous_candidate_photometry_scored"] is False
    assert report["counterpart_selected"] is False
    assert report["stellar_mass_inferred"] is False
    assert report["mass_current_constructed"] is False
    assert report["lensing_or_halo_payload_opened"] is False
    assert report["gravity_formula_or_parameter_changed"] is False
