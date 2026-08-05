from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19ab_bessel_nsc_transform_commissioning.json"
REPORT = ROOT / "results" / "sigma_v19ab_bessel_nsc_transform_commissioning" / "report.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_v19ab_provenance_and_frozen_failure() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["config_sha256"] == sha256(CONFIG)
    assert report["implementation"]["runner_sha256"] == sha256(
        ROOT / report["implementation"]["runner"]
    )
    for name in ("commissioning_sample", "validation_predictions", "validation_retrieval"):
        path = ROOT / report["outputs"][name]
        assert sha256(path) == report["outputs"][f"{name}_sha256"]

    assert report["gates"]["all_commissioning_gates_pass"] is False
    assert report["gates"]["color_only_top1_retrieval"] is False
    assert report["gates"]["color_only_mean_reciprocal_rank"] is False
    assert report["gates"]["full_offset_top1_retrieval"] is True
    assert report["gates"]["full_offset_mean_reciprocal_rank"] is True
    assert report["likelihood_application_authorized"] is False


def test_v19ab_validation_metrics_are_exactly_preserved() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    validation = report["validation"]
    assert validation["retrieval"]["full_offset_top1"] == 5
    assert validation["retrieval"]["full_offset_mean_reciprocal_rank"] == pytest.approx(1.0)
    assert validation["retrieval"]["color_only_top1"] == 2
    assert validation["retrieval"]["color_only_mean_reciprocal_rank"] == pytest.approx(0.6)
    assert validation["retrieval"]["full_offset_true_ranks"] == [1, 1, 1, 1, 1]
    assert validation["retrieval"]["color_only_true_ranks"] == [1, 1, 3, 3, 3]
    assert validation["median_absolute_error_mag"] == pytest.approx(
        {
            "g_minus_B": 0.12500248760620536,
            "r_minus_R": 0.05379194133588727,
            "i_minus_I": 0.08212201214371939,
            "z_minus_I": 0.08851888872652292,
        }
    )


def test_v19ab_split_and_retrieval_scope_stay_closed() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    sample = read_csv(ROOT / report["outputs"]["commissioning_sample"])
    retrieval = read_csv(ROOT / report["outputs"]["validation_retrieval"])
    assert len(sample) == 15
    assert [row["object_id"] for row in sample if row["split"] == "development"] == [
        "29",
        "78",
        "14",
        "07",
        "06",
        "37",
        "22",
        "16",
        "23",
        "24",
    ]
    assert [row["object_id"] for row in sample if row["split"] == "validation"] == [
        "26",
        "57",
        "66",
        "71",
        "21",
    ]
    assert len(retrieval) == 25
    assert {row["member_object_id"] for row in retrieval} == {"26", "57", "66", "71", "21"}
    assert {row["candidate_object_id"] for row in retrieval} == {"26", "57", "66", "71", "21"}


def test_v19ab_claim_boundary_remains_measurement_only() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["ambiguous_candidate_photometry_scored"] is False
    assert report["counterpart_selected"] is False
    assert report["stellar_mass_inferred"] is False
    assert report["mass_current_constructed"] is False
    assert report["lensing_or_halo_payload_opened"] is False
    assert report["gravity_formula_or_parameter_changed"] is False
