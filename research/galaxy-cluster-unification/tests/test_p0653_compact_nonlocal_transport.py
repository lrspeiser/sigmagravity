from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0653_compact_nonlocal_transport"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_conservation_repair_passes_every_field_gate():
    result = report()
    audit = result["field_audit"]
    assert audit["post_transport_compact_taper_applied"] is True
    assert audit["source_integral_fraction"] < 1e-10
    assert audit["maximum_flux_edge_fraction_of_RMS"] == 0.0
    assert audit["normalized_curl_RMS"] < 1e-10
    for name in (
        "field_curl",
        "field_source_integral",
        "edge_flux_closed",
        "transport_nontrivial",
        "compact_taper_applied",
    ):
        assert result["gate_results"][name] is True


def test_compact_primary_retains_large_root_complete_cv_gain():
    result = report()
    comparison = result["comparison"]
    assert result["CV_summary"]["CV_roots"] == 15
    assert comparison["CV_improvement_fraction_vs_lambda0"] > 0.21
    assert comparison["CV_improvement_fraction_vs_best_matched_multipole"] > 0.17
    assert comparison["candidate_CV_RMS_arcsec"] > comparison[
        "P0652_open_transport_CV_RMS_arcsec"
    ]


def test_spent_heldout_gate_alone_rejects_compact_operator():
    result = report()
    assert result["status"] == "fail"
    assert result["candidate_advanced_to_robustness"] is False
    assert sum(result["gate_results"].values()) == 13
    assert result["gate_results"]["spent_heldout_not_worse"] is False
    assert all(
        value
        for name, value in result["gate_results"].items()
        if name != "spent_heldout_not_worse"
    )
    assert result["full_refit"]["spent_heldout_worsening_fraction_vs_P0599"] > 0.15


def test_fold_and_full_roots_are_complete():
    folds = pd.read_csv(RESULTS / "fold_scores.csv")
    assert int(folds.validation_roots.sum()) == 15
    assert report()["full_refit"]["training_roots"] == 15
    assert report()["full_refit"]["spent_heldout_roots"] == 7


def test_formula_adds_no_fit_or_physical_scale():
    coverage = report()["coverage"]
    assert coverage["candidate_fields"] == 1
    assert coverage["amplitude_rows"] == 1
    assert coverage["fitted_field_amplitude_parameters"] == 0
    assert coverage["new_physical_length_constants"] == 0
    assert coverage["per_object_spatial_gravity_parameters"] == 0


def test_blindness_hashes_and_figure_are_preserved():
    result = report()
    assert result["sealed_P0633_kinematics_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False
    assert result["protocol_sha256"] == digest(
        ROOT / "configs/p0653_compact_nonlocal_transport.json"
    )
    assert result["source_sha256"] == digest(
        ROOT / "scripts/run_p0653_compact_nonlocal_transport.py"
    )
    assert (RESULTS / "compact_nonlocal_transport.png").stat().st_size > 20000
