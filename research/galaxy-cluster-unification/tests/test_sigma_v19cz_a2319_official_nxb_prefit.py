from __future__ import annotations

import hashlib
import json
from pathlib import Path

import fit_sigma_v19cy_a2319_spectra as fitter
import run_sigma_v19cz_a2319_official_nxb_prefit as retry

ROOT = Path(__file__).resolve().parents[1]


def test_official_second_stage_preserves_delivered_shape_parameters() -> None:
    model = (
        ROOT
        / "data/raw/sigma_v19cy_a2319_response_support/rsl_nxb_model_v1.mo"
    )
    _, specs = fitter.parse_nxb_model(model.read_text(encoding="utf-8"))
    assert fitter.nxb_free_parameter_indices(specs, 1) == [
        3,
        7,
        14,
        20,
        23,
        29,
        35,
        41,
        47,
        50,
        53,
        56,
    ]
    assert fitter.nxb_free_parameter_indices(
        specs, 1, "official_preserve_delivered_shapes"
    ) == [2, 3, 7, 14, 19, 20, 23, 29, 35, 41, 46, 47, 49, 50, 52, 53, 55, 56]


def test_v19cz_is_nxb_only_and_keeps_holdouts_sealed() -> None:
    config = json.loads(retry.CONFIG.read_text(encoding="utf-8"))
    assert config["fit_policy"]["source_spectra_loaded"] is False
    assert config["fit_policy"]["source_energy_columns_read"] is False
    assert config["fit_policy"]["source_fit_authorized"] is False
    assert config["authorization"]["access_a3667_validation"] is False
    assert config["authorization"]["access_a754_holdout"] is False
    assert config["authorization"]["open_lensing_halo_or_gravity_targets"] is False
    assert len(config["inputs"]) == 10
    assert len({row["region"] for row in config["inputs"]}) == 7


def test_prefit_summary_counts_only_free_bound_hits() -> None:
    summary = retry.summarize_prefit(
        {
            "region": "a",
            "statistic": 120.0,
            "dof": 100,
            "converged": True,
            "hard_bound_hits": [2, 5, 7],
            "metadata": {"nxb_free_parameter_indices": [2, 3, 7]},
            "source_spectra_loaded": False,
            "source_energy_distribution_used": False,
            "statistic_by_spectrum": {"1": 120.0},
        }
    )
    assert summary["reduced_chi_square"] == 1.2
    assert summary["free_parameter_hard_bound_hits"] == [2, 7]
    assert summary["all_numeric_hard_bound_hits"] == [2, 5, 7]


def test_terminal_v19cz_failure_and_artifact_integrity() -> None:
    report_path = ROOT / "results/sigma_v19cz_a2319_official_nxb_prefit/report.json"
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == (
        "6ae50188cf5f1a947b39187ded407c0413f1214c14c630f08b2fefd8faca2868"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "failed_official_nxb_prefit_gate"
    assert report["terminal_gate_passed"] is False
    assert report["a2319_source_fit_authorized"] is False
    assert report["validation_or_holdout_accessed"] is False
    assert report["lensing_halo_or_gravity_target_accessed"] is False
    assert report["artifact_count"] == 21
    assert report["artifact_bytes"] == 745_514
    observed = {
        row["region"]: round(row["reduced_chi_square"], 6)
        for row in report["prefits"]
    }
    assert observed == {
        "a": 1.138390,
        "b": 14.605876,
        "d": 11.670159,
        "b_prime": 10.970364,
        "c_prime": 13.470839,
        "d_prime": 17.961392,
        "e_prime": 11.034141,
    }
    for artifact in report["artifacts"]:
        path = ROOT / artifact["path"]
        assert path.stat().st_size == artifact["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
