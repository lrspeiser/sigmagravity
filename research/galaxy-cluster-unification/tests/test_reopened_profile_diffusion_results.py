import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_profile_diffusion_protocol_was_frozen_as_complete_factorials():
    protocol = json.loads(
        (ROOT / "configs/reopened_hybrid_profile_diffusion_protocol.json").read_text()
    )
    rows = sum(
        len(specification["values"])
        for specification in protocol["sensitivity_families"].values()
    )
    assert protocol["status"] == "frozen_before_reopened_hybrid_scores"
    assert rows == 38
    assert protocol["raw_lensing"]["gravity_parameters_fit_to_raw_lensing"] == 0
    assert (
        protocol["baseline_variant_name"]
        == "local_control:radial_diffusion_strength=0"
    )


def test_profile_diffusion_analysis_binds_completed_inputs_and_outcomes():
    report = json.loads(
        (
            ROOT / "results/reopened_hybrid_profile_diffusion_analysis/report.json"
        ).read_text()
    )
    paths = {
        "protocol_sha256": ROOT
        / "configs/reopened_hybrid_profile_diffusion_protocol.json",
        "robustness_protocol_sha256": ROOT
        / "configs/reopened_hybrid_profile_diffusion_raw_robustness_protocol.json",
        "main_report_sha256": ROOT
        / "results/reopened_hybrid_profile_diffusion/report.json",
        "scores_sha256": ROOT
        / "results/reopened_hybrid_profile_diffusion/scores.csv",
        "robustness_report_sha256": ROOT
        / "results/reopened_hybrid_profile_diffusion_raw_robustness/report.json",
    }
    for name, path in paths.items():
        assert report["inputs"][name] == _sha256(path)
    assert report["status"] == "completed"
    assert report["coverage"] == {
        "rows": 38,
        "diffusion_factorial_rows": 27,
        "memory_plus_diffusion_rows": 9,
        "eight_start_complete_root_rows": 28,
        "universal_parameter_boundary_rows": 38,
    }
    assert (
        report["best_complete_cross_domain"]["variant"]
        == "best_memory_control:radial_diffusion_strength=0"
    )
    assert (
        report["best_complete_raw_diffusion"]["variant"]
        == "diff_added_acceleration_l0p7:radial_diffusion_strength=1"
    )


def test_profile_diffusion_effects_and_root_failures_are_not_hidden():
    report = json.loads(
        (
            ROOT / "results/reopened_hybrid_profile_diffusion_analysis/report.json"
        ).read_text()
    )
    scores = pd.read_csv(
        ROOT / "results/reopened_hybrid_profile_diffusion_analysis/joined_scores.csv"
    )
    for effect in report["diffusion_factorial_effects"]:
        assert abs(sum(effect["variance_percent"].values()) - 100.0) < 1.0e-8
    memory_rows = scores[
        scores["family"].str.startswith("best_memory_plus_diffusion")
    ]
    assert len(memory_rows) == 9
    assert not memory_rows["raw_eight_start_all_roots_converged"].any()
    assert scores["Cassini_proxy_pass"].all()
    assert scores["Earth_pass"].all()
    assert scores["Mercury_pass"].all()


def test_profile_diffusion_is_a_lever_but_not_a_cross_domain_advance():
    report = json.loads(
        (
            ROOT / "results/reopened_hybrid_profile_diffusion_analysis/report.json"
        ).read_text()
    )
    raw = report["best_complete_raw_diffusion"]["change_from_local_control"]
    assert raw["raw_eight_start_RMS_change_arcsec"] < 0.0
    assert raw["SPARC_outer_RMSE_change_km_s"] > 8.0
    low_cost = report["smallest_galaxy_cost_raw_improvement"][
        "change_from_local_control"
    ]
    assert low_cost["raw_eight_start_RMS_change_arcsec"] < 0.0
    assert abs(low_cost["SPARC_outer_RMSE_change_km_s"]) < 0.1
    memory = report["best_memory_plus_diffusion_galaxy_row"]
    assert memory["change_from_memory_control"]["SPARC_outer_RMSE_change_km_s"] < 0.0
    assert not memory["raw_eight_start_all_roots_converged"]


def test_profile_diffusion_is_consolidated_into_the_program_scorecard():
    program = json.loads(
        (ROOT / "results/reopened_hybrid_program/program_summary.json").read_text()
    )
    assert program["coverage"]["scored_rows"] == 913
    assert program["coverage"]["unique_formula_settings"] == 801
    assert program["coverage"]["unique_formula_evaluation_contexts"] == 832
    assert program["coverage"]["eight_start_raw_robustness_rows"] == 546
    assert program["observed_sensitivities"]["profile_diffusion_audits"] == {
        "analysis_report_sha256": _sha256(
            ROOT / "results/reopened_hybrid_profile_diffusion_analysis/report.json"
        ),
        "full_variants": 38,
        "diffusion_factorial_cells": 27,
        "memory_plus_diffusion_cells": 9,
        "eight_start_raw_replays": 38,
        "stable_root_complete_replays": 28,
        "universal_parameter_boundary_rows": 38,
    }
    assert (
        program["best_eight_start_verified_cross_domain_compromise"]["stage"]
        == "profile_diffusion"
    )
    assert (
        program["best_eight_start_verified_cross_domain_compromise"]["variant"]
        == "best_memory_control:radial_diffusion_strength=0"
    )
