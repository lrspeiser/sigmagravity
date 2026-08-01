from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_replacement_inventory_parses_sources_and_stops_before_r1(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.csv"
    queue = tmp_path / "queue.csv"
    dynamics = tmp_path / "dynamics.csv"
    photometry = tmp_path / "photometry.csv"
    report = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "inventory_r1_replacement_candidates.py"),
            "--candidate-output",
            str(candidates),
            "--queue-output",
            str(queue),
            "--dynamics-output",
            str(dynamics),
            "--photometry-output",
            str(photometry),
            "--report-output",
            str(report),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(report.read_text(encoding="utf-8"))
    source_rows = pd.read_csv(candidates)
    unique_queue = pd.read_csv(queue)
    profiles = pd.read_csv(dynamics)
    light_fits = pd.read_csv(photometry)

    assert result["parsed_observables"]["newman_velocity_bins"] == 35
    assert result["parsed_observables"]["kaleidoscope_velocity_bins"] == 35
    assert len(profiles) == 70
    assert len(light_fits) == 13
    assert set(light_fits["profile_kind"]) == {"parametric_dPIE_starlight_fit"}
    a2537 = light_fits.loc[
        (light_fits["source_sample"] == "Newman2013") & (light_fits["system"] == "A2537")
    ].iloc[0]
    assert a2537["cluster_redshift"] == 0.294
    assert a2537["stellar_m_to_l_v_sps"] == 2.32
    assert a2537["stellar_mass_sps_1e11_msun"] == 5.86 * 2.32
    assert len(source_rows) == 13
    assert len(unique_queue) == 11
    assert result["candidate_gate"]["published_coverage_candidates"] == 10
    assert result["candidate_gate"]["published_count_gate_passes"]
    assert result["candidate_gate"]["analysis_ready_systems"] == 0
    assert result["candidate_gate"]["lensing_mcmc_ensemble_systems"] == 3
    assert result["candidate_gate"]["systems_with_local_bcg_stellar_component"] == 7
    assert result["candidate_gate"]["systems_with_observable_level_lens_positions"] == 10
    assert result["candidate_gate"]["systems_with_position_redshift_likelihood_inputs"] == 10
    assert result["candidate_gate"]["alternative_metric_forward_model_lensing_ready_systems"] == 0
    assert not result["candidate_gate"]["strict_R1_gate_passes"]
    assert result["stage_decision"]["R1_sample_freeze"] == "not_authorized"
    assert not unique_queue["analysis_ready"].any()
    assert unique_queue["local_bcg_stellar_component_profile"].sum() == 7


def test_a963_is_excluded_only_by_published_lensing_count() -> None:
    config = json.loads(
        (ROOT / "configs" / "r1_replacement_sample_gate.json").read_text(encoding="utf-8")
    )
    a963 = next(
        row
        for row in config["records"]
        if row["source_sample"] == "Newman2013" and row["system"] == "A963"
    )
    assert a963["expected_dynamics_points"] >= 3
    assert a963["published_lensing_constraint_points"] == 2
