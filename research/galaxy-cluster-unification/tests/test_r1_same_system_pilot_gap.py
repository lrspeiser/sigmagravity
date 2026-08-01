from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_same_system_targets_freeze_numeric_checkpoints() -> None:
    config = json.loads((ROOT / "configs/r1_same_system_pilot_targets.json").read_text())
    assert config["frozen_before_gap_audit"] is True
    strict = config["strict_system_definition"]
    assert strict["numerical_dynamics_radial_points_minimum"] == 3
    assert strict["independent_lensing_radial_points_on_dynamics_support_minimum"] == 3
    assert strict["coordinate_covariance_not_borrowed_from_fitted_gr_residuals_required"] is True
    assert config["current_verified_baseline"]["systems_passing_three_plus_three_structural_overlap"] == 2
    assert config["current_verified_baseline"]["target_strict_systems"] == 10
    assert config["authorization"]["inspect_gravity_residuals"] is False


def test_same_system_gap_audit_keeps_theory_gate_closed(tmp_path: Path) -> None:
    output = tmp_path / "ledger.csv"
    report_path = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/audit_r1_same_system_pilot_gap.py"),
            "--output",
            str(output),
            "--report",
            str(report_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(report_path.read_text())
    ledger = pd.read_csv(output)
    assert len(ledger) == 15
    assert report["numerical_resolved_dynamics_systems"] == 13
    assert report["systems_with_local_observable_lens_positions"] == 12
    assert report["systems_passing_three_plus_three_structural_overlap"] == 2
    assert report["structural_pass_systems"] == ["MACS J1206", "RX J2129"]
    assert report["structural_system_gap"] == 8
    assert report["strict_ready_system_gap"] == 10
    assert report["systems_with_complete_baryonic_forward_inputs"] == 0
    assert report["systems_with_theory_neutral_joint_covariance"] == 0
    assert report["cycle_1_checkpoint"]["passed"] is False
    assert not ledger["strict_r1_ready"].any()
    a2261 = ledger.loc[ledger["system"] == "Abell 2261"].iloc[0]
    assert not bool(a2261["numerical_dynamics_profile_local"])
    assert bool(a2261["raw_dynamics_acquisition_gate_passed"])
    assert bool(a2261["dynamics_reconstruction_attempted"])
    assert not bool(a2261["dynamics_reconstruction_gate_passed"])
    assert "0.3-arcsec ceiling" in a2261["primary_obstruction"]
    assert report["a2261_frozen_extended_support_target_kpc"] == 36.0
    a1689 = ledger.loc[ledger["system"] == "Abell 1689"].iloc[0]
    assert bool(a1689["geometry_prescreen_pass"])
    assert bool(a1689["raw_dynamics_acquisition_gate_passed"])
    assert bool(a1689["dynamics_reconstruction_attempted"])
    assert not bool(a1689["dynamics_reconstruction_gate_passed"])
    assert not bool(a1689["numerical_dynamics_profile_local"])
    assert not bool(a1689["three_plus_three_structural_pass"])
    a383 = ledger.loc[ledger["system"] == "A383"].iloc[0]
    assert bool(a383["raw_dynamics_acquisition_gate_passed"])
    assert bool(a383["dynamics_reconstruction_attempted"])
    assert not bool(a383["dynamics_reconstruction_gate_passed"])
    assert "0.234" in a383["primary_obstruction"]
    assert "0.215" in a383["primary_obstruction"]
    assert "0.200-Angstrom ceiling" in a383["primary_obstruction"]
    assert report["raw_dynamics_geometry_qualified_pending_reconstruction"] == []
    assert report["failed_frozen_raw_reconstruction_systems"] == [
        "A2537",
        "A383",
        "Abell 1689",
        "Abell 2261",
        "MS2137",
    ]
    a2537 = ledger.loc[ledger["system"] == "A2537"].iloc[0]
    assert bool(a2537["raw_dynamics_acquisition_gate_passed"])
    assert bool(a2537["dynamics_reconstruction_attempted"])
    assert not bool(a2537["dynamics_reconstruction_gate_passed"])
    assert "0.225" in a2537["primary_obstruction"]
    assert "0.220" in a2537["primary_obstruction"]
    assert "disturbed-control" in a2537["primary_obstruction"]
    assert report["route_rethink_triggered"] is True
    assert report["next_stage"]["primary_system"] == "RX J2129"
    ms2137 = ledger.loc[ledger["system"] == "MS2137"].iloc[0]
    assert bool(ms2137["raw_dynamics_acquisition_gate_passed"])
    assert bool(ms2137["dynamics_reconstruction_attempted"])
    assert not bool(ms2137["dynamics_reconstruction_gate_passed"])
    assert "1.438" in ms2137["primary_obstruction"]
    assert "20-spaxel-per-opposite-half" in ms2137["primary_obstruction"]
    assert report["authorization"]["fit_new_force_or_action"] is False
