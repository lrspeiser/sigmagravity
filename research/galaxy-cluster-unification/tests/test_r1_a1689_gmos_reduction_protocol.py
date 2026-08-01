from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_a1689_reduction_protocol_freezes_numerical_gates() -> None:
    config = json.loads((ROOT / "configs/r1_a1689_gmos_reduction_covariance_protocol.json").read_text())
    assert "before_any_science_reduction" in config["status"]
    assert config["spatial_extraction"]["signed_bins"] == 9
    assert len(config["spatial_extraction"]["signed_bin_edges_arcsec"]) == 10
    assert config["calibration_acceptance"]["maximum_wavelength_solution_rms_angstrom"] == 0.2
    assert config["spatial_extraction"]["minimum_median_signal_to_noise_per_angstrom_per_signed_bin"] == 5.0
    assert config["covariance_protocol"]["replicates"] == 200
    assert config["covariance_protocol"]["minimum_successful_replicates"] == 180
    assert config["profile_acceptance"]["minimum_finite_symmetrized_radial_bins"] == 4
    assert config["profile_acceptance"]["maximum_sigma_shift_fraction_over_systematic_grid"] == 0.1
    assert config["authorization"]["fit_gravity_response"] is False
    assert config["authorization"]["fit_new_force_or_action"] is False


def test_a1689_protocol_gate_opens_only_environment_setup() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/audit_r1_a1689_gmos_reduction_protocol.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((ROOT / "results/r1_a1689_gmos_reduction_protocol/report.json").read_text())
    assert report["raw_acquisition_gate_passed"] is True
    assert report["lens_geometry_gate_passed"] is True
    assert report["template_checksum_passed"] is True
    assert report["science_products_present_at_freeze"] == []
    assert report["protocol_frozen_before_science_products"] is True
    assert report["gates"]["protocol_freeze_gate_passed"] is True
    assert report["authorization"]["acquire_bpm_and_install_environment"] is True
    assert report["authorization"]["execute_science_reduction"] is False
    assert report["gates"]["gravity_response_fit_authorized"] is False
