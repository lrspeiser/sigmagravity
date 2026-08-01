from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_a2261_reduction_protocol_freezes_hard_overlap_target() -> None:
    config = json.loads((ROOT / "configs/r1_a2261_gmos_reduction_covariance_protocol.json").read_text())
    assert "before_any_science_reduction" in config["status"]
    assert config["spatial_extraction"]["signed_bins"] == 9
    assert config["spatial_extraction"]["signed_bin_edges_arcsec"] == [-10.5, -7.0, -4.5, -2.0, -0.3, 0.3, 2.0, 4.5, 7.0, 10.5]
    assert config["profile_acceptance"]["minimum_finite_signed_bins"] == 9
    assert config["profile_acceptance"]["both_outer_signed_bins_required"] is True
    assert config["profile_acceptance"]["minimum_realized_support_kpc"] == 36.0
    assert config["profile_acceptance"]["minimum_independent_lens_families_inside_realized_support"] == 3
    assert config["covariance_protocol"]["replicates"] == 200
    assert config["profile_acceptance"]["maximum_sigma_shift_fraction_over_systematic_grid"] == 0.1
    assert config["authorization"]["fit_gravity_response"] is False


def test_a2261_protocol_gate_opens_only_p1() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/audit_r1_a2261_gmos_reduction_protocol.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((ROOT / "results/r1_a2261_gmos_reduction_protocol/report.json").read_text())
    assert report["raw_acquisition_gate_passed"] is True
    assert report["pre_pixel_three_family_geometry_gate_passed"] is True
    assert report["template_checksum_passed"] is True
    assert report["science_products_present_at_freeze"] == []
    assert report["protocol_frozen_before_science_products"] is True
    assert report["gates"]["protocol_freeze_gate_passed"] is True
    assert report["authorization"]["acquire_bpm_and_audit_environment"] is True
    assert report["authorization"]["execute_science_reduction"] is False
    assert report["gates"]["gravity_response_fit_authorized"] is False
