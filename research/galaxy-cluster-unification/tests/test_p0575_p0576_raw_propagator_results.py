import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def report(name: str) -> dict:
    return json.loads((ROOT / "results" / name / "report.json").read_text(encoding="utf-8"))


def test_p0575_uses_only_twelve_spectroscopic_pre_jwst_images():
    result = report("p0575_smacs0723_raw_position")
    assert result["coverage"]["raw_images"] == 12
    assert result["coverage"]["spectroscopic_families"] == 4
    assert result["coverage"]["calibration_images"] == 6
    assert result["coverage"]["heldout_images"] == 6
    assert result["coverage"]["per_family_deflection_amplitudes"] == 0


def test_p0575_ordinary_poisson_arrival_fails_raw_transfer():
    result = report("p0575_smacs0723_raw_position")
    outcome = result["result"]
    assert outcome["gated_improvement_vs_local_fraction"] < -0.06
    assert outcome["heldout_families_improved"] == 0
    assert outcome["lenstool_reference_heldout_source_plane_RMS_arcsec"] < outcome["gated_heldout_source_plane_RMS_arcsec"]
    assert not result["gates"]["additional_raw_cluster_followup_authorized"]


def test_p0575b_failure_survives_all_splits_and_padding_checks():
    result = report("p0575b_raw_position_robustness")
    outcome = result["result"]
    assert outcome["splits_gated_improves_vs_local"] == 0
    assert outcome["median_gated_improvement_vs_local_fraction"] < -0.07
    assert outcome["lenstool_reference_best_splits"] == 6
    padding = np.asarray(list(outcome["padding_gated_improvement_vs_local_fraction"].values()))
    assert np.all(padding < -0.06)
    assert np.ptp(padding) < 0.001
    assert result["gates"]["raw_failure_survives_robustness"]


def test_p0576_selects_long_wavelength_boundary_candidate():
    result = report("p0576_fractional_routed_propagator")
    selected = result["selected"]
    assert selected["candidate_id"] == "p1.5__f1"
    assert selected["fractional_power_p"] == 1.5
    assert selected["deflection_route_fraction"] == 1.0
    assert not result["gates"]["selected_not_power_boundary"]
    assert not result["gates"]["selected_not_fraction_boundary"]


def test_p0576_improves_both_heldout_families_and_approaches_reference():
    result = report("p0576_fractional_routed_propagator")
    outcome = result["result"]
    assert outcome["improvement_vs_local_fraction"] == pytest.approx(0.44829729845724986)
    assert outcome["heldout_families_improved"] == 2
    assert outcome["selected_heldout_source_plane_RMS_arcsec"] < 0.72
    assert outcome["selected_heldout_source_plane_RMS_arcsec"] < outcome["ordinary_P0574_heldout_source_plane_RMS_arcsec"]
    assert outcome["selected_heldout_source_plane_RMS_arcsec"] / outcome["lenstool_reference_heldout_source_plane_RMS_arcsec"] < 1.11
    assert result["gates"]["fractional_propagator_followup_authorized"]


def test_p0576_preserves_symmetry_structural_nulls():
    cross = report("p0576_fractional_routed_propagator")["cross_domain"]
    assert cross["solar_routed_fraction"] == 0.0
    assert cross["SPARC_angular_velocity_change_km_s"] == 0.0
