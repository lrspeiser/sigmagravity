import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
P0569 = ROOT / "results" / "p0569_measured_baryon_extent_audit"
P0570 = ROOT / "results" / "p0570_physical_baryon_residual_lensing"


def read_report(directory):
    return json.loads((directory / "report.json").read_text(encoding="utf-8"))


def test_p0569_measured_extent_audit_has_frozen_coverage():
    report = read_report(P0569)
    assert report["coverage"] == {
        "systems": 4,
        "component_system_rows": 36,
        "components": 9,
    }
    assert len(report["primary"]["systems"]) == 4


def test_p0569_measured_extent_does_not_reproduce_p0568_band():
    report = read_report(P0569)
    assert report["primary"]["median_equivalent_sigma_RMS_kpc"] > 150.0
    assert report["primary"]["median_equivalent_sigma_R80_kpc"] > 160.0
    assert report["primary"]["systems_matching_either_sigma_definition"] == 0
    assert not report["gates"]["measured_baryonic_extent_is_sufficient_scale_explanation"]


def test_p0570_screen_and_exact_fit_coverage_are_complete():
    report = read_report(P0570)
    assert report["coverage"]["systems"] == 4
    assert report["coverage"]["screen_candidates"] == 45
    assert report["coverage"]["exact_system_fits"] == 8
    assert report["coverage"]["components"] == 5
    screen = pd.read_csv(P0570 / "screen_scores.csv")
    assert len(screen[screen.row_type.eq("aggregate")]) == 45
    assert np.isfinite(screen.source_plane_RMS_arcsec).all()


def test_p0570_selected_physical_residual_fails_predictive_gates():
    report = read_report(P0570)
    assert report["selected"]["component"] == "accept_gas_sqrt_morphology"
    assert report["selected"]["extent_scale"] == 0.75
    assert report["selected"]["response_q"] == 2.0
    assert report["validation"]["improvement_fraction"] < 0.05
    assert report["validation"]["selected_to_compact_ratio"] > 1.25
    assert not report["validation"]["selection_selected_all_roots"]
    assert not report["gates"]["exact_selection_roots_pass"]
    assert not report["gates"]["validation_improvement_pass"]
    assert not report["gates"]["compact_halo_ratio_pass"]
    assert not report["gates"]["formula_promoted"]


def test_p0570_potential_field_is_conservative_and_has_circular_null():
    report = read_report(P0570)
    audit = pd.read_csv(P0570 / "field_audits.csv")
    assert audit.normalized_curl_RMS.max() <= 2e-3
    assert report["numerical"]["circular_point_mass_residual_fraction"] <= 1e-2
    assert report["gates"]["curl_pass"]
    assert report["gates"]["circular_null_pass"]
    assert report["cross_domain"]["axisymmetric_null"]
