from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_hst_profile_protocol_remains_residual_blind() -> None:
    protocol = json.loads(
        (ROOT / "configs/r1_rxj2129_bcg_icl_protocol.json").read_text(encoding="utf-8")
    )
    assert protocol["authorization"]["gravity_response_fit"] is False
    assert protocol["authorization"]["lens_mass_fit"] is False
    assert protocol["component_identifiability_gate"]["failure_response"].startswith(
        "Retain the nonparametric"
    )


def test_nonparametric_hst_profile_passes_frozen_coverage_and_center_gates() -> None:
    report = json.loads(
        (ROOT / "results/r1_rxj2129_bcg_icl/profile_extraction_report.json").read_text(
            encoding="utf-8"
        )
    )
    profile = pd.read_csv(ROOT / "data/derived/r1_rxj2129_hst_surface_brightness_profile.csv")
    assert report["gravity_or_lens_residual_read"] is False
    assert report["profile_extraction_gate_pass"] is True
    assert report["radial_bins_usable"] >= report["minimum_required_usable_bins"]
    assert report["refined_center_offset_arcsec"] <= 0.30
    assert len(profile) == 60
    assert profile["profile_gate_usable"].sum() == report["radial_bins_usable"]
    assert np.all(np.diff(profile["radius_mid_arcsec"]) > 0)


def test_hst_profile_joint_covariance_is_finite_symmetric_and_psd() -> None:
    covariance = pd.read_csv(
        ROOT / "data/derived/r1_rxj2129_hst_surface_brightness_covariance.csv",
        index_col="row",
    ).to_numpy()
    assert covariance.shape == (98, 98)
    assert np.all(np.isfinite(covariance))
    assert np.allclose(covariance, covariance.T)
    assert np.min(np.linalg.eigvalsh(covariance)) >= -1e-30
    cross_band = covariance[:49, 49:]
    assert np.any(np.abs(cross_band) > 0)
