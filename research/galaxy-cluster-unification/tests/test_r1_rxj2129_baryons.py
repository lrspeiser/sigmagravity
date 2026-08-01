from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_baryonic_protocol_is_residual_blind_and_blocks_readiness() -> None:
    protocol = json.loads(
        (ROOT / "configs/r1_rxj2129_baryonic_protocol.json").read_text(encoding="utf-8")
    )
    assert protocol["selection_blinding"]["gravity_residual_inspected"] is False
    assert protocol["authorization"]["fit_gravity_response"] is False
    assert protocol["authorization"]["set_strict_r1_ready"] is False
    assert protocol["component_gates"]["complete_baryonic_forward_inputs"] is False
    assert "forbidden_inference" in protocol["gas_anchor"]


def test_baryonic_reconstruction_preserves_shared_bcg_covariance() -> None:
    report = json.loads(
        (ROOT / "results/r1_rxj2129_baryons/report.json").read_text(encoding="utf-8")
    )
    profile = pd.read_csv(ROOT / "data/derived/r1_rxj2129_bcg_hernquist_profile.csv")
    covariance = pd.read_csv(
        ROOT / "data/derived/r1_rxj2129_bcg_hernquist_acceleration_covariance.csv",
        index_col="row",
    ).to_numpy()

    assert len(profile) == 4
    assert np.all(np.diff(profile["radius_kpc"]) > 0)
    assert np.all(np.diff(profile["bcg_mass_enclosed_msun"]) > 0)
    assert np.all(profile["gas_profile_numeric_at_bin"] == 0)
    assert np.allclose(covariance, covariance.T)
    assert np.min(np.linalg.eigvalsh(covariance)) >= -1e-35
    assert np.any(np.abs(covariance - np.diag(np.diag(covariance))) > 0)
    assert report["gravity_residual_read_or_fit"] is False
    assert report["complete_baryonic_forward_inputs"] is False
    assert report["strict_r1_ready"] is False


def test_satellite_ledger_uses_only_predeclared_membership_classes() -> None:
    candidates = pd.read_csv(ROOT / "data/derived/r1_rxj2129_satellite_candidates.csv")
    report = json.loads(
        (ROOT / "results/r1_rxj2129_baryons/report.json").read_text(encoding="utf-8")
    )
    assert set(candidates["membership_class"]) <= {
        "secure_spec_member",
        "possible_photo_member",
    }
    assert np.all(candidates["separation_arcsec"] > 1.0)
    assert np.all(candidates["stellar_mass_upper_0p30dex_msun"] >= candidates["stellar_mass_msun"])
    assert report["satellites"]["selected_candidate_count_full_footprint"] == len(candidates)
    assert report["satellites"]["catalog_footprint_caveat"]
