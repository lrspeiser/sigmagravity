from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_a1689_final_profile_gate_is_the_conjunction_of_frozen_checks() -> None:
    systematic = json.loads((ROOT / "results/r1_a1689_gmos_systematics/report.json").read_text())
    if not systematic["authorization"]["assemble_final_signed_and_symmetrized_covariance"]:
        assert systematic["gates"]["P3e_systematic_shift_gate_passed"] is False
        assert not (ROOT / "results/r1_a1689_gmos_final_profile/report.json").exists()
        return
    report = json.loads((ROOT / "results/r1_a1689_gmos_final_profile/report.json").read_text())
    expected = all(report["checks"].values())
    assert report["gates"]["P3_profile_covariance_gate_passed"] is expected
    assert report["gates"]["A1689_numerical_dynamics_profile_promoted"] is expected
    assert report["authorization"]["record_numerical_dynamics_profile"] is expected
    assert report["gates"]["gravity_response_fit_authorized"] is False
    assert report["gates"]["weyl_response_fit_authorized"] is False
    assert report["authorization"]["fit_new_force_or_action"] is False


def test_a1689_final_profile_and_covariance_are_consistent() -> None:
    systematic = json.loads((ROOT / "results/r1_a1689_gmos_systematics/report.json").read_text())
    if not systematic["authorization"]["assemble_final_signed_and_symmetrized_covariance"]:
        assert not (ROOT / "data/derived/r1_a1689_gmos_final_profile.csv").exists()
        assert not (ROOT / "data/derived/r1_a1689_gmos_final_covariance.npz").exists()
        return
    profile = pd.read_csv(ROOT / "data/derived/r1_a1689_gmos_final_profile.csv")
    arrays = np.load(ROOT / "data/derived/r1_a1689_gmos_final_covariance.npz")
    assert len(profile) == 5
    assert profile["radial_bin"].tolist() == [1, 2, 3, 4, 5]
    assert np.all(np.diff(profile["radius_kpc"]) > 0)
    assert arrays["signed_total_joint_covariance"].shape == (18, 18)
    assert arrays["radial_total_joint_covariance"].shape == (10, 10)
    assert np.allclose(arrays["signed_total_joint_covariance"], arrays["signed_total_joint_covariance"].T)
    assert np.allclose(arrays["radial_total_joint_covariance"], arrays["radial_total_joint_covariance"].T)
    assert np.allclose(profile["sigma_km_s"], arrays["radial_joint"][5:])
