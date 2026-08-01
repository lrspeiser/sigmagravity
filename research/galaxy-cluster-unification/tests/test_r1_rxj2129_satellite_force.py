import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_satellite_force_report_preserves_failed_or_passed_gate_honestly() -> None:
    report = json.loads(
        (ROOT / "results/r1_rxj2129_satellite_membership/force_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["gravity_or_lens_residual_read"] is False
    assert report["inner_candidates"] == 66
    assert report["monte_carlo_draws"] == 2000
    assert report["lens_member_dark_subhalo_likelihood_complete"] is False


def test_satellite_force_profile_covariance_and_draws_are_finite() -> None:
    profile = pd.read_csv(
        ROOT / "data/derived/r1_rxj2129_satellite_acceleration_profile.csv"
    )
    assert len(profile) == 4
    assert np.isfinite(profile.select_dtypes(include=[np.number])).all().all()
    covariance = pd.read_csv(
        ROOT / "data/derived/r1_rxj2129_satellite_acceleration_covariance.csv",
        index_col="row",
    ).to_numpy()
    assert covariance.shape == (4, 4)
    assert np.allclose(covariance, covariance.T)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-30
    draws = np.load(ROOT / "data/derived/r1_rxj2129_satellite_acceleration_draws.npz")
    assert draws["radial_acceleration_m_s2"].shape == (2000, 4)
