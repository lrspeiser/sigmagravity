from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0631_observation_matched_replicas"


def test_full_catalog_and_frozen_split_are_present():
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    assert report["replica_gate_pass"] is True
    assert report["split_counts"] == {"train": 81, "development": 27, "holdout": 23}
    scores = pd.read_csv(RESULTS / "replica_scores.csv")
    assert len(scores) == 131
    assert scores.galaxy.nunique() == 131


def test_nontrivial_camera_and_profile_checks_pass():
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    aggregate = report["aggregate"]["all"]
    assert aggregate["median_angular_photometry_rmse_dex"] < 1.0e-3
    assert aggregate["median_pixelized_rotation_rmse_km_s"] < 1.0
    assert aggregate["median_abs_total_light_fractional_error"] < 0.01


def test_representative_artifacts_span_declared_regimes():
    expected = {"DDO154", "NGC2403", "NGC2841", "NGC7814"}
    particles = pd.read_csv(RESULTS / "particle_checks.csv")
    assert set(particles.galaxy) == expected
    assert (particles.particles == 65536).all()
    assert particles.luminosity_fractional_error.abs().max() < 1.0e-12
    for galaxy in expected:
        assert (RESULTS / "representatives" / f"{galaxy}_replica.png").stat().st_size > 10000
        assert (RESULTS / "representatives" / f"{galaxy}_replica.npz").stat().st_size > 10000
