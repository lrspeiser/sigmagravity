import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_satellite_membership_report_is_blind_and_machine_readable() -> None:
    report = json.loads(
        (
            ROOT / "results/r1_rxj2129_satellite_membership/classifier_report.json"
        ).read_text(encoding="utf-8")
    )
    assert report["gravity_or_lens_residual_read"] is False
    assert report["metrics"]["training_rows"] == 43
    assert report["bootstrap_replicates"] == 500
    assert report["candidate_count_inside_30arcsec"] == 66
    assert report["off_center_mass_acceleration_likelihood_complete"] is False


def test_satellite_membership_probabilities_are_valid() -> None:
    likelihood = pd.read_csv(
        ROOT / "data/derived/r1_rxj2129_satellite_membership_likelihood.csv"
    )
    columns = [
        "membership_probability",
        "membership_probability_p16",
        "membership_probability_p84",
    ]
    values = likelihood[columns].to_numpy()
    assert np.isfinite(values).all()
    assert (values >= 0.0).all() and (values <= 1.0).all()
    assert (
        likelihood["membership_probability_p16"]
        <= likelihood["membership_probability"]
    ).all()
    assert (
        likelihood["membership_probability"]
        <= likelihood["membership_probability_p84"]
    ).all()
    bootstrap = np.load(
        ROOT / "data/derived/r1_rxj2129_satellite_membership_bootstrap.npz"
    )
    assert bootstrap["membership_probability"].shape == (500, 66)
    assert len(bootstrap["clash_ids"]) == 66
