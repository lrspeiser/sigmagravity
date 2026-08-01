import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_rxj2129_lens_observable_gate() -> None:
    report = json.loads(
        (ROOT / "results/r1_rxj2129_lens_observables/report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["gravity_or_published_gr_mass_map_read"] is False
    assert report["independent_lens_residual_evaluated"] is False
    assert report["observable_likelihood_gate_pass"] is True
    assert report["observed_counts"] == {
        "listed_images": 25,
        "spectroscopic_likelihood_images": 21,
        "spectroscopic_source_families": 7,
        "excluded_photometric_images": 4,
        "strict_images_inside_dynamics_support": 3,
        "strict_inner_source_families": 3,
    }
    assert report["coordinate_covariance_shape"] == [42, 42]


def test_rxj2129_lens_ledger_and_covariance() -> None:
    images = pd.read_csv(ROOT / "data/derived/r1_rxj2129_lens_observables.csv")
    assert len(images) == 25
    assert images["likelihood_included"].sum() == 21
    assert set(images.loc[~images["likelihood_included"], "source_family"]) == {2}
    assert not images["published_model_rms_ingested"].any()
    covariance = pd.read_csv(
        ROOT / "data/derived/r1_rxj2129_lens_coordinate_covariance.csv",
        index_col="row",
    ).to_numpy()
    assert covariance.shape == (42, 42)
    assert np.allclose(covariance, covariance.T)
    assert np.allclose(np.diag(covariance), 0.25)
    assert np.linalg.eigvalsh(covariance).min() >= 0.0
