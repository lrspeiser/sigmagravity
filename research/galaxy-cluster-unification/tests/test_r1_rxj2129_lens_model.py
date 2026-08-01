import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_rxj2129_independent_lens_model_preserves_the_blind_gate() -> None:
    protocol = json.loads(
        (ROOT / "configs/r1_rxj2129_lens_model_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    report = json.loads(
        (ROOT / "results/r1_rxj2129_lens_model/report.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["status"] == "frozen_before_independent_image_residual_evaluation"
    assert report["published_gr_mass_map_read"] is False
    assert report["published_best_fit_mass_parameters_used"] is False
    assert report["published_model_residual_used"] is False
    assert report["new_force_or_action_fit"] is False
    assert report["counts"]["images"] == 21
    assert report["counts"]["source_families"] == 7
    assert report["counts"]["training_images"] == 14
    assert report["counts"]["heldout_images"] == 7


def test_rxj2129_lens_model_selection_uses_exact_heldout_images() -> None:
    report = json.loads(
        (ROOT / "results/r1_rxj2129_lens_model/report.json").read_text(
            encoding="utf-8"
        )
    )
    heldout_a = report["training_fits"]["model_A"]["heldout_exact_score"]
    heldout_b = report["training_fits"]["model_B"]["heldout_exact_score"]
    assert heldout_a["images"] == 7
    assert heldout_b["images"] == 7
    assert heldout_a["all_roots_converged"] is True
    assert heldout_b["all_roots_converged"] is True
    assert heldout_a["degrees_of_freedom"] == 14
    assert heldout_b["degrees_of_freedom"] == 14
    assert report["model_selection"]["selected_model"] == "model_A"
    assert report["model_selection"]["model_B_selected"] is False
    assert (
        report["model_selection"]["model_B_heldout_radial_rms_improvement_fraction"]
        < 0.10
    )


def test_rxj2129_lens_exact_predictions_and_covariance() -> None:
    report = json.loads(
        (ROOT / "results/r1_rxj2129_lens_model/report.json").read_text(
            encoding="utf-8"
        )
    )
    prediction = pd.read_csv(
        ROOT / "data/derived/r1_rxj2129_lens_image_predictions.csv"
    )
    final = prediction[prediction["stage"] == "all_images"]
    assert len(final) == 21
    assert final["root_converged"].all()
    radial_rms = np.sqrt(
        np.mean(final["delta_x_arcsec"] ** 2 + final["delta_y_arcsec"] ** 2)
    )
    assert np.isclose(
        radial_rms,
        report["all_image_refit"]["exact_score"]["exact_radial_rms_arcsec"],
    )
    assert report["independent_lens_engineering_gate_pass"] is True
    assert report["heldout_predictive_closure_established"] is False
    assert report["weyl_response_reconstruction_authorized"] is False
    assert report["advance_checks"]["selected_model_all_image_exact_radial_rms"] is True
    assert report["advance_checks"]["selected_model_exact_coordinate_reduced_chi2"] is True

    covariance = pd.read_csv(
        ROOT / "data/derived/r1_rxj2129_lens_parameter_covariance.csv", index_col=0
    ).to_numpy()
    assert covariance.shape == (24, 24)
    assert np.isfinite(covariance).all()
    assert np.allclose(covariance, covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    assert eigenvalues.min() >= -1e-9 * max(1.0, eigenvalues.max())
