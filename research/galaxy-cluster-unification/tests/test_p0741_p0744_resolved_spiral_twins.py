from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _report(stage: str) -> dict:
    return json.loads((RESULTS / stage / "report.json").read_text(encoding="utf-8"))


def test_haar_transform_is_an_orthonormal_round_trip() -> None:
    from voidscreen.multiscale_galaxy_generator import haar2_forward, haar2_inverse

    rng = np.random.default_rng(20260802)
    source = rng.normal(size=(16, 16))
    coefficients = haar2_forward(source)
    restored = haar2_inverse(coefficients)
    assert np.allclose(restored, source, rtol=0.0, atol=1e-12)
    assert np.isclose(np.sum(np.square(coefficients)), np.sum(np.square(source)))


def test_sparse_galaxy_package_is_deterministic_mass_preserving_and_velocity_blind() -> None:
    from voidscreen.multiscale_galaxy_generator import (
        extract_galaxy_parameters,
        render_galaxy,
    )

    axis = np.linspace(-4.0, 4.0, 33)
    xx, yy = np.meshgrid(axis, axis)
    gas = np.exp(-np.hypot(xx, yy) / 2.0) * (1.0 + 0.15 * np.cos(2.0 * np.arctan2(yy, xx)))
    stars = 3.0 * np.exp(-np.hypot(xx, yy) / 1.2)
    package = extract_galaxy_parameters(
        "test",
        axis,
        gas,
        stars,
        coefficient_count_per_component=128,
        source_observables={"surfaceDensityOnly": True},
    )
    duplicate = extract_galaxy_parameters(
        "test",
        axis,
        gas,
        stars,
        coefficient_count_per_component=128,
        source_observables={"surfaceDensityOnly": True},
    )
    rendered = render_galaxy(package, axis)
    spacing = axis[1] - axis[0]

    assert package["contentSha256"] == duplicate["contentSha256"]
    assert package["gravityParameters"] == {}
    assert package["velocityTargetsUsed"] is False
    assert np.isclose(rendered["gas"].sum() * spacing**2, gas.sum() * spacing**2)
    assert np.isclose(rendered["stars"].sum() * spacing**2, stars.sum() * spacing**2)


def test_development_selection_and_velocity_reveal_preserve_the_seals() -> None:
    p0741 = _report("p0741_fused_spiral_baryonic_registration_development")
    p0742 = _report("p0742_spiral_twin_roundtrip_development")
    p0743 = _report("p0743_multiscale_spiral_twin_development")
    p0744 = _report("p0744_development_velocity_field_reveal")

    assert p0741["status"] == "pass"
    assert p0741["velocityOrDispersionArraysOpened"] == 0
    assert p0742["status"] == "fail"
    assert p0742["selectedTier"] is None
    assert p0743["status"] == "pass"
    assert p0743["selectedTier"] == "haar_256"
    assert p0743["observedVelocityArraysOpened"] == 0
    assert p0744["status"] == "pass"
    assert p0744["targetArraysOpened"] == 8
    assert p0744["validationArraysOpened"] == 0
    assert p0744["holdoutArraysOpened"] == 0
    assert p0744["gravityParametersFitted"] == 0
    assert p0744["darkMatterParameters"] == 0
    assert all(p0744["checks"].values())


def test_real_and_fake_predictions_are_scored_separately() -> None:
    scores = pd.read_csv(
        RESULTS
        / "p0744_development_velocity_field_reveal"
        / "velocity_field_scores.csv"
    )
    source = scores[scores.map_kind == "registered_baryons"]
    twin = scores[scores.map_kind == "fake_twin"]

    assert len(scores) == 16
    assert set(source.galaxy) == {"NGC2403", "NGC3198", "NGC5055", "NGC7793"}
    assert set(source.model) == {"newtonian_thin_sheet", "fixed_simple_mond"}
    assert np.isfinite(
        scores[
            [
                "gas_weighted_rmse_km_s",
                "gas_weighted_uncertainty_rms_km_s",
                "field_error_ratio",
                "twin_source_transport_rmse_km_s",
            ]
        ].to_numpy()
    ).all()
    assert scores.twin_source_transport_rmse_km_s.max() < 8.0
    assert source[source.model == "fixed_simple_mond"].gas_weighted_rmse_km_s.median() < source[
        source.model == "newtonian_thin_sheet"
    ].gas_weighted_rmse_km_s.median()
    assert np.allclose(
        source.sort_values(["galaxy", "model"]).twin_source_transport_rmse_km_s,
        twin.sort_values(["galaxy", "model"]).twin_source_transport_rmse_km_s,
    )


def test_validation_reveal_preserves_generator_failure_and_holdout_seal() -> None:
    p0745a = _report("p0745a_validation_baryonic_registration")
    p0745b = _report("p0745b_validation_fused_baryonic_registration")
    p0745c = _report("p0745c_validation_multiscale_spiral_twins")
    p0746 = _report("p0746_validation_velocity_field_reveal")
    p0747 = _report("p0747_post_reveal_kinematic_axis_diagnostic")

    assert p0745a["status"] == "fail"
    assert p0745a["validationArraysOpened"] == 6
    assert p0745a["velocityOrDispersionArraysOpened"] == 0
    assert p0745b["status"] == "pass"
    assert p0745b["validationArraysOpened"] == 10
    assert p0745b["velocityOrDispersionArraysOpened"] == 0
    assert p0745c["status"] == "fail"
    assert p0745c["selectedTier"] is None
    assert p0745c["tiers"][0]["checks"]["gasMaximumNormalizedL2"] is False
    assert p0745c["observedVelocityArraysOpened"] == 0
    assert p0745c["holdoutArraysOpened"] == 0
    assert p0746["status"] == "fail"
    assert p0746["targetArraysOpened"] == 4
    assert p0746["validationArraysOpened"] == 4
    assert p0746["holdoutArraysOpened"] == 0
    assert p0746["gravityParametersFitted"] == 0
    assert p0746["checks"]["maximumTwinSourcePredictionTransportRmseKmS"] is False
    assert p0747["status"] == "pass"
    assert p0747["holdoutArraysOpened"] == 0
    assert p0747["fittedObservationNuisances"] == 2
    assert p0747["fittedGravityParameters"] == 0


def test_validation_scores_separate_formula_twin_and_geometry_errors() -> None:
    raw = pd.read_csv(
        RESULTS / "p0746_validation_velocity_field_reveal" / "velocity_field_scores.csv"
    )
    diagnostic = pd.read_csv(
        RESULTS
        / "p0747_post_reveal_kinematic_axis_diagnostic"
        / "diagnostic_velocity_field_scores.csv"
    )
    geometry = pd.read_csv(
        RESULTS / "p0747_post_reveal_kinematic_axis_diagnostic" / "kinematic_axis_audit.csv"
    ).set_index("galaxy")

    assert set(raw.galaxy) == {"NGC3521", "NGC6946"}
    assert len(raw) == 8
    assert geometry.loc["NGC3521", "kinematic_phase_offset_deg_in_registered_plane"] < 1.0
    assert geometry.loc["NGC6946", "kinematic_phase_offset_deg_in_registered_plane"] > 50.0
    for galaxy in ("NGC3521", "NGC6946"):
        for model in ("newtonian_thin_sheet", "fixed_simple_mond"):
            source = raw[
                (raw.galaxy == galaxy)
                & (raw.model == model)
                & (raw.map_kind == "registered_baryons")
            ].iloc[0]
            twin = raw[
                (raw.galaxy == galaxy)
                & (raw.model == model)
                & (raw.map_kind == "fake_twin")
            ].iloc[0]
            assert np.isclose(
                source.twin_source_transport_rmse_km_s,
                twin.twin_source_transport_rmse_km_s,
            )
    ngc6946_mond_raw = raw[
        (raw.galaxy == "NGC6946")
        & (raw.model == "fixed_simple_mond")
        & (raw.map_kind == "registered_baryons")
    ].iloc[0]
    ngc6946_mond_axis = diagnostic[
        (diagnostic.galaxy == "NGC6946")
        & (diagnostic.model == "fixed_simple_mond")
        & (diagnostic.prediction_kind == "registered_baryons_kinematic_axis")
    ].iloc[0]
    assert ngc6946_mond_axis.gas_weighted_rmse_km_s < 0.5 * ngc6946_mond_raw.gas_weighted_rmse_km_s
    assert ngc6946_mond_axis.error_band == "miss"
