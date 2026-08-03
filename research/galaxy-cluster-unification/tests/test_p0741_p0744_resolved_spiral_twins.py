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
