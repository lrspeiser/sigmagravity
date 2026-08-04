from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from voidscreen.sigma_covariant_feature_inference import (
    EquivariantDataset,
    MetricFeature,
    build_metric_feature_library,
    convergence_to_shear,
    fit_equivariant_ridge,
    predict_residual,
    score_prediction,
    tensor_e_projection,
)

ROOT = Path(__file__).resolve().parents[1]


def test_constant_convergence_has_no_interior_shear() -> None:
    convergence = np.ones((65, 65))
    shear_1, shear_2 = convergence_to_shear(convergence, padding_factor=1)
    assert np.max(np.abs(shear_1)) < 1.0e-12
    assert np.max(np.abs(shear_2)) < 1.0e-12


def test_tensor_projection_and_metric_triplets_are_rotation_equivariant() -> None:
    axis = np.linspace(-4.0, 4.0, 81)
    east, north = np.meshgrid(axis, axis)
    total = np.exp(-(east**2 / 2.0 + north**2 / 0.7))
    gas = 0.6 * np.exp(-((east - 0.7) ** 2 / 2.5 + north**2 / 1.2))
    stars = np.maximum(total - 0.3 * gas, 0.0)
    first = build_metric_feature_library(
        gas + stars,
        gas,
        stars,
        spacing_kpc=float(axis[1] - axis[0]),
        scales_kpc=(0.6,),
        padding_factor=2,
    )
    rotated = build_metric_feature_library(
        np.rot90(gas + stars),
        np.rot90(gas),
        np.rot90(stars),
        spacing_kpc=float(axis[1] - axis[0]),
        scales_kpc=(0.6,),
        padding_factor=2,
    )
    by_name = {feature.name: feature for feature in first}
    rotated_by_name = {feature.name: feature for feature in rotated}
    interior = np.s_[12:-12, 12:-12]
    for name, feature in by_name.items():
        other = rotated_by_name[name]
        np.testing.assert_allclose(
            other.convergence[interior],
            np.rot90(feature.convergence)[interior],
            atol=2.0e-8,
            rtol=2.0e-5,
        )
        np.testing.assert_allclose(
            other.shear_1[interior],
            -np.rot90(feature.shear_1)[interior],
            atol=2.0e-7,
            rtol=2.0e-4,
        )
        np.testing.assert_allclose(
            other.shear_2[interior],
            -np.rot90(feature.shear_2)[interior],
            atol=2.0e-6,
            rtol=2.0e-4,
        )


def test_tensor_e_projection_of_zero_tensor_is_zero() -> None:
    zeros = np.zeros((33, 35))
    projected = tensor_e_projection(zeros, zeros, padding_factor=2)
    assert projected.shape == zeros.shape
    assert np.count_nonzero(projected) == 0


def _synthetic_dataset(name: str, multiplier: float) -> EquivariantDataset:
    axis = np.linspace(-2.0, 2.0, 41)
    east, north = np.meshgrid(axis, axis)
    convergence_a = np.exp(-(east**2 + 0.7 * north**2))
    convergence_b = east * np.exp(-(0.8 * east**2 + north**2))
    a_shear = convergence_to_shear(convergence_a)
    b_shear = convergence_to_shear(convergence_b)
    feature_a = MetricFeature(
        "a",
        "scalar_scale",
        convergence_a,
        a_shear[0],
        a_shear[1],
    )
    feature_b = MetricFeature(
        "b",
        "total_tidal",
        convergence_b,
        b_shear[0],
        b_shear[1],
    )
    base = tuple(np.zeros_like(east) for _ in range(3))
    target = tuple(
        multiplier * (1.7 * first - 0.4 * second)
        for first, second in zip(
            (feature_a.convergence, feature_a.shear_1, feature_a.shear_2),
            (feature_b.convergence, feature_b.shear_1, feature_b.shear_2),
            strict=True,
        )
    )
    # Feature amplitudes and target amplitudes scale together, so the physical
    # coefficients remain common across the two synthetic objects.
    scaled_features = {
        "a": MetricFeature(
            "a",
            "scalar_scale",
            multiplier * feature_a.convergence,
            multiplier * feature_a.shear_1,
            multiplier * feature_a.shear_2,
        ),
        "b": MetricFeature(
            "b",
            "total_tidal",
            multiplier * feature_b.convergence,
            multiplier * feature_b.shear_1,
            multiplier * feature_b.shear_2,
        ),
    }
    return EquivariantDataset(
        name=name,
        mask=np.hypot(east, north) < 1.7,
        base=base,
        target=target,
        features=scaled_features,
    )


def test_shared_ridge_recovers_one_metric_coefficients_and_transfers() -> None:
    first = _synthetic_dataset("first", 1.0)
    second = _synthetic_dataset("second", 2.3)
    fit = fit_equivariant_ridge([first], family="total_tidal", alpha=0.0)
    np.testing.assert_allclose(fit.coefficients["a"], 1.7, atol=1.0e-10)
    np.testing.assert_allclose(fit.coefficients["b"], -0.4, atol=1.0e-10)
    predicted = predict_residual(second, fit.coefficients)
    for actual, expected in zip(predicted, second.target, strict=True):
        np.testing.assert_allclose(actual, expected, atol=1.0e-10)
    assert score_prediction(second, fit.coefficients)["full_field_NRMSE"] < 1.0e-10


def test_completed_v15_report_obeys_frozen_protocol() -> None:
    path = ROOT / "results" / "sigma_v15_spent_invariant_inference" / "report.json"
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["sample_is_spent"] is True
    assert report["observational_validation_claim"] is False
    assert report["per_cluster_gravity_parameters"] == 0
    assert report["one_metric_feature_triplets"] is True
    assert len(report["base_AQUAL_scores"]) == 2
    assert set(report["family_results"]) == {
        "scalar_scale",
        "total_tidal",
        "component_overlap",
    }
    assert report["selected_family"] in report["family_results"]
    for result in report["family_results"].values():
        assert len(result["self_fit_scores"]) == 2
        assert -1.0 <= result["directional_prediction_agreement_cosine"] <= 1.0
    assert report["selected_family"] == "scalar_scale"
    assert report["gate_results"]["absolute_local_source_sufficiency"] is False
    assert report["gate_results"]["material_improvement_over_scalar_scale"] is False
    np.testing.assert_allclose(
        report["family_results"]["scalar_scale"][
            "symmetric_cross_cluster_full_field_NRMSE"
        ],
        0.7714068128866686,
        rtol=0.0,
        atol=1.0e-12,
    )

    sensitivity_path = (
        ROOT
        / "results"
        / "sigma_v15b_spent_invariant_resolution_sensitivity"
        / "report.json"
    )
    if sensitivity_path.exists():
        sensitivity = json.loads(sensitivity_path.read_text(encoding="utf-8"))
        assert sensitivity["protocol_version"] == (
            "SIGMA-V15B-SPENT-INVARIANT-RESOLUTION-SENSITIVITY-1.0.0"
        )
        assert sensitivity["selected_family"] == "scalar_scale"
        sensitivity_error = sensitivity["family_results"]["scalar_scale"][
            "symmetric_cross_cluster_full_field_NRMSE"
        ]
        np.testing.assert_allclose(
            sensitivity_error,
            0.7498484801531847,
            rtol=0.0,
            atol=1.0e-12,
        )
        relative_gain = (0.7714068128866686 - sensitivity_error) / 0.7714068128866686
        assert relative_gain < 0.1
        assert sensitivity["gate_results"]["absolute_local_source_sufficiency"] is False
