#!/usr/bin/env python3
"""Infer transferable local covariant source information from spent cluster maps."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import G, c
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0715_sky_lensing_engine_validation import (
    BARYON_MAPS,
    frozen_sky_field,
    glafic_comparator,
)

from voidscreen.sigma_covariant_feature_inference import (
    FAMILY_ORDER,
    EquivariantDataset,
    MetricFeature,
    build_metric_feature_library,
    fit_equivariant_ridge,
    predict_residual,
    score_prediction,
)
from voidscreen.sigma_operator_inference import apodization_window
from voidscreen.sky_lensing import lens_invariants


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def critical_surface_density_msun_kpc2(
    lens_redshift: float,
    source_redshift: float,
) -> float:
    if not 0.0 < lens_redshift < source_redshift:
        raise ValueError("source must lie behind a positive-redshift lens")
    lens_distance = Planck18.angular_diameter_distance(lens_redshift)
    source_distance = Planck18.angular_diameter_distance(source_redshift)
    lens_source_distance = Planck18.angular_diameter_distance_z1z2(
        lens_redshift,
        source_redshift,
    )
    value = c**2 / (4.0 * np.pi * G) * source_distance / (lens_distance * lens_source_distance)
    return float(value.to_value(u.Msun / u.kpc**2))


def resample_surface(
    source_axis: np.ndarray,
    surface: np.ndarray,
    target_axis: np.ndarray,
) -> np.ndarray:
    interpolator = RegularGridInterpolator(
        (source_axis, source_axis),
        surface,
        bounds_error=True,
    )
    east, north = np.meshgrid(target_axis, target_axis)
    points = np.column_stack([north.ravel(), east.ravel()])
    return interpolator(points).reshape(east.shape)


def crop_feature(feature: MetricFeature, indices: np.ndarray) -> MetricFeature:
    block = np.ix_(indices, indices)
    return MetricFeature(
        name=feature.name,
        family=feature.family,
        convergence=feature.convergence[block],
        shear_1=feature.shear_1[block],
        shear_2=feature.shear_2[block],
    )


def sample_cluster(cluster: str, config: dict) -> tuple[EquivariantDataset, dict]:
    measurement = config["map_measurement"]
    sample = config["sample"]
    target_half = float(measurement["target_half_width_kpc"])
    context_half = float(measurement["baryon_context_half_width_kpc"])
    target_points = int(measurement["target_grid_points"])
    target_axis = np.linspace(-target_half, target_half, target_points)
    spacing = float(target_axis[1] - target_axis[0])
    context_points = round(2.0 * context_half / spacing) + 1
    context_axis = np.linspace(-context_half, context_half, context_points)
    if not np.isclose(context_axis[1] - context_axis[0], spacing, rtol=0.0, atol=1e-10):
        raise RuntimeError("context and target spacings must match")
    crop_indices = np.flatnonzero(np.abs(context_axis) <= target_half + 1.0e-9)
    if len(crop_indices) != target_points or not np.allclose(
        context_axis[crop_indices], target_axis
    ):
        raise RuntimeError("context crop must reproduce the target axis exactly")

    map_path = BARYON_MAPS / f"{cluster}_baryons.npz"
    with np.load(map_path) as data:
        lens_redshift = float(data["redshift"])
        center = SkyCoord(
            float(data["center_ra_deg"]) * u.deg,
            float(data["center_dec_deg"]) * u.deg,
        )
        source_axis = data["axis_kpc"].astype(float)
        stellar_surface = resample_surface(
            source_axis,
            data["stellar_surface_density_msun_kpc2"].astype(float),
            context_axis,
        )
        gas_surface = resample_surface(
            source_axis,
            data["gas_surface_density_msun_kpc2"].astype(float),
            context_axis,
        )

    sigma_critical = critical_surface_density_msun_kpc2(
        lens_redshift,
        float(sample["source_redshift"]),
    )
    gas_convergence = np.maximum(gas_surface / sigma_critical, 0.0)
    stellar_convergence = np.maximum(stellar_surface / sigma_critical, 0.0)
    feature_config = config["feature_library"]
    full_features = build_metric_feature_library(
        gas_convergence + stellar_convergence,
        gas_convergence,
        stellar_convergence,
        spacing_kpc=spacing,
        scales_kpc=tuple(float(value) for value in feature_config["gaussian_scales_kpc"]),
        padding_factor=int(feature_config["fourier_padding_factor"]),
        component_floor=float(feature_config["component_floor_convergence"]),
    )
    features = {feature.name: crop_feature(feature, crop_indices) for feature in full_features}

    kpc_per_arcsec = float(Planck18.kpc_proper_per_arcmin(lens_redshift).value / 60.0)
    east_kpc, north_kpc = np.meshgrid(target_axis, target_axis)
    east_arcsec = east_kpc / kpc_per_arcsec
    north_arcsec = north_kpc / kpc_per_arcsec
    fields = {
        "base": frozen_sky_field(cluster, lens_redshift, sample["base_model"]),
        "target": glafic_comparator(cluster, lens_redshift, center),
    }
    invariants = {
        name: lens_invariants(
            field,
            east_arcsec,
            north_arcsec,
            float(sample["source_redshift"]),
            step_arcsec=float(measurement["jacobian_step_arcsec"]),
        )
        for name, field in fields.items()
    }
    base = tuple(
        getattr(invariants["base"], channel) for channel in ("convergence", "shear_1", "shear_2")
    )
    target = tuple(
        getattr(invariants["target"], channel) for channel in ("convergence", "shear_1", "shear_2")
    )
    window = apodization_window(
        (target_points, target_points),
        float(measurement["tukey_alpha"]),
    )
    radius = np.hypot(east_kpc, north_kpc)
    mask = (radius <= float(measurement["analysis_radius_kpc"])) & (
        window >= float(measurement["minimum_window_weight"])
    )
    dataset = EquivariantDataset(
        name=cluster,
        mask=mask,
        base=base,
        target=target,
        features=features,
    )
    metadata = {
        "cluster": cluster,
        "lens_redshift": lens_redshift,
        "critical_surface_density_msun_kpc2": sigma_critical,
        "target_axis_kpc": target_axis,
        "mask_pixels": int(np.count_nonzero(mask)),
        "baryon_map_hash": sha256(map_path),
    }
    return dataset, metadata


def symmetric_error(scores: list[dict[str, float]]) -> float:
    return float(np.sqrt(np.mean([score["full_field_NRMSE"] ** 2 for score in scores])))


def coefficient_contributions(
    datasets: list[EquivariantDataset],
    coefficients: dict[str, float],
) -> list[dict[str, float | str]]:
    records = []
    for name, coefficient in coefficients.items():
        powers = []
        for dataset in datasets:
            feature = dataset.features[name]
            values = np.concatenate(
                [
                    feature.convergence[dataset.mask],
                    feature.shear_1[dataset.mask],
                    feature.shear_2[dataset.mask],
                ]
            )
            powers.append(float(np.mean(np.square(coefficient * values))))
        records.append(
            {
                "feature": name,
                "coefficient": float(coefficient),
                "pooled_contribution_RMS": float(np.sqrt(np.mean(powers))),
            }
        )
    return sorted(records, key=lambda row: row["pooled_contribution_RMS"], reverse=True)


def prediction_agreement_cosine(
    datasets: list[EquivariantDataset],
    first: dict[str, float],
    second: dict[str, float],
) -> float:
    """Compare two fitted operators through predictions on both baryon maps."""
    first_values = []
    second_values = []
    for dataset in datasets:
        first_prediction = predict_residual(dataset, first)
        second_prediction = predict_residual(dataset, second)
        first_values.append(np.concatenate([channel[dataset.mask] for channel in first_prediction]))
        second_values.append(
            np.concatenate([channel[dataset.mask] for channel in second_prediction])
        )
    first_vector = np.concatenate(first_values)
    second_vector = np.concatenate(second_values)
    denominator = float(np.linalg.norm(first_vector) * np.linalg.norm(second_vector))
    return float(np.dot(first_vector, second_vector) / denominator) if denominator > 0.0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer transferable covariant source features on spent cluster maps."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v15_spent_invariant_inference.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v15_spent_invariant_inference",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["status"] != (
        "frozen before constructing the full feature matrices or reading cross-cluster scores"
    ):
        raise RuntimeError("the v15 protocol is not in its frozen pre-score state")
    clusters = [sample_cluster(name, config) for name in config["sample"]["clusters"]]
    datasets = [value[0] for value in clusters]
    metadata = [value[1] for value in clusters]
    families = sorted(FAMILY_ORDER, key=FAMILY_ORDER.get)
    alpha_grid = [float(value) for value in config["fit"]["ridge_alpha_grid"]]

    family_results = {}
    for family in families:
        alpha_rows = []
        for alpha in alpha_grid:
            direction_scores = []
            for train_index, test_index in ((0, 1), (1, 0)):
                fit = fit_equivariant_ridge(
                    [datasets[train_index]],
                    family=family,
                    alpha=alpha,
                )
                score = score_prediction(datasets[test_index], fit.coefficients)
                direction_scores.append(
                    {
                        "train_cluster": datasets[train_index].name,
                        "test_cluster": datasets[test_index].name,
                        **score,
                    }
                )
            alpha_rows.append(
                {
                    "alpha": alpha,
                    "symmetric_cross_cluster_full_field_NRMSE": symmetric_error(direction_scores),
                    "directions": direction_scores,
                }
            )
        selected_alpha_row = min(
            alpha_rows,
            key=lambda row: (
                row["symmetric_cross_cluster_full_field_NRMSE"],
                row["alpha"],
            ),
        )
        selected_alpha = float(selected_alpha_row["alpha"])
        directional_fits = [
            fit_equivariant_ridge([dataset], family=family, alpha=selected_alpha)
            for dataset in datasets
        ]
        self_fit_scores = [
            {
                "cluster": dataset.name,
                **score_prediction(dataset, fit.coefficients),
            }
            for dataset, fit in zip(datasets, directional_fits, strict=True)
        ]
        joint_fit = fit_equivariant_ridge(
            datasets,
            family=family,
            alpha=selected_alpha,
        )
        family_results[family] = {
            "selected_alpha": selected_alpha,
            "symmetric_cross_cluster_full_field_NRMSE": selected_alpha_row[
                "symmetric_cross_cluster_full_field_NRMSE"
            ],
            "cross_cluster_scores": selected_alpha_row["directions"],
            "self_fit_scores": self_fit_scores,
            "directional_prediction_agreement_cosine": prediction_agreement_cosine(
                datasets,
                directional_fits[0].coefficients,
                directional_fits[1].coefficients,
            ),
            "alpha_sweep": alpha_rows,
            "joint_fit_scores": [
                {"cluster": dataset.name, **score_prediction(dataset, joint_fit.coefficients)}
                for dataset in datasets
            ],
            "joint_fit_condition_number": joint_fit.standardized_condition_number,
            "joint_coefficients": joint_fit.coefficients,
            "ranked_joint_contributions": coefficient_contributions(
                datasets,
                joint_fit.coefficients,
            ),
        }

    selected_family = min(
        families,
        key=lambda family: (
            family_results[family]["symmetric_cross_cluster_full_field_NRMSE"],
            FAMILY_ORDER[family],
        ),
    )
    scalar_error = family_results["scalar_scale"]["symmetric_cross_cluster_full_field_NRMSE"]
    total_error = family_results["total_tidal"]["symmetric_cross_cluster_full_field_NRMSE"]
    component_error = family_results["component_overlap"][
        "symmetric_cross_cluster_full_field_NRMSE"
    ]
    gates = config["diagnostic_gates"]
    absolute_source_sufficiency = bool(
        family_results[selected_family]["symmetric_cross_cluster_full_field_NRMSE"]
        <= gates["maximum_symmetric_cross_cluster_full_field_NRMSE_for_local_source_sufficiency"]
    )
    improvement_over_scalar = float(
        (scalar_error - min(total_error, component_error)) / scalar_error
    )
    total_improvement = float((scalar_error - total_error) / scalar_error)
    component_improvement_over_best_noncomponent = float(
        (min(scalar_error, total_error) - component_error) / min(scalar_error, total_error)
    )
    selected_cross_scores = family_results[selected_family]["cross_cluster_scores"]
    power_gate = bool(
        all(
            row["residual_power_closed"] >= gates["minimum_cross_cluster_residual_power_closed"]
            for row in selected_cross_scores
        )
    )
    shear_gate = bool(
        all(
            row["residual_shear_alignment_cosine"]
            >= gates["minimum_cross_cluster_residual_shear_alignment_cosine"]
            for row in selected_cross_scores
        )
    )
    margin = float(gates["minimum_relative_cross_cluster_improvement_over_scalar_scale"])
    if component_improvement_over_best_noncomponent >= margin:
        inferred_requirement = (
            "component-resolved gas-star overlap/orientation carries transferable information; "
            "the next covariant source must generate it without treating components as adjustable material labels"
        )
    elif total_improvement >= margin:
        inferred_requirement = "nonlinear total-baryon tidal invariants carry transferable information; a component label is not yet required"
    else:
        inferred_requirement = (
            "the tested local scalar/tidal/component invariants do not transfer materially beyond scale-only baryons; "
            "the next mechanism must add wider environment, dynamics, or causal state"
        )
    if not absolute_source_sufficiency or not power_gate:
        inferred_requirement += "; the local library is quantitatively insufficient and cannot be promoted as a weak-field closure"

    selected_coefficients = family_results[selected_family]["joint_coefficients"]
    figure, axes = plt.subplots(3, 3, figsize=(14, 12), constrained_layout=True)
    for row, dataset in enumerate(datasets):
        target_residual = tuple(
            target - base for target, base in zip(dataset.target, dataset.base, strict=True)
        )
        prediction = predict_residual(dataset, selected_coefficients)
        extent = [-350.0, 350.0, -350.0, 350.0]
        maximum = float(np.nanpercentile(np.abs(target_residual[0][dataset.mask]), 98.0))
        maps = (
            target_residual[0],
            prediction[0],
            prediction[0] - target_residual[0],
        )
        titles = ("required delta kappa", "predicted delta kappa", "prediction error")
        for column, (values, title) in enumerate(zip(maps, titles, strict=True)):
            image = axes[row, column].imshow(
                values,
                origin="lower",
                extent=extent,
                cmap="coolwarm",
                vmin=-maximum,
                vmax=maximum,
            )
            axes[row, column].set_title(f"{dataset.name}: {title}")
            axes[row, column].set(xlabel="east kpc", ylabel="north kpc")
            figure.colorbar(image, ax=axes[row, column], shrink=0.75)
    errors = [
        family_results[family]["symmetric_cross_cluster_full_field_NRMSE"] for family in families
    ]
    axes[2, 0].bar(families, errors, color=["#777777", "#4477aa", "#228833"])
    axes[2, 0].axhline(
        gates["maximum_symmetric_cross_cluster_full_field_NRMSE_for_local_source_sufficiency"],
        color="black",
        linestyle="--",
        label="source-sufficiency gate",
    )
    axes[2, 0].set(ylabel="symmetric cross-cluster NRMSE", title="Spent transfer test")
    axes[2, 0].tick_params(axis="x", rotation=20)
    axes[2, 0].legend(fontsize=8)
    top = family_results[selected_family]["ranked_joint_contributions"][:8]
    axes[2, 1].barh(
        [row["feature"] for row in reversed(top)],
        [row["pooled_contribution_RMS"] for row in reversed(top)],
        color="#aa3377",
    )
    axes[2, 1].set(xlabel="pooled triplet contribution RMS", title="Leading joint-fit features")
    alignment = [row["residual_shear_alignment_cosine"] for row in selected_cross_scores]
    power = [row["residual_power_closed"] for row in selected_cross_scores]
    labels = [f"{row['train_cluster']} to {row['test_cluster']}" for row in selected_cross_scores]
    positions = np.arange(len(labels))
    axes[2, 2].bar(positions - 0.18, alignment, width=0.36, label="shear alignment")
    axes[2, 2].bar(positions + 0.18, power, width=0.36, label="residual power closed")
    axes[2, 2].axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    axes[2, 2].set_xticks(positions, labels, rotation=15)
    axes[2, 2].set_ylim(min(-0.2, min(power) - 0.1), 1.0)
    axes[2, 2].set(title=f"{selected_family} cross-transfer", ylabel="dimensionless")
    axes[2, 2].legend(fontsize=8)

    args.output.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output / "spent_invariant_inference.png", dpi=180)
    plt.close(figure)
    report = {
        "status": "completed Sigma v15 spent covariant-invariant inference",
        "protocol_version": config["protocol_version"],
        "sample_is_spent": True,
        "observational_validation_claim": False,
        "per_cluster_gravity_parameters": 0,
        "one_metric_feature_triplets": True,
        "input_hashes": {
            "config": sha256(args.config),
            **{f"{row['cluster']}_baryon_map": row["baryon_map_hash"] for row in metadata},
        },
        "sample_metadata": [
            {
                key: value
                for key, value in row.items()
                if key not in {"target_axis_kpc", "baryon_map_hash"}
            }
            for row in metadata
        ],
        "base_AQUAL_scores": [
            {"cluster": dataset.name, **score_prediction(dataset, {})} for dataset in datasets
        ],
        "feature_counts": {
            family: len(
                [
                    feature
                    for feature in datasets[0].features.values()
                    if FAMILY_ORDER[feature.family] <= FAMILY_ORDER[family]
                ]
            )
            for family in families
        },
        "family_results": family_results,
        "selected_family": selected_family,
        "relative_improvements": {
            "best_non_scalar_over_scalar_scale": improvement_over_scalar,
            "total_tidal_over_scalar_scale": total_improvement,
            "component_overlap_over_best_noncomponent": component_improvement_over_best_noncomponent,
        },
        "gate_results": {
            "absolute_local_source_sufficiency": absolute_source_sufficiency,
            "minimum_residual_power_closed_both_directions": power_gate,
            "minimum_residual_shear_alignment_both_directions": shear_gate,
            "material_improvement_over_scalar_scale": improvement_over_scalar >= margin,
        },
        "inferred_action_requirement": inferred_requirement,
        "claim_boundary": config["claim_boundary"],
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["gate_results"], indent=2, sort_keys=True))
    print(json.dumps(report["relative_improvements"], indent=2, sort_keys=True))
    for family in families:
        result = family_results[family]
        print(
            f"{family}: alpha={result['selected_alpha']:g}, "
            f"cross_NRMSE={result['symmetric_cross_cluster_full_field_NRMSE']:.6f}"
        )
    print(f"selected={selected_family}")
    print(inferred_requirement)


if __name__ == "__main__":
    main()
