#!/usr/bin/env python3
"""Test whether measured outer baryons predict the spent missing cluster shear."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import pairwise
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import G, c
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0715_sky_lensing_engine_validation import (
    BARYON_MAPS,
    frozen_sky_field,
    glafic_comparator,
)

from voidscreen.sigma_boundary_inference import (
    BoundaryDecomposition,
    decompose_boundary_shear,
    harmonic_shear_basis,
    shear_alignment_and_power_closed,
)
from voidscreen.sigma_covariant_feature_inference import (
    EquivariantDataset,
    MetricFeature,
    convergence_to_shear,
    fit_equivariant_ridge_features,
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
    lens_distance = Planck18.angular_diameter_distance(lens_redshift)
    source_distance = Planck18.angular_diameter_distance(source_redshift)
    lens_source_distance = Planck18.angular_diameter_distance_z1z2(
        lens_redshift,
        source_redshift,
    )
    value = c**2 / (4.0 * np.pi * G) * source_distance / (lens_distance * lens_source_distance)
    return float(value.to_value(u.Msun / u.kpc**2))


def resample_grid(
    source_axis: np.ndarray,
    values: np.ndarray,
    target_axis: np.ndarray,
) -> np.ndarray:
    interpolator = RegularGridInterpolator(
        (source_axis, source_axis),
        values,
        bounds_error=True,
    )
    east, north = np.meshgrid(target_axis, target_axis)
    points = np.column_stack([north.ravel(), east.ravel()])
    return interpolator(points).reshape(east.shape)


def metric_feature_from_full_convergence(
    name: str,
    convergence: np.ndarray,
    source_axis: np.ndarray,
    target_axis: np.ndarray,
    *,
    padding_factor: int,
) -> MetricFeature:
    shear_1, shear_2 = convergence_to_shear(
        convergence,
        padding_factor=padding_factor,
    )
    return MetricFeature(
        name=name,
        family="scalar_scale",
        convergence=resample_grid(source_axis, convergence, target_axis),
        shear_1=resample_grid(source_axis, shear_1, target_axis),
        shear_2=resample_grid(source_axis, shear_2, target_axis),
    )


def feature_names(config: dict) -> dict[str, list[str]]:
    scales = [float(value) for value in config["feature_library"]["internal_gaussian_scales_kpc"]]
    edges = [float(value) for value in config["feature_library"]["outer_annulus_edges_kpc"]]
    internal = [f"internal_total_smooth_{scale:g}kpc" for scale in scales]
    total_boundary = [f"boundary_total_{lower:g}_{upper:g}kpc" for lower, upper in pairwise(edges)]
    contrast_boundary = [
        f"boundary_gas_minus_star_{lower:g}_{upper:g}kpc" for lower, upper in pairwise(edges)
    ]
    return {
        "internal_only": internal,
        "boundary_total": internal + total_boundary,
        "boundary_components": internal + total_boundary + contrast_boundary,
    }


def build_baryon_features(
    source_axis: np.ndarray,
    gas_convergence: np.ndarray,
    stellar_convergence: np.ndarray,
    target_axis: np.ndarray,
    config: dict,
) -> dict[str, MetricFeature]:
    feature_config = config["feature_library"]
    padding_factor = int(feature_config["fourier_padding_factor"])
    spacing = float(source_axis[1] - source_axis[0])
    east, north = np.meshgrid(source_axis, source_axis)
    radius = np.hypot(east, north)
    analysis_radius = float(config["map_measurement"]["analysis_radius_kpc"])
    total = gas_convergence + stellar_convergence
    internal_total = np.where(radius <= analysis_radius, total, 0.0)
    features: dict[str, MetricFeature] = {}
    for scale in feature_config["internal_gaussian_scales_kpc"]:
        scale = float(scale)
        smoothed = gaussian_filter(
            internal_total,
            scale / spacing,
            mode="constant",
        )
        feature = metric_feature_from_full_convergence(
            f"internal_total_smooth_{scale:g}kpc",
            smoothed,
            source_axis,
            target_axis,
            padding_factor=padding_factor,
        )
        features[feature.name] = feature

    edges = [float(value) for value in feature_config["outer_annulus_edges_kpc"]]
    for lower, upper in pairwise(edges):
        annulus = (radius >= lower) & (radius < upper)
        total_feature = metric_feature_from_full_convergence(
            f"boundary_total_{lower:g}_{upper:g}kpc",
            np.where(annulus, total, 0.0),
            source_axis,
            target_axis,
            padding_factor=padding_factor,
        )
        contrast_feature = metric_feature_from_full_convergence(
            f"boundary_gas_minus_star_{lower:g}_{upper:g}kpc",
            np.where(annulus, gas_convergence - stellar_convergence, 0.0),
            source_axis,
            target_axis,
            padding_factor=padding_factor,
        )
        features[total_feature.name] = total_feature
        features[contrast_feature.name] = contrast_feature
    return features


def sample_cluster(
    cluster: str,
    config: dict,
) -> tuple[EquivariantDataset, dict, dict[int, BoundaryDecomposition]]:
    measurement = config["map_measurement"]
    sample = config["sample"]
    half_width = float(measurement["target_half_width_kpc"])
    points = int(measurement["target_grid_points"])
    target_axis = np.linspace(-half_width, half_width, points)
    east_kpc, north_kpc = np.meshgrid(target_axis, target_axis)
    radius_kpc = np.hypot(east_kpc, north_kpc)

    map_path = BARYON_MAPS / f"{cluster}_baryons.npz"
    with np.load(map_path) as data:
        lens_redshift = float(data["redshift"])
        center = SkyCoord(
            float(data["center_ra_deg"]) * u.deg,
            float(data["center_dec_deg"]) * u.deg,
        )
        source_axis = data["axis_kpc"].astype(float)
        gas_surface = data["gas_surface_density_msun_kpc2"].astype(float)
        stellar_surface = data["stellar_surface_density_msun_kpc2"].astype(float)
    declared_radius = float(measurement["baryon_map_radius_kpc"])
    if not np.isclose(source_axis[0], -declared_radius) or not np.isclose(
        source_axis[-1], declared_radius
    ):
        raise RuntimeError("registered baryon-map radius changed")
    sigma_critical = critical_surface_density_msun_kpc2(
        lens_redshift,
        float(sample["source_redshift"]),
    )
    features = build_baryon_features(
        source_axis,
        gas_surface / sigma_critical,
        stellar_surface / sigma_critical,
        target_axis,
        config,
    )
    expected_names = set(feature_names(config)["boundary_components"])
    if set(features) != expected_names:
        raise RuntimeError("constructed boundary feature inventory changed")

    kpc_per_arcsec = float(Planck18.kpc_proper_per_arcmin(lens_redshift).value / 60.0)
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
        (points, points),
        float(measurement["tukey_alpha"]),
    )
    mask = (radius_kpc <= float(measurement["analysis_radius_kpc"])) & (
        window >= float(measurement["minimum_window_weight"])
    )
    dataset = EquivariantDataset(
        name=cluster,
        mask=mask,
        base=base,
        target=target,
        features=features,
    )

    missing = tuple(
        target_channel - base_channel
        for target_channel, base_channel in zip(target, base, strict=True)
    )
    decomposition_config = config["boundary_decomposition"]
    basis = harmonic_shear_basis(
        east_kpc,
        north_kpc,
        minimum_order=int(decomposition_config["harmonic_minimum_order"]),
        maximum_order=int(decomposition_config["harmonic_maximum_order"]),
        reference_radius_kpc=float(decomposition_config["harmonic_reference_radius_kpc"]),
    )
    padding_factors = [
        int(decomposition_config["primary_fourier_padding_factor"]),
        *[int(value) for value in decomposition_config["padding_sensitivity_factors"]],
    ]
    decompositions = {
        factor: decompose_boundary_shear(
            missing[0],
            missing[1],
            missing[2],
            radius_kpc,
            mask,
            basis,
            taper_start_kpc=float(measurement["internal_taper_start_kpc"]),
            taper_end_kpc=float(
                measurement.get(
                    "internal_taper_end_kpc",
                    measurement["analysis_radius_kpc"],
                )
            ),
            padding_factor=factor,
        )
        for factor in padding_factors
    }
    metadata = {
        "cluster": cluster,
        "lens_redshift": lens_redshift,
        "critical_surface_density_msun_kpc2": sigma_critical,
        "mask_pixels": int(np.count_nonzero(mask)),
        "baryon_map_hash": sha256(map_path),
        "target_axis_kpc": target_axis,
    }
    return dataset, metadata, decompositions


def symmetric_error(scores: list[dict[str, float]]) -> float:
    return float(np.sqrt(np.mean([row["full_field_NRMSE"] ** 2 for row in scores])))


def boundary_prediction(
    dataset: EquivariantDataset,
    coefficients: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    boundary_coefficients = {
        name: value for name, value in coefficients.items() if name.startswith("boundary_")
    }
    prediction = predict_residual(dataset, boundary_coefficients)
    return prediction[1], prediction[2]


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer the spent internal-versus-boundary cluster Weyl field."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v16_spent_boundary_decomposition.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v16_spent_boundary_decomposition",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["status"] != (
        "frozen before computing the required harmonic field or cross-cluster boundary scores"
    ):
        raise RuntimeError("the v16 protocol is not in its frozen pre-score state")
    measurement = config["map_measurement"]
    taper_separation = float(measurement["internal_taper_start_kpc"]) - float(
        measurement["analysis_radius_kpc"]
    )
    required_taper_separation = float(
        measurement.get("minimum_taper_start_separation_from_score_kpc", 0.0)
    )
    taper_separation_pass = bool(taper_separation >= required_taper_separation)
    if "minimum_taper_start_separation_from_score_kpc" in measurement and not (
        taper_separation_pass
    ):
        raise RuntimeError("the convergence taper intrudes on the scored boundary disk")

    sampled = [sample_cluster(cluster, config) for cluster in config["sample"]["clusters"]]
    datasets = [row[0] for row in sampled]
    metadata = [row[1] for row in sampled]
    all_decompositions = [row[2] for row in sampled]
    primary_padding = int(config["boundary_decomposition"]["primary_fourier_padding_factor"])
    primary_decompositions = [values[primary_padding] for values in all_decompositions]
    families = feature_names(config)
    alpha_grid = [float(value) for value in config["fit"]["ridge_alpha_grid"]]

    family_results = {}
    for family, names in families.items():
        alpha_rows = []
        for alpha in alpha_grid:
            direction_scores = []
            for train_index, test_index in ((0, 1), (1, 0)):
                fit = fit_equivariant_ridge_features(
                    [datasets[train_index]],
                    feature_names=names,
                    alpha=alpha,
                )
                full_score = score_prediction(datasets[test_index], fit.coefficients)
                boundary_1, boundary_2 = boundary_prediction(
                    datasets[test_index],
                    fit.coefficients,
                )
                target_boundary = primary_decompositions[test_index]
                boundary_score = shear_alignment_and_power_closed(
                    boundary_1,
                    boundary_2,
                    target_boundary.boundary_shear_1,
                    target_boundary.boundary_shear_2,
                    datasets[test_index].mask,
                )
                direction_scores.append(
                    {
                        "train_cluster": datasets[train_index].name,
                        "test_cluster": datasets[test_index].name,
                        **full_score,
                        **boundary_score,
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
        joint_fit = fit_equivariant_ridge_features(
            datasets,
            feature_names=names,
            alpha=selected_alpha,
        )
        family_results[family] = {
            "feature_names": names,
            "selected_alpha": selected_alpha,
            "symmetric_cross_cluster_full_field_NRMSE": selected_alpha_row[
                "symmetric_cross_cluster_full_field_NRMSE"
            ],
            "cross_cluster_scores": selected_alpha_row["directions"],
            "alpha_sweep": alpha_rows,
            "joint_fit_scores": [
                {"cluster": dataset.name, **score_prediction(dataset, joint_fit.coefficients)}
                for dataset in datasets
            ],
            "joint_coefficients": joint_fit.coefficients,
            "joint_fit_condition_number": joint_fit.standardized_condition_number,
            "ranked_joint_contributions": coefficient_contributions(
                datasets,
                joint_fit.coefficients,
            ),
        }

    selected_family = min(
        families,
        key=lambda family: family_results[family]["symmetric_cross_cluster_full_field_NRMSE"],
    )
    internal_error = family_results["internal_only"]["symmetric_cross_cluster_full_field_NRMSE"]
    best_boundary_family = min(
        ("boundary_total", "boundary_components"),
        key=lambda family: family_results[family]["symmetric_cross_cluster_full_field_NRMSE"],
    )
    best_boundary_result = family_results[best_boundary_family]
    relative_boundary_improvement = float(
        (internal_error - best_boundary_result["symmetric_cross_cluster_full_field_NRMSE"])
        / internal_error
    )

    decomposition_rows = []
    for dataset, decompositions in zip(datasets, all_decompositions, strict=True):
        primary = decompositions[primary_padding]
        sensitivities = {
            str(factor): {
                "harmonic_oracle_NRMSE": value.harmonic_fit.normalized_RMSE,
                "harmonic_oracle_power_closed": value.harmonic_fit.power_closed,
                "boundary_to_total_shear_power_ratio": value.boundary_to_total_shear_power_ratio,
            }
            for factor, value in sorted(decompositions.items())
        }
        power_values = [row["harmonic_oracle_power_closed"] for row in sensitivities.values()]
        decomposition_rows.append(
            {
                "cluster": dataset.name,
                "primary_padding_factor": primary_padding,
                "boundary_to_total_shear_power_ratio": primary.boundary_to_total_shear_power_ratio,
                "harmonic_oracle_NRMSE": primary.harmonic_fit.normalized_RMSE,
                "harmonic_oracle_power_closed": primary.harmonic_fit.power_closed,
                "harmonic_coefficients": primary.harmonic_fit.coefficients,
                "padding_sensitivity": sensitivities,
                "padding_power_closed_span": float(max(power_values) - min(power_values)),
            }
        )

    gates = config["diagnostic_gates"]
    harmonic_gate = bool(
        all(
            row["harmonic_oracle_power_closed"]
            >= gates["minimum_harmonic_oracle_boundary_power_closed_each_cluster"]
            for row in decomposition_rows
        )
    )
    padding_gate = bool(
        all(
            row["padding_power_closed_span"]
            <= gates["maximum_padding_sensitivity_in_harmonic_power_closed"]
            for row in decomposition_rows
        )
    )
    cross_scores = best_boundary_result["cross_cluster_scores"]
    alignment_gate = bool(
        all(
            row["boundary_shear_alignment_cosine"]
            >= gates["minimum_cross_cluster_boundary_shear_alignment_cosine"]
            for row in cross_scores
        )
    )
    boundary_power_gate = bool(
        all(
            row["boundary_shear_power_closed"]
            >= gates["minimum_cross_cluster_boundary_shear_power_closed"]
            for row in cross_scores
        )
    )
    improvement_gate = bool(
        relative_boundary_improvement
        >= gates["minimum_relative_cross_cluster_improvement_over_internal_only"]
    )
    absolute_gate = bool(
        best_boundary_result["symmetric_cross_cluster_full_field_NRMSE"]
        <= gates["maximum_symmetric_cross_cluster_full_field_NRMSE_for_boundary_source_sufficiency"]
    )
    measured_boundary_pass = bool(
        improvement_gate and absolute_gate and alignment_gate and boundary_power_gate
    )
    if harmonic_gate and padding_gate and measured_boundary_pass:
        inferred_requirement = "measured outer baryons predict a transferable harmonic field; the next action must derive a covariant baryonic boundary propagator before any holdout"
    elif harmonic_gate and padding_gate:
        inferred_requirement = "a robust harmonic boundary component exists in the spent target, but the available projected outer baryons do not predict it universally; obtain wider and line-of-sight environmental data before selecting another carrier"
    else:
        inferred_requirement = "the finite-window harmonic boundary interpretation is not robust enough to justify a static boundary mechanism; formulate a baryon-unique dynamical-state question and enumerate velocity, shock, and relaxation data"

    boundary_figure_coefficients = family_results[best_boundary_family]["joint_coefficients"]
    figure, axes = plt.subplots(3, 3, figsize=(14, 12), constrained_layout=True)
    for row, (dataset, decomposition) in enumerate(
        zip(datasets, primary_decompositions, strict=True)
    ):
        predicted_boundary = boundary_prediction(dataset, boundary_figure_coefficients)
        maps = (
            np.hypot(decomposition.boundary_shear_1, decomposition.boundary_shear_2),
            np.hypot(
                decomposition.harmonic_fit.predicted_shear_1,
                decomposition.harmonic_fit.predicted_shear_2,
            ),
            np.hypot(predicted_boundary[0], predicted_boundary[1]),
        )
        titles = (
            "required boundary-shear magnitude",
            "harmonic-oracle magnitude",
            f"{best_boundary_family} boundary magnitude",
        )
        maximum = float(np.nanpercentile(maps[0][dataset.mask], 98.0))
        for column, (values, title) in enumerate(zip(maps, titles, strict=True)):
            image = axes[row, column].imshow(
                values,
                origin="lower",
                extent=[-350, 350, -350, 350],
                cmap="magma",
                vmin=0.0,
                vmax=maximum,
            )
            axes[row, column].set_title(f"{dataset.name}: {title}")
            axes[row, column].set(xlabel="east kpc", ylabel="north kpc")
            figure.colorbar(image, ax=axes[row, column], shrink=0.75)
    family_labels = list(families)
    family_errors = [
        family_results[family]["symmetric_cross_cluster_full_field_NRMSE"]
        for family in family_labels
    ]
    axes[2, 0].bar(family_labels, family_errors, color=["#777777", "#4477aa", "#228833"])
    axes[2, 0].axhline(
        gates["maximum_symmetric_cross_cluster_full_field_NRMSE_for_boundary_source_sufficiency"],
        color="black",
        linestyle="--",
    )
    axes[2, 0].set(ylabel="symmetric cross-cluster NRMSE", title="Boundary transfer")
    axes[2, 0].tick_params(axis="x", rotation=18)
    axes[2, 1].bar(
        [row["cluster"] for row in decomposition_rows],
        [row["harmonic_oracle_power_closed"] for row in decomposition_rows],
        color="#aa3377",
    )
    axes[2, 1].axhline(
        gates["minimum_harmonic_oracle_boundary_power_closed_each_cluster"],
        color="black",
        linestyle="--",
    )
    axes[2, 1].set(ylim=(0.0, 1.0), ylabel="boundary power closed", title="Harmonic upper bound")
    positions = np.arange(len(cross_scores))
    axes[2, 2].bar(
        positions - 0.18,
        [row["boundary_shear_alignment_cosine"] for row in cross_scores],
        width=0.36,
        label="alignment",
    )
    axes[2, 2].bar(
        positions + 0.18,
        [row["boundary_shear_power_closed"] for row in cross_scores],
        width=0.36,
        label="power closed",
    )
    axes[2, 2].set_xticks(
        positions,
        [f"{row['train_cluster']} to {row['test_cluster']}" for row in cross_scores],
        rotation=15,
    )
    axes[2, 2].axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    axes[2, 2].set(title=f"{best_boundary_family} boundary transfer", ylabel="dimensionless")
    axes[2, 2].legend(fontsize=8)

    args.output.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output / "spent_boundary_decomposition.png", dpi=180)
    plt.close(figure)
    report = {
        "status": "completed Sigma v16 spent boundary decomposition",
        "protocol_version": config["protocol_version"],
        "sample_is_spent": True,
        "observational_validation_claim": False,
        "per_cluster_gravity_parameters": 0,
        "per_cluster_shear_or_orientation_parameters": 0,
        "one_metric_feature_triplets": True,
        "decomposition_integrity": {
            "taper_start_minus_analysis_radius_kpc": taper_separation,
            "required_minimum_separation_kpc": required_taper_separation,
            "passes_declared_separation": taper_separation_pass,
        },
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
        "decomposition_results": decomposition_rows,
        "base_AQUAL_scores": [
            {"cluster": dataset.name, **score_prediction(dataset, {})} for dataset in datasets
        ],
        "family_results": family_results,
        "selected_family": selected_family,
        "best_boundary_family": best_boundary_family,
        "relative_boundary_improvement_over_internal_only": relative_boundary_improvement,
        "gate_results": {
            "harmonic_oracle_each_cluster": harmonic_gate,
            "harmonic_padding_robustness": padding_gate,
            "boundary_material_improvement": improvement_gate,
            "boundary_absolute_source_sufficiency": absolute_gate,
            "boundary_alignment_both_directions": alignment_gate,
            "boundary_power_closed_both_directions": boundary_power_gate,
            "measured_outer_baryon_boundary_source": measured_boundary_pass,
        },
        "inferred_action_requirement": inferred_requirement,
        "claim_boundary": config["claim_boundary"],
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["gate_results"], indent=2, sort_keys=True))
    for row in decomposition_rows:
        print(
            f"{row['cluster']}: boundary/total={row['boundary_to_total_shear_power_ratio']:.4f}, "
            f"harmonic_power={row['harmonic_oracle_power_closed']:.4f}, "
            f"padding_span={row['padding_power_closed_span']:.4f}"
        )
    for family in families:
        result = family_results[family]
        print(
            f"{family}: alpha={result['selected_alpha']:g}, "
            f"cross_NRMSE={result['symmetric_cross_cluster_full_field_NRMSE']:.6f}"
        )
    print(f"selected={selected_family}; best_boundary={best_boundary_family}")
    print(inferred_requirement)


if __name__ == "__main__":
    main()
