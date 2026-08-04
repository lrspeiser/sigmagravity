#!/usr/bin/env python3
"""Run the frozen v17E target-gated thermal-stress transfer diagnostic."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_sigma_v17d_thermal_stress_maps import feature_inventory
from infer_sigma_v16_spent_boundary import feature_names, sample_cluster

from voidscreen.sigma_covariant_feature_inference import (
    EquivariantDataset,
    MetricFeature,
    fit_equivariant_ridge_features,
    predict_residual,
    score_prediction,
)

DEFAULT_CONFIG = ROOT / "configs" / "sigma_v17e_thermal_stress_transfer.json"
DEFAULT_THERMAL_REPORT = ROOT / "results" / "sigma_v17d_thermal_stress_maps" / "report.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v17e_thermal_stress_transfer"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def with_added_base(
    dataset: EquivariantDataset,
    addition: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> EquivariantDataset:
    return EquivariantDataset(
        name=dataset.name,
        mask=dataset.mask,
        base=tuple(
            base + extra for base, extra in zip(dataset.base, addition, strict=True)
        ),
        target=dataset.target,
        features=dataset.features,
    )


def symmetric_error(scores: list[dict[str, Any]]) -> float:
    return float(
        np.sqrt(np.mean([float(row["full_field_NRMSE"]) ** 2 for row in scores]))
    )


def thermal_feature_names(config: dict[str, Any], family: str) -> list[str]:
    families = config["thermal_features"]["families"]
    if family not in families:
        raise ValueError(f"unknown thermal feature family: {family}")
    scales = [float(value) for value in config["thermal_features"]["gaussian_scales_kpc"]]
    return [
        f"{source}_smooth_{scale:g}kpc"
        for source in families[family]
        for scale in scales
    ]


def validate_authorization(
    config_path: Path,
    config: dict[str, Any],
    thermal_report_path: Path,
    thermal_report: dict[str, Any],
) -> dict[str, Path]:
    expected_status = (
        "frozen after the thermal-source protocol and before any regional temperature "
        "result, thermal source map, v17 inverse coefficient, or v17 lensing score existed"
    )
    if config.get("status") != expected_status:
        raise RuntimeError("the v17E transfer protocol is not in its frozen state")
    if not config_path.is_file() or not thermal_report_path.is_file():
        raise RuntimeError("a frozen v17E input is missing")
    for path_key, hash_key in (
        ("dynamical_stress_gate", "dynamical_stress_gate_sha256"),
        ("static_baseline_config", "static_baseline_config_sha256"),
        ("static_incremental_control_report", "static_incremental_control_report_sha256"),
        ("thermal_source_protocol", "thermal_source_protocol_sha256"),
    ):
        path = ROOT / config["parents"][path_key]
        if sha256(path) != config["parents"][hash_key]:
            raise RuntimeError(f"frozen v17E parent changed: {path_key}")

    authorization = config["authorization"]
    if thermal_report.get("status") != authorization["required_thermal_source_status"]:
        raise RuntimeError("both target-blind thermal source maps are not frozen")
    for flag, expected in authorization["required_source_flags"].items():
        if thermal_report.get(flag) is not expected:
            raise RuntimeError(f"thermal source authorization flag failed: {flag}")
    if thermal_report.get("config_sha256") != config["parents"][
        "thermal_source_protocol_sha256"
    ]:
        raise RuntimeError("thermal source maps used another source protocol")

    static_report_path = ROOT / config["parents"]["static_incremental_control_report"]
    static_report = json.loads(static_report_path.read_text(encoding="utf-8"))
    expected_static = float(
        config["static_baseline"][
            "expected_symmetric_cross_cluster_full_field_NRMSE"
        ]
    )
    if not np.isclose(
        float(static_report["preserved_internal_symmetric_NRMSE"]),
        expected_static,
        rtol=0.0,
        atol=float(config["static_baseline"]["maximum_reproduction_error"]),
    ):
        raise RuntimeError("the frozen v16D preserved-internal reference changed")

    cluster_rows = {row["cluster"]: row for row in thermal_report.get("clusters", [])}
    expected_clusters = list(config["sample"]["clusters"])
    if set(cluster_rows) != set(expected_clusters):
        raise RuntimeError("thermal source cluster inventory changed")
    products: dict[str, Path] = {}
    for cluster in expected_clusters:
        row = cluster_rows[cluster]
        product = ROOT / row["product"]
        if not product.is_file() or sha256(product) != row["product_sha256"]:
            raise RuntimeError(f"frozen thermal source product changed: {cluster}")
        products[cluster] = product
    return products


def _thermal_arrays_at_axis(
    product: Path,
    target_axis: np.ndarray,
    thermal_map_config: dict[str, Any],
) -> dict[str, np.ndarray]:
    with np.load(product) as data:
        stored_axis = data["target_axis_kpc"].astype(float)
        if stored_axis.shape == target_axis.shape and np.allclose(
            stored_axis,
            target_axis,
            rtol=0.0,
            atol=1.0e-10,
        ):
            return {name: data[name].astype(float) for name in data.files}
        source_axis = data["source_axis_kpc"].astype(float)
        q_total = data["q_total"].astype(float)
        q_contrast = data["q_contrast"].astype(float)

    resolution_config = copy.deepcopy(thermal_map_config)
    feature_config = resolution_config["feature_construction"]
    feature_config["target_half_width_kpc"] = float(max(abs(target_axis[0]), abs(target_axis[-1])))
    feature_config["target_grid_points"] = int(target_axis.size)
    arrays = feature_inventory(
        source_axis,
        q_total,
        q_contrast,
        resolution_config,
    )
    if not np.allclose(arrays["target_axis_kpc"], target_axis, rtol=0.0, atol=1.0e-10):
        raise RuntimeError("recomputed thermal feature grid does not match the lensing grid")
    return arrays


def load_thermal_features(
    product: Path,
    target_axis: np.ndarray,
    config: dict[str, Any],
    thermal_map_config: dict[str, Any],
) -> dict[str, MetricFeature]:
    arrays = _thermal_arrays_at_axis(product, target_axis, thermal_map_config)
    all_sources = sorted(
        {
            source
            for sources in config["thermal_features"]["families"].values()
            for source in sources
        }
    )
    scales = [float(value) for value in config["thermal_features"]["gaussian_scales_kpc"]]
    features: dict[str, MetricFeature] = {}
    for source in all_sources:
        for scale in scales:
            name = f"{source}_smooth_{scale:g}kpc"
            keys = [f"{name}_{channel}" for channel in ("convergence", "shear_1", "shear_2")]
            if any(key not in arrays for key in keys):
                raise RuntimeError(f"thermal product is missing one-metric triplet: {name}")
            features[name] = MetricFeature(
                name=name,
                family="scalar_scale",
                convergence=arrays[keys[0]],
                shear_1=arrays[keys[1]],
                shear_2=arrays[keys[2]],
            )
    expected = set(thermal_feature_names(config, "thermal_component"))
    if set(features) != expected:
        raise RuntimeError("thermal feature inventory changed")
    return features


def build_datasets(
    config: dict[str, Any],
    base_config: dict[str, Any],
    thermal_map_config: dict[str, Any],
    products: dict[str, Path],
    grid_points: int,
) -> list[EquivariantDataset]:
    resolution_base = copy.deepcopy(base_config)
    resolution_base["map_measurement"]["target_grid_points"] = int(grid_points)
    datasets = []
    for cluster in config["sample"]["clusters"]:
        dataset, metadata, _ = sample_cluster(cluster, resolution_base)
        target_axis = np.asarray(metadata["target_axis_kpc"], dtype=float)
        thermal = load_thermal_features(
            products[cluster],
            target_axis,
            config,
            thermal_map_config,
        )
        overlap = set(dataset.features).intersection(thermal)
        if overlap:
            raise RuntimeError(f"thermal and static feature names overlap: {sorted(overlap)}")
        datasets.append(
            EquivariantDataset(
                name=dataset.name,
                mask=dataset.mask,
                base=dataset.base,
                target=dataset.target,
                features={**dataset.features, **thermal},
            )
        )
    return datasets


def prepare_static_baseline(
    datasets: list[EquivariantDataset],
    base_config: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[list[EquivariantDataset]]]:
    internal_names = feature_names(base_config)["internal_only"]
    alpha = float(config["static_baseline"]["internal_ridge_alpha"])
    direction_rows = []
    adjusted_by_direction = []
    for train_index, test_index in ((0, 1), (1, 0)):
        fit = fit_equivariant_ridge_features(
            [datasets[train_index]],
            feature_names=internal_names,
            alpha=alpha,
        )
        adjusted = [
            with_added_base(dataset, predict_residual(dataset, fit.coefficients))
            for dataset in datasets
        ]
        direction_rows.append(
            {
                "train_cluster": datasets[train_index].name,
                "test_cluster": datasets[test_index].name,
                "internal_coefficients": fit.coefficients,
                "internal_fit_condition_number": fit.standardized_condition_number,
                **score_prediction(adjusted[test_index], {}),
            }
        )
        adjusted_by_direction.append(adjusted)
    return direction_rows, adjusted_by_direction


def score_thermal_at_alpha(
    datasets: list[EquivariantDataset],
    adjusted_by_direction: list[list[EquivariantDataset]],
    names: list[str],
    alpha: float,
) -> dict[str, Any]:
    directions = []
    for direction_index, (train_index, test_index) in enumerate(((0, 1), (1, 0))):
        adjusted = adjusted_by_direction[direction_index]
        fit = fit_equivariant_ridge_features(
            [adjusted[train_index]],
            feature_names=names,
            alpha=alpha,
        )
        directions.append(
            {
                "train_cluster": datasets[train_index].name,
                "test_cluster": datasets[test_index].name,
                "thermal_coefficients": fit.coefficients,
                "thermal_fit_condition_number": fit.standardized_condition_number,
                **score_prediction(adjusted[test_index], fit.coefficients),
            }
        )
    return {
        "alpha": alpha,
        "symmetric_cross_cluster_full_field_NRMSE": symmetric_error(directions),
        "directions": directions,
    }


def evaluate_primary(
    config: dict[str, Any],
    base_config: dict[str, Any],
    datasets: list[EquivariantDataset],
) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline_directions, adjusted = prepare_static_baseline(datasets, base_config, config)
    baseline = {
        "directions": baseline_directions,
        "symmetric_cross_cluster_full_field_NRMSE": symmetric_error(baseline_directions),
    }
    alpha_grid = [float(value) for value in config["fit"]["thermal_ridge_alpha_grid"]]
    family_results = {}
    for family in config["thermal_features"]["families"]:
        names = thermal_feature_names(config, family)
        sweep = [
            score_thermal_at_alpha(datasets, adjusted, names, alpha) for alpha in alpha_grid
        ]
        selected = min(
            sweep,
            key=lambda row: (
                row["symmetric_cross_cluster_full_field_NRMSE"],
                row["alpha"],
            ),
        )
        zero_limit_alpha = float(config["fit"]["zero_thermal_limit_alpha"])
        zero_limit = next(row for row in sweep if row["alpha"] == zero_limit_alpha)
        family_results[family] = {
            "feature_names": names,
            "selected_alpha": selected["alpha"],
            "symmetric_cross_cluster_full_field_NRMSE": selected[
                "symmetric_cross_cluster_full_field_NRMSE"
            ],
            "cross_cluster_scores": selected["directions"],
            "alpha_sweep": sweep,
            "zero_thermal_limit_NRMSE": zero_limit[
                "symmetric_cross_cluster_full_field_NRMSE"
            ],
            "zero_thermal_limit_difference_from_static": abs(
                zero_limit["symmetric_cross_cluster_full_field_NRMSE"]
                - baseline["symmetric_cross_cluster_full_field_NRMSE"]
            ),
        }
    return baseline, family_results


def resolution_change(
    primary_symmetric: float,
    primary_directions: list[dict[str, Any]],
    doubled_symmetric: float,
    doubled_directions: list[dict[str, Any]],
) -> dict[str, Any]:
    tiny = np.finfo(float).tiny

    def relative(first: float, second: float) -> float:
        return float(abs(second - first) / max(abs(first), tiny))

    changes: dict[str, float] = {
        "symmetric_full_field_NRMSE_relative_change": relative(
            primary_symmetric,
            doubled_symmetric,
        )
    }
    doubled_by_pair = {
        (row["train_cluster"], row["test_cluster"]): row for row in doubled_directions
    }
    for row in primary_directions:
        pair = (row["train_cluster"], row["test_cluster"])
        other = doubled_by_pair[pair]
        tag = f"{pair[0]}_to_{pair[1]}"
        changes[f"{tag}_full_field_NRMSE_relative_change"] = relative(
            float(row["full_field_NRMSE"]),
            float(other["full_field_NRMSE"]),
        )
        changes[f"{tag}_residual_shear_alignment_absolute_change"] = abs(
            float(other["residual_shear_alignment_cosine"])
            - float(row["residual_shear_alignment_cosine"])
        )
        changes[f"{tag}_residual_power_closed_absolute_change"] = abs(
            float(other["residual_power_closed"])
            - float(row["residual_power_closed"])
        )
    return {
        "components": changes,
        "maximum_change": max(changes.values()),
    }


def render_figure(
    config: dict[str, Any],
    baseline: dict[str, Any],
    families: dict[str, Any],
    best_family: str,
    doubled: dict[str, Any],
    output: Path,
) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for family, result in families.items():
        axes[0].plot(
            [row["alpha"] for row in result["alpha_sweep"]],
            [row["symmetric_cross_cluster_full_field_NRMSE"] for row in result["alpha_sweep"]],
            marker="o",
            label=family,
        )
    axes[0].set_xscale("symlog", linthresh=1.0e-7)
    axes[0].axhline(
        float(config["diagnostic_gates"]["maximum_symmetric_cross_cluster_full_field_NRMSE"]),
        color="black",
        linestyle="--",
    )
    axes[0].set(xlabel="thermal ridge alpha", ylabel="symmetric NRMSE", title="Spent cross-transfer")
    axes[0].legend(fontsize=8)

    best = families[best_family]
    axes[1].bar(
        ["static", "thermal 193", "thermal 385"],
        [
            baseline["symmetric_cross_cluster_full_field_NRMSE"],
            best["symmetric_cross_cluster_full_field_NRMSE"],
            doubled["symmetric_cross_cluster_full_field_NRMSE"],
        ],
        color=["#777777", "#228833", "#66c2a5"],
    )
    axes[1].set(ylabel="symmetric NRMSE", title=f"Selected: {best_family}")

    directions = best["cross_cluster_scores"]
    positions = np.arange(len(directions))
    axes[2].bar(
        positions - 0.18,
        [row["residual_shear_alignment_cosine"] for row in directions],
        width=0.36,
        label="shear alignment",
    )
    axes[2].bar(
        positions + 0.18,
        [row["residual_power_closed"] for row in directions],
        width=0.36,
        label="power closed",
    )
    axes[2].set_xticks(
        positions,
        [f"{row['train_cluster']} to {row['test_cluster']}" for row in directions],
        rotation=15,
    )
    axes[2].axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    axes[2].axhline(0.25, color="black", linestyle=":", linewidth=0.8)
    axes[2].set(ylabel="dimensionless", title="Transferred residual structure")
    axes[2].legend(fontsize=8)

    path = output / "thermal_stress_transfer.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen target-gated v17E thermal-stress transfer diagnostic."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--thermal-report", type=Path, default=DEFAULT_THERMAL_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config_path = args.config.resolve()
    thermal_report_path = args.thermal_report.resolve()
    output = args.output.resolve()
    if (output / "report.json").exists():
        raise RuntimeError("the immutable v17E report already exists")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    thermal_report = json.loads(thermal_report_path.read_text(encoding="utf-8"))

    # This authorization completes before sample_cluster can open a spent target.
    products = validate_authorization(
        config_path,
        config,
        thermal_report_path,
        thermal_report,
    )
    base_config_path = ROOT / config["parents"]["static_baseline_config"]
    thermal_map_config_path = ROOT / config["parents"]["thermal_source_protocol"]
    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    thermal_map_config = json.loads(thermal_map_config_path.read_text(encoding="utf-8"))

    primary_points = int(config["resolution_control"]["primary_grid_points"])
    primary_datasets = build_datasets(
        config,
        base_config,
        thermal_map_config,
        products,
        primary_points,
    )
    baseline, families = evaluate_primary(config, base_config, primary_datasets)
    expected_static = float(
        config["static_baseline"][
            "expected_symmetric_cross_cluster_full_field_NRMSE"
        ]
    )
    static_difference = abs(
        baseline["symmetric_cross_cluster_full_field_NRMSE"] - expected_static
    )
    if static_difference > float(
        config["static_baseline"]["maximum_reproduction_error"]
    ):
        raise RuntimeError("v17E did not reproduce the frozen static baseline")

    best_family = min(
        families,
        key=lambda family: (
            families[family]["symmetric_cross_cluster_full_field_NRMSE"],
            family,
        ),
    )
    best = families[best_family]
    doubled_points = int(
        config["resolution_control"]["doubled_linear_resolution_grid_points"]
    )
    doubled_datasets = build_datasets(
        config,
        base_config,
        thermal_map_config,
        products,
        doubled_points,
    )
    doubled_baseline_directions, doubled_adjusted = prepare_static_baseline(
        doubled_datasets,
        base_config,
        config,
    )
    doubled = score_thermal_at_alpha(
        doubled_datasets,
        doubled_adjusted,
        thermal_feature_names(config, best_family),
        float(best["selected_alpha"]),
    )
    doubled["static_symmetric_cross_cluster_full_field_NRMSE"] = symmetric_error(
        doubled_baseline_directions
    )
    stability = resolution_change(
        float(best["symmetric_cross_cluster_full_field_NRMSE"]),
        best["cross_cluster_scores"],
        float(doubled["symmetric_cross_cluster_full_field_NRMSE"]),
        doubled["directions"],
    )

    baseline_error = float(baseline["symmetric_cross_cluster_full_field_NRMSE"])
    best_error = float(best["symmetric_cross_cluster_full_field_NRMSE"])
    relative_improvement = (baseline_error - best_error) / baseline_error
    gates = config["diagnostic_gates"]
    gate_results = {
        "static_baseline_reproduced": static_difference
        <= float(config["static_baseline"]["maximum_reproduction_error"]),
        "zero_thermal_limit_reproduces_static": all(
            result["zero_thermal_limit_difference_from_static"]
            <= float(gates["maximum_zero_thermal_limit_difference"])
            for result in families.values()
        ),
        "material_improvement": relative_improvement
        >= float(gates["minimum_relative_improvement_over_static_baseline"]),
        "absolute_source_sufficiency": best_error
        <= float(gates["maximum_symmetric_cross_cluster_full_field_NRMSE"]),
        "residual_shear_alignment_both_directions": all(
            row["residual_shear_alignment_cosine"]
            >= float(gates["minimum_residual_shear_alignment_cosine_each_direction"])
            for row in best["cross_cluster_scores"]
        ),
        "residual_power_both_directions": all(
            row["residual_power_closed"]
            >= float(gates["minimum_residual_power_closed_each_direction"])
            for row in best["cross_cluster_scores"]
        ),
        "doubled_resolution_stable": stability["maximum_change"]
        <= float(gates["maximum_map_resolution_change"]),
    }
    gate_results["advance"] = all(gate_results.values())
    decision = (
        "projected gas thermal stress advances as transferable source information; derive its response from one healthy universally coupled action"
        if gate_results["advance"]
        else "the tested resolved projected gas thermal-stress source fails at least one frozen transfer gate; do not rescue it with an object switch or private parameter"
    )

    output.mkdir(parents=True, exist_ok=True)
    figure = render_figure(config, baseline, families, best_family, doubled, output)
    report = {
        "status": "completed Sigma v17E thermal-stress transfer diagnostic",
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "sample_is_spent": True,
        "observational_validation_claim": False,
        "source_maps_frozen_before_target_opened": True,
        "lensing_target_opened": True,
        "one_metric_feature_triplets": True,
        "per_cluster_gravity_parameters": 0,
        "per_cluster_amplitude_scale_shear_or_orientation": False,
        "inverse_coefficients_are_physical_constants": False,
        "input_hashes": {
            "config": sha256(config_path),
            "thermal_source_report": sha256(thermal_report_path),
            "static_baseline_config": sha256(base_config_path),
            "thermal_source_protocol": sha256(thermal_map_config_path),
            **{f"{cluster}_thermal_source": sha256(path) for cluster, path in products.items()},
        },
        "primary_grid_points": primary_points,
        "doubled_grid_points": doubled_points,
        "preserved_static_baseline": baseline,
        "static_baseline_reproduction_difference": static_difference,
        "family_results": families,
        "selected_family": best_family,
        "relative_improvement_over_static_baseline": relative_improvement,
        "doubled_resolution_result": doubled,
        "resolution_stability": stability,
        "gate_results": gate_results,
        "decision": decision,
        "figure": figure.relative_to(ROOT).as_posix(),
        "figure_sha256": sha256(figure),
    }
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"{best_family}: symmetric NRMSE={best_error:.6f}")
    print(f"relative improvement={relative_improvement:.3%}")
    print(f"advance={gate_results['advance']}")
    print(report_path)


if __name__ == "__main__":
    main()
