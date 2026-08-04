#!/usr/bin/env python3
"""Run the conditional one-length Sigma v17F root-scale diagnostic."""

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

from build_sigma_v17d_thermal_stress_maps import one_metric_triplet
from infer_sigma_v16_spent_boundary import sample_cluster
from run_sigma_v17e_thermal_stress_transfer import (
    compare_field_extents,
    prepare_static_baseline,
    resolution_change,
    symmetric_error,
)
from run_sigma_v17e_thermal_stress_transfer import (
    validate_authorization as validate_v17e_authorization,
)

from voidscreen.sigma_covariant_feature_inference import (
    EquivariantDataset,
    MetricFeature,
    fit_equivariant_ridge_features,
    predict_residual,
    score_prediction,
)

DEFAULT_CONFIG = ROOT / "configs" / "sigma_v17f_root_scale_propagator.json"
DEFAULT_V17E_REPORT = ROOT / "results" / "sigma_v17e_thermal_stress_transfer" / "report.json"
DEFAULT_THERMAL_REPORT = ROOT / "results" / "sigma_v17d_thermal_stress_maps" / "report.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v17f_root_scale_propagator"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def helmholtz_propagate(
    source: np.ndarray,
    spacing_kpc: float,
    length_kpc: float,
    padding_factor: int,
) -> np.ndarray:
    """Solve (1-L^2 Laplacian)s=q with zero padding on a square grid."""
    values = np.asarray(source, dtype=float)
    if (
        values.ndim != 2
        or values.shape[0] != values.shape[1]
        or not np.isfinite(values).all()
        or not np.isfinite(spacing_kpc)
        or spacing_kpc <= 0.0
        or not np.isfinite(length_kpc)
        or length_kpc < 0.0
        or padding_factor < 2
    ):
        raise ValueError("invalid Helmholtz propagation inputs")
    if length_kpc == 0.0:
        return values.copy()

    side = values.shape[0]
    padded_side = int(padding_factor * side)
    offset = (padded_side - side) // 2
    padded = np.zeros((padded_side, padded_side), dtype=float)
    padded[offset : offset + side, offset : offset + side] = values
    frequencies = 2.0 * np.pi * np.fft.fftfreq(padded_side, d=spacing_kpc)
    k_east, k_north = np.meshgrid(frequencies, frequencies)
    denominator = 1.0 + length_kpc**2 * (k_east**2 + k_north**2)
    propagated = np.fft.ifft2(np.fft.fft2(padded) / denominator).real
    cropped = propagated[offset : offset + side, offset : offset + side]
    if not np.isfinite(cropped).all():
        raise RuntimeError("Helmholtz propagation produced a non-finite field")
    return cropped


def feature_name(source_family: str, length_kpc: float) -> str:
    return f"root_{source_family}_L{length_kpc:g}kpc"


def propagated_feature(
    product: Path,
    source_family: str,
    length_kpc: float,
    target_axis: np.ndarray,
    padding_factor: int,
) -> MetricFeature:
    with np.load(product) as data:
        source_axis = data["source_axis_kpc"].astype(float)
        source = data[source_family].astype(float)
    if source_axis.ndim != 1 or source_axis.size != source.shape[0]:
        raise RuntimeError(f"invalid thermal source grid in {product}")
    spacings = np.diff(source_axis)
    if spacings.size == 0 or not np.allclose(
        spacings, spacings[0], rtol=0.0, atol=1e-10
    ):
        raise RuntimeError(f"nonuniform thermal source grid in {product}")
    propagated = helmholtz_propagate(
        source,
        float(spacings[0]),
        length_kpc,
        padding_factor,
    )
    convergence, shear_1, shear_2 = one_metric_triplet(
        source_axis,
        propagated,
        target_axis,
        padding_factor,
    )
    name = feature_name(source_family, length_kpc)
    return MetricFeature(
        name=name,
        family="scalar_scale",
        convergence=convergence,
        shear_1=shear_1,
        shear_2=shear_2,
    )


def build_datasets(
    config: dict[str, Any],
    v17e_config: dict[str, Any],
    base_config: dict[str, Any],
    products: dict[str, Path],
    grid_points: int,
) -> list[EquivariantDataset]:
    resolution_base = copy.deepcopy(base_config)
    resolution_base["map_measurement"]["target_grid_points"] = int(grid_points)
    sources = list(config["projected_root_equation"]["source_families"])
    lengths = [float(value) for value in config["propagation"]["L_sigma_kpc_grid"]]
    padding = int(config["propagation"]["fourier_padding_factor"])
    datasets = []
    for cluster in v17e_config["sample"]["clusters"]:
        dataset, metadata, _ = sample_cluster(cluster, resolution_base)
        target_axis = np.asarray(metadata["target_axis_kpc"], dtype=float)
        root_features = {}
        for source_family in sources:
            for length in lengths:
                feature = propagated_feature(
                    products[cluster],
                    source_family,
                    length,
                    target_axis,
                    padding,
                )
                root_features[feature.name] = feature
        overlap = set(dataset.features).intersection(root_features)
        if overlap:
            raise RuntimeError(f"root and static feature names overlap: {sorted(overlap)}")
        datasets.append(
            EquivariantDataset(
                name=dataset.name,
                mask=dataset.mask,
                base=dataset.base,
                target=dataset.target,
                features={**dataset.features, **root_features},
            )
        )
    return datasets


def score_candidate(
    datasets: list[EquivariantDataset],
    adjusted_by_direction: list[list[EquivariantDataset]],
    name: str,
    half_width_kpc: float,
    fixed_betas: dict[tuple[str, str], float] | None = None,
) -> dict[str, Any]:
    directions = []
    for direction_index, (train_index, test_index) in enumerate(((0, 1), (1, 0))):
        adjusted = adjusted_by_direction[direction_index]
        pair = (datasets[train_index].name, datasets[test_index].name)
        if fixed_betas is None:
            fit = fit_equivariant_ridge_features(
                [adjusted[train_index]],
                feature_names=[name],
                alpha=0.0,
            )
            beta = float(fit.coefficients[name])
            condition = fit.standardized_condition_number
        else:
            beta = float(fixed_betas[pair])
            condition = None
        coefficients = {name: beta}
        prediction = predict_residual(adjusted[test_index], coefficients)
        required = tuple(
            target - base
            for target, base in zip(
                adjusted[test_index].target,
                adjusted[test_index].base,
                strict=True,
            )
        )
        directions.append(
            {
                "train_cluster": pair[0],
                "test_cluster": pair[1],
                "beta_sigma": beta,
                "fit_condition_number": condition,
                "halo_scale_diagnostic": compare_field_extents(
                    required,
                    prediction,
                    adjusted[test_index].mask,
                    half_width_kpc,
                ),
                **score_prediction(adjusted[test_index], coefficients),
            }
        )
    positive = all(
        np.isfinite(row["beta_sigma"]) and row["beta_sigma"] > 0.0
        for row in directions
    )
    beta_difference = (
        float(abs(np.log10(directions[0]["beta_sigma"] / directions[1]["beta_sigma"])))
        if positive
        else None
    )
    return {
        "feature_name": name,
        "symmetric_cross_cluster_full_field_NRMSE": symmetric_error(directions),
        "directional_beta_log10_difference_dex": beta_difference,
        "directions": directions,
    }


def validate_authorization(
    config_path: Path,
    config: dict[str, Any],
    v17e_report_path: Path,
    thermal_report_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Path],
]:
    expected_status = (
        "frozen before any regional temperature result, thermal source map, v17E "
        "inverse coefficient, v17E lensing score, or v17F result existed"
    )
    if config.get("status") != expected_status or not config_path.is_file():
        raise RuntimeError("the v17F protocol is not in its frozen state")
    for path_key, hash_key in (
        ("thermal_source_protocol", "thermal_source_protocol_sha256"),
        ("thermal_transfer_protocol", "thermal_transfer_protocol_sha256"),
        ("static_baseline_protocol", "static_baseline_protocol_sha256"),
    ):
        path = ROOT / config["parents"][path_key]
        if not path.is_file() or sha256(path) != config["parents"][hash_key]:
            raise RuntimeError(f"frozen v17F parent changed: {path_key}")
    if not v17e_report_path.is_file() or not thermal_report_path.is_file():
        raise RuntimeError("v17F is not authorized because an upstream report is absent")

    v17e_config_path = ROOT / config["parents"]["thermal_transfer_protocol"]
    thermal_config_path = ROOT / config["parents"]["thermal_source_protocol"]
    base_config_path = ROOT / config["parents"]["static_baseline_protocol"]
    v17e_config = json.loads(v17e_config_path.read_text(encoding="utf-8"))
    thermal_config = json.loads(thermal_config_path.read_text(encoding="utf-8"))
    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    v17e_report = json.loads(v17e_report_path.read_text(encoding="utf-8"))
    thermal_report = json.loads(thermal_report_path.read_text(encoding="utf-8"))
    if v17e_report.get("status") != config["authorization"]["required_v17e_status"]:
        raise RuntimeError("v17E has not completed with the required status")
    if v17e_report.get("input_hashes", {}).get("config") != sha256(v17e_config_path):
        raise RuntimeError("v17E used another transfer protocol")
    if v17e_report.get("gate_results", {}).get("advance") is not True:
        raise RuntimeError("v17F is prohibited because v17E did not advance")
    products = validate_v17e_authorization(
        v17e_config_path,
        v17e_config,
        thermal_report_path,
        thermal_report,
    )
    expected_thermal_hash = v17e_report.get("input_hashes", {}).get(
        "thermal_source_report"
    )
    if expected_thermal_hash != sha256(thermal_report_path):
        raise RuntimeError("thermal source report changed after v17E")
    for cluster, product in products.items():
        if v17e_report["input_hashes"].get(f"{cluster}_thermal_source") != sha256(
            product
        ):
            raise RuntimeError(f"thermal source changed after v17E: {cluster}")
    return v17e_report, v17e_config, thermal_config, base_config, products


def render_figure(
    candidates: list[dict[str, Any]],
    selected: dict[str, Any],
    baseline_error: float,
    flexible_error: float,
    output: Path,
) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        by_source.setdefault(row["source_family"], []).append(row)
    for source, rows in by_source.items():
        ordered = sorted(rows, key=lambda item: item["L_sigma_kpc"])
        axes[0, 0].plot(
            [row["L_sigma_kpc"] for row in ordered],
            [row["symmetric_cross_cluster_full_field_NRMSE"] for row in ordered],
            marker="o",
            label=source,
        )
    axes[0, 0].axhline(baseline_error, color="gray", linestyle="--", label="static")
    axes[0, 0].axhline(flexible_error, color="black", linestyle=":", label="v17E")
    axes[0, 0].set(xlabel=r"$L_\Sigma$ (kpc)", ylabel="symmetric NRMSE", title="One-length sweep")
    axes[0, 0].legend(fontsize=8)

    directions = selected["directions"]
    labels = [f"{row['train_cluster']}→{row['test_cluster']}" for row in directions]
    axes[0, 1].bar(labels, [row["beta_sigma"] for row in directions], color="#4477aa")
    axes[0, 1].set(ylabel=r"training $\beta_\Sigma$", title="Directional constant estimates")

    radius_labels = []
    required_values = []
    predicted_values = []
    for row in directions:
        for radius in ("R50", "R80"):
            radius_labels.append(f"{row['test_cluster']} {radius}")
            required_values.append(
                row["halo_scale_diagnostic"]["required"]["radii_kpc"][radius]
            )
            predicted_values.append(
                row["halo_scale_diagnostic"]["predicted"]["radii_kpc"][radius]
            )
    positions = np.arange(len(radius_labels))
    axes[1, 0].bar(positions - 0.18, required_values, width=0.36, label="required")
    axes[1, 0].bar(positions + 0.18, predicted_values, width=0.36, label="predicted")
    axes[1, 0].set_xticks(positions, radius_labels, rotation=20, ha="right")
    axes[1, 0].set(ylabel="field-energy radius (kpc)", title="Amplitude-independent extent")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].bar(
        ["static", "v17E flexible", "v17F one length"],
        [baseline_error, flexible_error, selected["symmetric_cross_cluster_full_field_NRMSE"]],
        color=["#888888", "#66c2a5", "#cc6677"],
    )
    axes[1, 1].set(ylabel="symmetric NRMSE", title="Complexity reduction")
    figure_path = output / "root_scale_propagator.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return figure_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--v17e-report", type=Path, default=DEFAULT_V17E_REPORT)
    parser.add_argument("--thermal-report", type=Path, default=DEFAULT_THERMAL_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    v17e_report_path = args.v17e_report.resolve()
    thermal_report_path = args.thermal_report.resolve()
    output = args.output.resolve()
    if (output / "report.json").exists():
        raise RuntimeError("the immutable v17F report already exists")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    (
        v17e_report,
        v17e_config,
        _thermal_config,
        base_config,
        products,
    ) = validate_authorization(
        config_path,
        config,
        v17e_report_path,
        thermal_report_path,
    )

    primary_points = int(config["resolution_control"]["primary_grid_points"])
    datasets = build_datasets(
        config,
        v17e_config,
        base_config,
        products,
        primary_points,
    )
    baseline_directions, adjusted = prepare_static_baseline(
        datasets,
        base_config,
        v17e_config,
    )
    baseline_error = symmetric_error(baseline_directions)
    half_width_kpc = float(config["propagation"]["target_half_width_kpc"])
    candidates = []
    sources = list(config["projected_root_equation"]["source_families"])
    for source_index, source_family in enumerate(sources):
        for length in config["propagation"]["L_sigma_kpc_grid"]:
            length_kpc = float(length)
            candidate = score_candidate(
                datasets,
                adjusted,
                feature_name(source_family, length_kpc),
                half_width_kpc,
            )
            candidate.update(
                {
                    "source_family": source_family,
                    "source_tie_break_index": source_index,
                    "L_sigma_kpc": length_kpc,
                }
            )
            candidates.append(candidate)
    positive_candidates = [
        row
        for row in candidates
        if all(
            np.isfinite(direction["beta_sigma"])
            and direction["beta_sigma"] > 0.0
            for direction in row["directions"]
        )
    ]
    selection_pool = positive_candidates if positive_candidates else candidates
    selected = min(
        selection_pool,
        key=lambda row: (
            row["symmetric_cross_cluster_full_field_NRMSE"],
            row["L_sigma_kpc"],
            row["source_tie_break_index"],
        ),
    )

    doubled_points = int(
        config["resolution_control"]["doubled_linear_resolution_grid_points"]
    )
    doubled_datasets = build_datasets(
        config,
        v17e_config,
        base_config,
        products,
        doubled_points,
    )
    doubled_baseline_directions, doubled_adjusted = prepare_static_baseline(
        doubled_datasets,
        base_config,
        v17e_config,
    )
    primary_betas = {
        (row["train_cluster"], row["test_cluster"]): float(row["beta_sigma"])
        for row in selected["directions"]
    }
    doubled = score_candidate(
        doubled_datasets,
        doubled_adjusted,
        selected["feature_name"],
        half_width_kpc,
        fixed_betas=primary_betas,
    )
    stability = resolution_change(
        float(selected["symmetric_cross_cluster_full_field_NRMSE"]),
        selected["directions"],
        float(doubled["symmetric_cross_cluster_full_field_NRMSE"]),
        doubled["directions"],
    )

    selected_error = float(selected["symmetric_cross_cluster_full_field_NRMSE"])
    flexible_error = float(
        v17e_report["family_results"][v17e_report["selected_family"]][
            "symmetric_cross_cluster_full_field_NRMSE"
        ]
    )
    improvement = (baseline_error - selected_error) / baseline_error
    degradation = (selected_error - flexible_error) / flexible_error
    gates = config["gates"]
    upper_length = max(float(value) for value in config["propagation"]["L_sigma_kpc_grid"])
    gate_results = {
        "upstream_v17e_advanced": v17e_report["gate_results"]["advance"] is True,
        "material_improvement": improvement
        >= float(gates["minimum_relative_improvement_over_static_baseline"]),
        "absolute_source_sufficiency": selected_error
        <= float(gates["maximum_symmetric_cross_cluster_full_field_NRMSE"]),
        "competitive_with_flexible_v17e": degradation
        <= float(gates["maximum_relative_NRMSE_degradation_from_flexible_v17e"]),
        "positive_attractive_beta_both_directions": all(
            np.isfinite(row["beta_sigma"]) and row["beta_sigma"] > 0.0
            for row in selected["directions"]
        ),
        "directional_beta_consistent": selected["directional_beta_log10_difference_dex"]
        is not None
        and selected["directional_beta_log10_difference_dex"]
        <= float(gates["maximum_directional_beta_log10_difference_dex"]),
        "residual_shear_alignment_both_directions": all(
            row["residual_shear_alignment_cosine"]
            >= float(gates["minimum_residual_shear_alignment_cosine_each_direction"])
            for row in selected["directions"]
        ),
        "residual_power_both_directions": all(
            row["residual_power_closed"]
            >= float(gates["minimum_residual_power_closed_each_direction"])
            for row in selected["directions"]
        ),
        "halo_scale_both_directions": all(
            row["halo_scale_diagnostic"]["valid"]
            and row["halo_scale_diagnostic"]["maximum_fractional_radius_error"]
            <= float(gates["maximum_fractional_R50_or_R80_error_each_direction"])
            for row in selected["directions"]
        ),
        "length_identified_inside_grid": float(selected["L_sigma_kpc"])
        < upper_length,
        "doubled_resolution_stable": stability["valid"]
        and stability["maximum_change"]
        <= float(gates["maximum_resolution_change"]),
    }
    gate_results["advance"] = all(gate_results.values())
    if gate_results["advance"] and float(selected["L_sigma_kpc"]) == 0.0:
        decision = "source-derived scale advances; derive a scale-free action without a propagation-length constant"
    elif gate_results["advance"]:
        decision = "one universal finite propagation length advances for covariant action derivation and later holdout freeze"
    else:
        decision = "the instantaneous thermal source plus one finite-range mediator fails; do not add private lengths or exponents"

    output.mkdir(parents=True, exist_ok=True)
    figure = render_figure(candidates, selected, baseline_error, flexible_error, output)
    report = {
        "status": "completed conditional Sigma v17F root-scale propagator diagnostic",
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "sample_is_spent": True,
        "observational_validation_claim": False,
        "projected_equation_is_covariant_action": False,
        "per_cluster_gravity_parameters": 0,
        "input_hashes": {
            "config": sha256(config_path),
            "v17e_report": sha256(v17e_report_path),
            "thermal_source_report": sha256(thermal_report_path),
            **{f"{cluster}_thermal_source": sha256(path) for cluster, path in products.items()},
        },
        "primary_grid_points": primary_points,
        "doubled_grid_points": doubled_points,
        "preserved_static_baseline": {
            "directions": baseline_directions,
            "symmetric_cross_cluster_full_field_NRMSE": baseline_error,
        },
        "flexible_v17e_NRMSE": flexible_error,
        "positive_candidate_count": len(positive_candidates),
        "candidate_sweep": candidates,
        "selected_candidate": selected,
        "relative_improvement_over_static_baseline": improvement,
        "relative_NRMSE_degradation_from_flexible_v17e": degradation,
        "doubled_resolution": doubled,
        "doubled_static_symmetric_NRMSE": symmetric_error(doubled_baseline_directions),
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
    print(
        f"{selected['source_family']} L={selected['L_sigma_kpc']:.6g} kpc: "
        f"symmetric NRMSE={selected_error:.6f}"
    )
    print(f"advance={gate_results['advance']}")
    print(report_path)


if __name__ == "__main__":
    main()
