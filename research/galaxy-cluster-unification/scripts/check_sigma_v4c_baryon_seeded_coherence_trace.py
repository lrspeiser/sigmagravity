from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from check_sigma_v4a_projected_variational_source import (
    analytic_eta,
    logarithmic_interior,
    score_prediction,
    sha256,
)
from check_sigma_v4b_vector_stress_memory import add_physical_deflection
from infer_sigma_v3c_spent_operator import sample_cluster

from voidscreen.sigma_coherence_trace import (
    coherence_trace_state,
    directional_disorder,
    helmholtz_relative_residual,
    projected_coherence_trace,
)
from voidscreen.sigma_operator_inference import windowed_fourier


def manufactured_vector(points: int = 25) -> np.ndarray:
    coordinate = np.arange(points, dtype=float)
    x, y = np.meshgrid(coordinate, coordinate)
    angle = 2.0 * np.pi * x / points + 0.35 * np.sin(2.0 * np.pi * y / points)
    magnitude = 2.0 + 0.4 * np.cos(2.0 * np.pi * (x + y) / points)
    return np.stack([magnitude * np.cos(angle), magnitude * np.sin(angle)], axis=-1)


def manufactured_checks() -> dict[str, float]:
    spacing = 0.5
    memory_length = 2.1
    vector_scale = 0.7
    vector = manufactured_vector()
    coordinate = np.arange(vector.shape[0], dtype=float)
    x, y = np.meshgrid(coordinate, coordinate)
    baryons = 1.0 + 0.2 * np.cos(2.0 * np.pi * (x - 2.0 * y) / vector.shape[0])

    uniform = np.zeros_like(vector)
    uniform[..., 0] = 3.0
    uniform_disorder = directional_disorder(
        uniform,
        spacing=spacing,
        memory_length=memory_length,
        vector_scale=vector_scale,
    )[3]

    angle = 0.713
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    rotated = np.einsum("ij,...j->...i", rotation, vector)
    original_disorder = directional_disorder(
        vector,
        spacing=spacing,
        memory_length=memory_length,
        vector_scale=vector_scale,
    )[3]
    rotated_disorder = directional_disorder(
        rotated,
        spacing=spacing,
        memory_length=memory_length,
        vector_scale=vector_scale,
    )[3]
    rotation_error = float(
        np.linalg.norm(rotated_disorder - original_disorder)
        / np.linalg.norm(original_disorder)
    )

    *_, seed, trace = coherence_trace_state(
        vector,
        baryons,
        spacing=spacing,
        memory_length=memory_length,
        vector_scale=vector_scale,
    )
    equation_residual = helmholtz_relative_residual(
        trace, seed, spacing=spacing, length=memory_length
    )
    integral_mismatch = abs(float(np.sum(trace) - np.sum(seed))) / float(np.sum(seed))
    trace_rms = float(np.sqrt(np.mean(np.square(trace))))

    high_field_scale = 0.02
    low_seed = coherence_trace_state(
        vector,
        baryons,
        spacing=spacing,
        memory_length=memory_length,
        vector_scale=high_field_scale,
    )[-2]
    high_seed = coherence_trace_state(
        1000.0 * vector,
        baryons,
        spacing=spacing,
        memory_length=memory_length,
        vector_scale=high_field_scale,
    )[-2]
    return {
        "helmholtz_equation_relative_residual": equation_residual,
        "uniform_vector_maximum_directional_variance": float(
            np.max(np.abs(uniform_disorder))
        ),
        "rotation_covariance_relative_error": rotation_error,
        "high_field_seed_fraction_after_1000x_vector_scaling": float(
            np.sum(high_seed) / np.sum(low_seed)
        ),
        "trace_integral_mismatch_fraction": integral_mismatch,
        "minimum_trace_state_over_trace_RMS": float(np.min(trace)) / trace_rms,
    }


def correction_templates(
    dataset: dict[str, object],
    *,
    memory_length: float,
    vector_scale: float,
    padding_factor: int,
) -> tuple[dict[str, np.ndarray], object]:
    axis = np.asarray(dataset["axis_kpc"], dtype=float)
    field = projected_coherence_trace(
        dataset["physical_deflection_east_kpc"],
        dataset["physical_deflection_north_kpc"],
        dataset["invariants"]["Newtonian"].convergence,
        spacing=float(axis[1] - axis[0]),
        memory_length=memory_length,
        vector_scale=vector_scale,
        padding_factor=padding_factor,
    )
    window = dataset["window"]
    transforms = {
        "convergence": windowed_fourier(field.unit_eta_kappa, window),
        "shear_1": windowed_fourier(field.unit_eta_shear_1, window),
        "shear_2": windowed_fourier(field.unit_eta_shear_2, window),
    }
    return transforms, field


def evaluate_parameters(
    datasets: list[dict[str, object]],
    *,
    memory_length: float,
    vector_scale: float,
    padding_factor: int,
    eta_override: float | None = None,
) -> dict[str, object]:
    generated = [
        correction_templates(
            dataset,
            memory_length=memory_length,
            vector_scale=vector_scale,
            padding_factor=padding_factor,
        )
        for dataset in datasets
    ]
    templates = [item[0] for item in generated]
    fields = [item[1] for item in generated]
    unconstrained, physical = analytic_eta(datasets, templates)
    eta = physical if eta_override is None else float(eta_override)
    score, per_cluster, per_channel = score_prediction(datasets, templates, eta)
    return {
        "L_sigma_kpc": float(memory_length),
        "ell_sigma_kpc": float(vector_scale),
        "eta_sigma": eta,
        "unconstrained_eta_sigma": unconstrained,
        "normalized_RMSE": score,
        "per_cluster": per_cluster,
        "per_channel": per_channel,
        "templates": templates,
        "fields": fields,
    }


def fit_shared(
    datasets: list[dict[str, object]], config: dict, *, padding_factor: int
) -> dict[str, object]:
    numerics = config["numerics"]
    length_bounds = [math.log(float(value)) for value in numerics["L_sigma_kpc_bounds"]]
    scale_bounds = [math.log(float(value)) for value in numerics["ell_sigma_kpc_bounds"]]

    def objective(parameters: np.ndarray) -> float:
        evaluated = evaluate_parameters(
            datasets,
            memory_length=math.exp(float(parameters[0])),
            vector_scale=math.exp(float(parameters[1])),
            padding_factor=padding_factor,
        )
        return float(evaluated["normalized_RMSE"]) ** 2

    fit = differential_evolution(
        objective,
        [tuple(length_bounds), tuple(scale_bounds)],
        seed=int(numerics["fit_seed"]),
        maxiter=int(numerics["differential_evolution_maxiter"]),
        popsize=int(numerics["differential_evolution_popsize"]),
        tol=float(numerics["differential_evolution_tolerance"]),
        polish=True,
        updating="immediate",
        workers=1,
    )
    evaluated = evaluate_parameters(
        datasets,
        memory_length=math.exp(float(fit.x[0])),
        vector_scale=math.exp(float(fit.x[1])),
        padding_factor=padding_factor,
    )
    evaluated["optimizer_success"] = bool(fit.success)
    evaluated["optimizer_message"] = str(fit.message)
    evaluated["function_evaluations"] = int(fit.nfev)
    return evaluated


def serializable_fit(fit: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in fit.items() if key not in {"templates", "fields"}}


def broad_power_fractions(
    datasets: list[dict[str, object]],
    templates: list[dict[str, np.ndarray]],
    threshold_kpc: float,
) -> list[dict[str, float | str]]:
    records = []
    for dataset, channels in zip(datasets, templates, strict=True):
        band = np.asarray(dataset["band"], dtype=bool)
        broad = band & (
            np.asarray(dataset["wavenumber"], dtype=float) <= 2.0 * np.pi / threshold_kpc
        )
        total = sum(float(np.sum(np.abs(values[band]) ** 2)) for values in channels.values())
        long_power = sum(
            float(np.sum(np.abs(values[broad]) ** 2)) for values in channels.values()
        )
        records.append(
            {
                "cluster": str(dataset["cluster"]),
                "correction_power_fraction_wavelength_ge_50kpc": long_power / total,
            }
        )
    return records


def source_statistics(datasets: list[dict[str, object]], fields: list[object]) -> list[dict]:
    records = []
    for dataset, field in zip(datasets, fields, strict=True):
        seed_sum = float(np.sum(field.full_baryonic_seed))
        trace_sum = float(np.sum(field.full_trace_state))
        trace_rms = float(np.sqrt(np.mean(np.square(field.full_trace_state))))
        baryons = np.maximum(
            np.asarray(dataset["invariants"]["Newtonian"].convergence, dtype=float), 0.0
        )
        baryon_sum = float(np.sum(baryons))
        records.append(
            {
                "cluster": str(dataset["cluster"]),
                "baryonic_seed_fraction_of_cropped_baryon_sum": seed_sum / baryon_sum,
                "trace_integral_mismatch_fraction": abs(trace_sum - seed_sum) / seed_sum,
                "minimum_full_trace_over_RMS": float(np.min(field.full_trace_state))
                / trace_rms,
                "baryon_weighted_directional_disorder": float(
                    np.sum(baryons * field.directional_disorder) / baryon_sum
                ),
                "baryon_weighted_high_field_activation": float(
                    np.sum(baryons * field.high_field_activation) / baryon_sum
                ),
                "cropped_trace_RMS": float(np.sqrt(np.mean(np.square(field.trace_state)))),
                "minimum_raw_directional_variance": float(
                    np.min(field.raw_directional_variance)
                ),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen Sigma v4C map audit.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v4c_baryon_seeded_coherence_trace_audit.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v4c_baryon_seeded_coherence_trace_audit",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    parent = ROOT / config["parent_config"]["path"]
    if sha256(parent) != config["parent_config"]["sha256"]:
        raise RuntimeError("parent v3C config hash does not match the frozen protocol")
    parent_config = json.loads(parent.read_text(encoding="utf-8"))
    datasets = [sample_cluster(name, parent_config) for name in config["sample"]["clusters"]]
    for dataset in datasets:
        add_physical_deflection(dataset, parent_config)
    args.output.mkdir(parents=True, exist_ok=True)

    checks = manufactured_checks()
    primary_padding = int(config["numerics"]["primary_padding_factor"])
    sensitivity_padding = int(config["numerics"]["padding_sensitivity_factor"])
    primary = fit_shared(datasets, config, padding_factor=primary_padding)
    sensitivity = fit_shared(datasets, config, padding_factor=sensitivity_padding)

    independent_fits: dict[str, dict[str, object]] = {}
    cross_transfer: list[dict[str, float | str]] = []
    for index, training in enumerate(datasets):
        fitted = fit_shared([training], config, padding_factor=primary_padding)
        independent_fits[str(training["cluster"])] = serializable_fit(fitted)
        target = datasets[1 - index]
        transferred = evaluate_parameters(
            [target],
            memory_length=float(fitted["L_sigma_kpc"]),
            vector_scale=float(fitted["ell_sigma_kpc"]),
            padding_factor=primary_padding,
            eta_override=float(fitted["eta_sigma"]),
        )
        cross_transfer.append(
            {
                "trained_on": str(training["cluster"]),
                "tested_on": str(target["cluster"]),
                "normalized_RMSE": float(transferred["normalized_RMSE"]),
            }
        )

    statistics = source_statistics(datasets, primary["fields"])
    broad_power = broad_power_fractions(
        datasets,
        primary["templates"],
        float(config["numerics"]["broad_wavelength_threshold_kpc"]),
    )
    gates_config = config["preregistered_gates"]
    parameter_interior = logarithmic_interior(
        float(primary["L_sigma_kpc"]), config["numerics"]["L_sigma_kpc_bounds"]
    ) and logarithmic_interior(
        float(primary["ell_sigma_kpc"]), config["numerics"]["ell_sigma_kpc_bounds"]
    )
    padding_change = abs(
        float(sensitivity["normalized_RMSE"]) - float(primary["normalized_RMSE"])
    ) / float(primary["normalized_RMSE"])
    gates = {
        "helmholtz_equation": checks["helmholtz_equation_relative_residual"]
        <= gates_config["maximum_helmholtz_equation_relative_residual"],
        "uniform_vector_null": checks["uniform_vector_maximum_directional_variance"]
        <= gates_config["maximum_uniform_vector_directional_variance"],
        "rotation_covariance": checks["rotation_covariance_relative_error"]
        <= gates_config["maximum_rotation_covariance_relative_error"],
        "high_field_suppression": checks[
            "high_field_seed_fraction_after_1000x_vector_scaling"
        ]
        <= gates_config["maximum_high_field_seed_fraction_after_1000x_vector_scaling"],
        "trace_integral": all(
            row["trace_integral_mismatch_fraction"]
            <= gates_config["maximum_trace_integral_mismatch_fraction"]
            for row in statistics
        ),
        "nonnegative_trace": all(
            row["minimum_full_trace_over_RMS"]
            >= gates_config["minimum_trace_state_over_trace_rms"]
            for row in statistics
        ),
        "broad_correction_power": all(
            row["correction_power_fraction_wavelength_ge_50kpc"]
            >= gates_config["minimum_correction_power_fraction_at_wavelength_ge_50kpc"]
            for row in broad_power
        ),
        "positive_physical_sign": float(primary["unconstrained_eta_sigma"]) > 0.0,
        "parameters_interior": parameter_interior,
        "joint_map_accuracy": float(primary["normalized_RMSE"])
        <= gates_config["maximum_joint_normalized_RMSE"],
        "each_cluster_twenty_percent_improvement": all(
            row["fraction_of_AQUAL_baseline"]
            <= gates_config["maximum_each_cluster_fraction_of_AQUAL_baseline_RMSE"]
            for row in primary["per_cluster"]
        ),
        "every_cluster_channel_improves": all(
            bool(row["improved"]) for row in primary["per_channel"]
        ),
        "cross_cluster_transfer": all(
            float(row["normalized_RMSE"])
            <= gates_config["maximum_each_cross_cluster_transfer_normalized_RMSE"]
            for row in cross_transfer
        ),
        "padding_stability": padding_change
        <= gates_config["maximum_padding_sensitivity_fractional_change_in_joint_RMSE"],
    }
    all_pass = bool(all(gates.values()))

    pd.DataFrame(primary["per_cluster"]).to_csv(
        args.output / "per_cluster_metrics.csv", index=False
    )
    pd.DataFrame(primary["per_channel"]).to_csv(
        args.output / "per_channel_metrics.csv", index=False
    )
    pd.DataFrame(cross_transfer).to_csv(args.output / "cross_transfer_metrics.csv", index=False)
    pd.DataFrame(statistics).to_csv(args.output / "source_statistics.csv", index=False)
    pd.DataFrame(broad_power).to_csv(args.output / "broad_power_metrics.csv", index=False)

    eta = float(primary["eta_sigma"])
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for row, (dataset, field) in enumerate(zip(datasets, primary["fields"], strict=True)):
        axis = dataset["axis_kpc"]
        target = dataset["invariants"]["halo"].convergence
        baseline = dataset["invariants"]["AQUAL"].convergence
        correction = eta * field.unit_eta_kappa
        prediction = baseline + correction
        limit = float(np.nanpercentile(np.abs(target), 99.0))
        image = axes[row, 0].imshow(
            target,
            origin="lower",
            extent=[axis[0], axis[-1], axis[0], axis[-1]],
            cmap="viridis",
            vmin=-limit,
            vmax=limit,
        )
        axes[row, 0].set_title(f"{dataset['cluster']} halo target kappa")
        figure.colorbar(image, ax=axes[row, 0], shrink=0.75)
        image = axes[row, 1].imshow(
            correction,
            origin="lower",
            extent=[axis[0], axis[-1], axis[0], axis[-1]],
            cmap="magma",
            vmin=0.0,
            vmax=float(np.nanpercentile(correction, 99.0)),
        )
        axes[row, 1].set_title("baryon-seeded trace correction")
        figure.colorbar(image, ax=axes[row, 1], shrink=0.75)
        image = axes[row, 2].imshow(
            prediction,
            origin="lower",
            extent=[axis[0], axis[-1], axis[0], axis[-1]],
            cmap="viridis",
            vmin=-limit,
            vmax=limit,
        )
        contour_max = float(np.nanpercentile(target, 99.0))
        axes[row, 2].contour(
            axis,
            axis,
            target,
            levels=np.linspace(0.2 * contour_max, contour_max, 5),
            colors="white",
            linewidths=0.7,
        )
        axes[row, 2].set_title("prediction color, target contours")
        figure.colorbar(image, ax=axes[row, 2], shrink=0.75)
        for column in range(3):
            axes[row, column].set(xlabel="east kpc", ylabel="north kpc")
    figure.suptitle(
        "Sigma v4C baryon-seeded coherence trace\n"
        f"L={primary['L_sigma_kpc']:.3g} kpc, ell={primary['ell_sigma_kpc']:.3g} kpc, "
        f"eta={eta:.3g}, joint NRMSE={primary['normalized_RMSE']:.3f}"
    )
    figure.savefig(args.output / "coherence_trace_audit.png", dpi=180)
    plt.close(figure)

    report = {
        "status": "completed Sigma v4C baryon-seeded coherence-trace audit",
        "sample_is_spent": True,
        "raw_holdout_opened": False,
        "config_sha256": sha256(args.config),
        "parent_config_sha256": sha256(parent),
        "manufactured_checks": checks,
        "primary_shared_fit": serializable_fit(primary),
        "padding_sensitivity_shared_fit": serializable_fit(sensitivity),
        "padding_sensitivity_fractional_change_in_joint_RMSE": padding_change,
        "independent_training_fits": independent_fits,
        "cross_transfer": cross_transfer,
        "source_statistics": statistics,
        "broad_power": broad_power,
        "gates": gates,
        "all_preregistered_gates_pass": all_pass,
        "decision": (
            "advance_to_covariant_coherence_trace_action_before_holdout"
            if all_pass
            else "retire_exact_v4c_baryon_seeded_coherence_trace"
        ),
        "claim_boundary": config["claim_boundary"],
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(serializable_fit(primary), indent=2, sort_keys=True))
    print(json.dumps(gates, indent=2, sort_keys=True))
    print(f"decision: {report['decision']}")


if __name__ == "__main__":
    main()
