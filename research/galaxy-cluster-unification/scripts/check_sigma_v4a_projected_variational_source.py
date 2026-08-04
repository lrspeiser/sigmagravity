from __future__ import annotations

import argparse
import hashlib
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

from infer_sigma_v3c_spent_operator import sample_cluster

from voidscreen.sigma_operator_inference import windowed_fourier
from voidscreen.sigma_variational_source import (
    helmholtz_memory,
    misalignment_potential_and_gradients,
    projected_variational_source,
    spectral_stf_hessian,
    variational_source_from_potential,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manufactured_checks() -> dict[str, float]:
    points = 24
    spacing = 0.3
    coordinate = np.arange(points) * spacing
    x, y = np.meshgrid(coordinate, coordinate)
    scalar = (
        0.7 * np.cos(2.0 * np.pi * x / (points * spacing))
        + 0.4 * np.sin(4.0 * np.pi * y / (points * spacing))
        + 0.3 * np.cos(2.0 * np.pi * (x + 2.0 * y) / (points * spacing))
    )
    direction_scalar = np.roll(scalar, (2, -3), axis=(0, 1))
    direction_scalar /= np.sqrt(np.mean(np.square(direction_scalar)))
    memory_length = 0.9
    tensor_scale = 1.7
    density, source = variational_source_from_potential(
        scalar,
        spacing=spacing,
        memory_length=memory_length,
        tensor_scale=tensor_scale,
    )
    step = 2.0e-6
    plus = variational_source_from_potential(
        scalar + step * direction_scalar,
        spacing=spacing,
        memory_length=memory_length,
        tensor_scale=tensor_scale,
    )[0]
    minus = variational_source_from_potential(
        scalar - step * direction_scalar,
        spacing=spacing,
        memory_length=memory_length,
        tensor_scale=tensor_scale,
    )[0]
    finite_full = float(np.sum(plus - minus) * spacing**2 / (2.0 * step))
    analytic_full = float(np.sum(source * direction_scalar) * spacing**2)
    full_error = abs(finite_full - analytic_full) / max(
        abs(finite_full), abs(analytic_full), 1.0e-15
    )

    local = spectral_stf_hessian(scalar, spacing=spacing)
    memory = helmholtz_memory(local, spacing=spacing, length=memory_length)
    direction_tensor = spectral_stf_hessian(
        np.roll(scalar, 3, axis=0), spacing=spacing
    )
    direction_tensor /= np.sqrt(np.mean(np.square(direction_tensor)))
    _, gradient_local, _ = misalignment_potential_and_gradients(local, memory)
    plus_local = misalignment_potential_and_gradients(
        local + step * direction_tensor, memory
    )[0]
    minus_local = misalignment_potential_and_gradients(
        local - step * direction_tensor, memory
    )[0]
    finite_local = float(np.sum(plus_local - minus_local) / (2.0 * step))
    analytic_local = float(np.sum(gradient_local * direction_tensor))
    local_error = abs(finite_local - analytic_local) / max(
        abs(finite_local), abs(analytic_local), 1.0e-15
    )
    source_rms = float(np.sqrt(np.mean(np.square(source))))
    return {
        "full_functional_directional_derivative_relative_error": full_error,
        "scalar_potential_directional_derivative_relative_error": local_error,
        "periodic_source_absolute_mean_over_rms": abs(float(np.mean(source)))
        / source_rms,
        "manufactured_potential_minimum": float(np.min(density)),
        "manufactured_potential_maximum": float(np.max(density)),
    }


def correction_templates(
    dataset: dict[str, object],
    *,
    memory_length: float,
    tensor_scale: float,
    padding_factor: int,
) -> tuple[dict[str, np.ndarray], object]:
    invariant = dataset["invariants"]["AQUAL"]
    axis = np.asarray(dataset["axis_kpc"], dtype=float)
    spacing = float(axis[1] - axis[0])
    field = projected_variational_source(
        invariant.shear_1,
        invariant.shear_2,
        spacing=spacing,
        memory_length=memory_length,
        tensor_scale=tensor_scale,
        padding_factor=padding_factor,
    )
    window = dataset["window"]
    transforms = {
        "convergence": windowed_fourier(field.unit_eta_kappa, window),
        "shear_1": windowed_fourier(field.unit_eta_shear_1, window),
        "shear_2": windowed_fourier(field.unit_eta_shear_2, window),
    }
    return transforms, field


def analytic_eta(
    datasets: list[dict[str, object]], templates: list[dict[str, np.ndarray]]
) -> tuple[float, float]:
    numerator = 0.0
    denominator = 0.0
    for dataset, template_channels in zip(datasets, templates, strict=True):
        band = dataset["band"]
        for channel, template in template_channels.items():
            baseline = dataset["transforms"]["AQUAL"][channel]
            target = dataset["transforms"]["halo"][channel]
            target_power = float(np.sum(np.abs(target[band]) ** 2))
            residual = target[band] - baseline[band]
            numerator += float(
                np.real(np.sum(np.conj(template[band]) * residual)) / target_power
            )
            denominator += float(np.sum(np.abs(template[band]) ** 2) / target_power)
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise RuntimeError("variational template has no finite scored power")
    unconstrained = numerator / denominator
    return float(unconstrained), float(max(0.0, unconstrained))


def score_prediction(
    datasets: list[dict[str, object]],
    templates: list[dict[str, np.ndarray]],
    eta: float,
) -> tuple[float, list[dict[str, float | str]], list[dict[str, float | str]]]:
    per_channel: list[dict[str, float | str]] = []
    per_cluster: list[dict[str, float | str]] = []
    cluster_squared = []
    for dataset, template_channels in zip(datasets, templates, strict=True):
        band = dataset["band"]
        channel_squared = []
        baseline_squared = []
        for channel, template in template_channels.items():
            baseline = dataset["transforms"]["AQUAL"][channel]
            target = dataset["transforms"]["halo"][channel]
            target_power = float(np.sum(np.abs(target[band]) ** 2))
            baseline_ratio = float(
                np.sum(np.abs(baseline[band] - target[band]) ** 2) / target_power
            )
            prediction_ratio = float(
                np.sum(np.abs(baseline[band] + eta * template[band] - target[band]) ** 2)
                / target_power
            )
            baseline_squared.append(baseline_ratio)
            channel_squared.append(prediction_ratio)
            per_channel.append(
                {
                    "cluster": str(dataset["cluster"]),
                    "channel": channel,
                    "AQUAL_baseline_normalized_RMSE": math.sqrt(baseline_ratio),
                    "prediction_normalized_RMSE": math.sqrt(prediction_ratio),
                    "improved": bool(prediction_ratio < baseline_ratio),
                }
            )
        baseline_cluster = math.sqrt(float(np.mean(baseline_squared)))
        prediction_cluster = math.sqrt(float(np.mean(channel_squared)))
        cluster_squared.append(float(np.mean(channel_squared)))
        per_cluster.append(
            {
                "cluster": str(dataset["cluster"]),
                "AQUAL_baseline_normalized_RMSE": baseline_cluster,
                "prediction_normalized_RMSE": prediction_cluster,
                "fraction_of_AQUAL_baseline": prediction_cluster / baseline_cluster,
            }
        )
    return math.sqrt(float(np.mean(cluster_squared))), per_cluster, per_channel


def evaluate_parameters(
    datasets: list[dict[str, object]],
    *,
    memory_length: float,
    tensor_scale: float,
    padding_factor: int,
    eta_override: float | None = None,
) -> dict[str, object]:
    generated = [
        correction_templates(
            dataset,
            memory_length=memory_length,
            tensor_scale=tensor_scale,
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
        "tau_sigma_projected": float(tensor_scale),
        "eta_sigma_kpc2": eta,
        "unconstrained_eta_sigma_kpc2": unconstrained,
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
    scale_bounds = [
        math.log(float(value)) for value in numerics["tau_sigma_projected_bounds"]
    ]

    def objective(parameters: np.ndarray) -> float:
        evaluated = evaluate_parameters(
            datasets,
            memory_length=math.exp(float(parameters[0])),
            tensor_scale=math.exp(float(parameters[1])),
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
        tensor_scale=math.exp(float(fit.x[1])),
        padding_factor=padding_factor,
    )
    evaluated["optimizer_success"] = bool(fit.success)
    evaluated["optimizer_message"] = str(fit.message)
    evaluated["function_evaluations"] = int(fit.nfev)
    return evaluated


def serializable_fit(fit: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in fit.items() if key not in {"templates", "fields"}}


def logarithmic_interior(value: float, bounds: list[float], margin: float = 0.01) -> bool:
    low, high = map(math.log, map(float, bounds))
    fraction = (math.log(float(value)) - low) / (high - low)
    return bool(margin < fraction < 1.0 - margin)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen Sigma v4A source audit.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v4a_projected_variational_source_audit.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v4a_projected_variational_source_audit",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    parent = ROOT / config["parent_config"]["path"]
    if sha256(parent) != config["parent_config"]["sha256"]:
        raise RuntimeError("parent v3C config hash does not match the frozen protocol")
    parent_config = json.loads(parent.read_text(encoding="utf-8"))
    datasets = [sample_cluster(name, parent_config) for name in config["sample"]["clusters"]]
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
            tensor_scale=float(fitted["tau_sigma_projected"]),
            padding_factor=primary_padding,
            eta_override=float(fitted["eta_sigma_kpc2"]),
        )
        cross_transfer.append(
            {
                "trained_on": str(training["cluster"]),
                "tested_on": str(target["cluster"]),
                "normalized_RMSE": float(transferred["normalized_RMSE"]),
            }
        )

    source_statistics = []
    for dataset, field in zip(datasets, primary["fields"], strict=True):
        full_rms = float(np.sqrt(np.mean(np.square(field.full_source))))
        source_statistics.append(
            {
                "cluster": str(dataset["cluster"]),
                "absolute_full_source_mean_over_RMS": abs(float(np.mean(field.full_source)))
                / full_rms,
                "positive_cropped_source_pixel_fraction": float(np.mean(field.source > 0.0)),
                "negative_cropped_source_pixel_fraction": float(np.mean(field.source < 0.0)),
                "cropped_source_RMS_per_kpc2": float(np.sqrt(np.mean(field.source**2))),
                "potential_maximum": float(np.max(field.potential)),
            }
        )

    gates_config = config["preregistered_gates"]
    source_mean_pass = all(
        row["absolute_full_source_mean_over_RMS"]
        <= gates_config["maximum_absolute_source_mean_over_source_rms"]
        for row in source_statistics
    )
    source_signs_pass = all(
        row["positive_cropped_source_pixel_fraction"]
        >= gates_config["minimum_positive_source_pixel_fraction"]
        and row["negative_cropped_source_pixel_fraction"]
        >= gates_config["minimum_negative_source_pixel_fraction"]
        for row in source_statistics
    )
    parameter_interior = logarithmic_interior(
        float(primary["L_sigma_kpc"]), config["numerics"]["L_sigma_kpc_bounds"]
    ) and logarithmic_interior(
        float(primary["tau_sigma_projected"]),
        config["numerics"]["tau_sigma_projected_bounds"],
    )
    every_cluster_improves = all(
        row["fraction_of_AQUAL_baseline"]
        <= gates_config["maximum_each_cluster_fraction_of_AQUAL_baseline_RMSE"]
        for row in primary["per_cluster"]
    )
    every_channel_improves = all(bool(row["improved"]) for row in primary["per_channel"])
    padding_fractional_change = abs(
        float(sensitivity["normalized_RMSE"]) - float(primary["normalized_RMSE"])
    ) / float(primary["normalized_RMSE"])
    gates = {
        "full_functional_directional_derivative": checks[
            "full_functional_directional_derivative_relative_error"
        ]
        <= gates_config["maximum_full_functional_directional_derivative_relative_error"],
        "scalar_potential_directional_derivative": checks[
            "scalar_potential_directional_derivative_relative_error"
        ]
        <= gates_config["maximum_scalar_potential_directional_derivative_relative_error"],
        "conserved_periodic_source": source_mean_pass,
        "signed_source_support": source_signs_pass,
        "positive_physical_action_sign": float(primary["unconstrained_eta_sigma_kpc2"]) > 0.0,
        "parameters_interior": parameter_interior,
        "joint_map_accuracy": float(primary["normalized_RMSE"])
        <= gates_config["maximum_joint_normalized_RMSE"],
        "each_cluster_twenty_percent_improvement": every_cluster_improves,
        "every_cluster_channel_improves": every_channel_improves,
        "cross_cluster_transfer": all(
            float(row["normalized_RMSE"])
            <= gates_config["maximum_each_cross_cluster_transfer_normalized_RMSE"]
            for row in cross_transfer
        ),
        "padding_stability": padding_fractional_change
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
    pd.DataFrame(source_statistics).to_csv(args.output / "source_statistics.csv", index=False)

    figure, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    eta = float(primary["eta_sigma_kpc2"])
    for row, (dataset, field) in enumerate(zip(datasets, primary["fields"], strict=True)):
        axis = dataset["axis_kpc"]
        target = dataset["invariants"]["halo"].convergence
        baseline = dataset["invariants"]["AQUAL"].convergence
        correction = eta * field.unit_eta_kappa
        prediction = baseline + correction
        maximum = float(np.nanpercentile(np.abs(target), 99.0))
        image = axes[row, 0].imshow(
            target,
            origin="lower",
            extent=[axis[0], axis[-1], axis[0], axis[-1]],
            cmap="viridis",
            vmin=-maximum,
            vmax=maximum,
        )
        axes[row, 0].set_title(f"{dataset['cluster']} halo target kappa")
        figure.colorbar(image, ax=axes[row, 0], shrink=0.75)
        correction_limit = float(np.nanpercentile(np.abs(correction), 99.0))
        image = axes[row, 1].imshow(
            correction,
            origin="lower",
            extent=[axis[0], axis[-1], axis[0], axis[-1]],
            cmap="coolwarm",
            vmin=-correction_limit,
            vmax=correction_limit,
        )
        axes[row, 1].set_title("signed variational correction")
        figure.colorbar(image, ax=axes[row, 1], shrink=0.75)
        image = axes[row, 2].imshow(
            prediction,
            origin="lower",
            extent=[axis[0], axis[-1], axis[0], axis[-1]],
            cmap="viridis",
            vmin=-maximum,
            vmax=maximum,
        )
        contour_maximum = float(np.nanpercentile(target, 99.0))
        axes[row, 2].contour(
            axis,
            axis,
            target,
            levels=np.linspace(0.2 * contour_maximum, contour_maximum, 5),
            colors="white",
            linewidths=0.7,
        )
        axes[row, 2].set_title("prediction color, target contours")
        figure.colorbar(image, ax=axes[row, 2], shrink=0.75)
        for column in range(3):
            axes[row, column].set(xlabel="east kpc", ylabel="north kpc")
    figure.suptitle(
        "Sigma v4A projected variational source\n"
        f"L={primary['L_sigma_kpc']:.3g} kpc, tau={primary['tau_sigma_projected']:.3g}, "
        f"eta={eta:.3g} kpc^2, joint NRMSE={primary['normalized_RMSE']:.3f}"
    )
    figure.savefig(args.output / "projected_variational_source_audit.png", dpi=180)
    plt.close(figure)

    report = {
        "status": "completed Sigma v4A projected variational source audit",
        "sample_is_spent": True,
        "raw_holdout_opened": False,
        "config_sha256": sha256(args.config),
        "parent_config_sha256": sha256(parent),
        "manufactured_checks": checks,
        "primary_shared_fit": serializable_fit(primary),
        "padding_sensitivity_shared_fit": serializable_fit(sensitivity),
        "padding_sensitivity_fractional_change_in_joint_RMSE": padding_fractional_change,
        "independent_training_fits": independent_fits,
        "cross_transfer": cross_transfer,
        "source_statistics": source_statistics,
        "gates": gates,
        "all_preregistered_gates_pass": all_pass,
        "decision": (
            "advance_to_covariant_3d_completion_before_holdout"
            if all_pass
            else "retire_exact_v4a_projected_variational_source"
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
