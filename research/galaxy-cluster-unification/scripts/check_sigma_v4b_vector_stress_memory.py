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
from infer_sigma_v3c_spent_operator import sample_cluster
from run_p0715_sky_lensing_engine_validation import frozen_sky_field

from voidscreen.sigma_operator_inference import windowed_fourier
from voidscreen.sigma_variational_source import (
    helmholtz_memory,
    misalignment_potential_and_gradients,
)
from voidscreen.sigma_vector_stress import (
    projected_vector_stress_source,
    spectral_gradient,
    variational_source_from_potential,
    vector_chain_gradient,
    vector_stress,
)


def manufactured_checks() -> dict[str, float]:
    points = 25
    spacing = 0.4
    coordinate = np.arange(points) * spacing
    x, y = np.meshgrid(coordinate, coordinate)
    scalar = (
        0.8 * np.cos(2.0 * np.pi * x / (points * spacing))
        + 0.5 * np.sin(4.0 * np.pi * y / (points * spacing))
        + 0.25 * np.cos(2.0 * np.pi * (x + 2.0 * y) / (points * spacing))
    )
    memory_length = 1.2
    vector_scale = 0.9
    direction_scalar = np.roll(scalar, (3, -2), axis=(0, 1))
    direction_scalar /= np.sqrt(np.mean(np.square(direction_scalar)))
    density, source = variational_source_from_potential(
        scalar,
        spacing=spacing,
        memory_length=memory_length,
        vector_scale=vector_scale,
    )
    step = 2.0e-6
    plus = variational_source_from_potential(
        scalar + step * direction_scalar,
        spacing=spacing,
        memory_length=memory_length,
        vector_scale=vector_scale,
    )[0]
    minus = variational_source_from_potential(
        scalar - step * direction_scalar,
        spacing=spacing,
        memory_length=memory_length,
        vector_scale=vector_scale,
    )[0]
    finite_full = float(np.sum(plus - minus) * spacing**2 / (2.0 * step))
    analytic_full = float(np.sum(source * direction_scalar) * spacing**2)
    full_error = abs(finite_full - analytic_full) / max(
        abs(finite_full), abs(analytic_full), 1.0e-15
    )

    physical = spectral_gradient(scalar, spacing=spacing)
    direction_vector = spectral_gradient(
        np.roll(scalar, (2, -1), axis=(0, 1)), spacing=spacing
    )
    direction_vector /= np.sqrt(np.mean(np.square(direction_vector)))
    normalized, local = vector_stress(physical, vector_scale=vector_scale)
    memory = helmholtz_memory(local, spacing=spacing, length=memory_length)
    _, gradient_local, _ = misalignment_potential_and_gradients(local, memory)
    chain = vector_chain_gradient(
        normalized, gradient_local, vector_scale=vector_scale
    )
    plus_local = vector_stress(
        physical + step * direction_vector, vector_scale=vector_scale
    )[1]
    minus_local = vector_stress(
        physical - step * direction_vector, vector_scale=vector_scale
    )[1]
    finite_local = float(
        (
            np.sum(misalignment_potential_and_gradients(plus_local, memory)[0])
            - np.sum(misalignment_potential_and_gradients(minus_local, memory)[0])
        )
        / (2.0 * step)
    )
    analytic_local = float(np.sum(chain * direction_vector))
    local_error = abs(finite_local - analytic_local) / max(
        abs(finite_local), abs(analytic_local), 1.0e-15
    )
    source_rms = float(np.sqrt(np.mean(source**2)))
    return {
        "local_stress_directional_derivative_relative_error": local_error,
        "full_functional_directional_derivative_relative_error": full_error,
        "periodic_source_absolute_mean_over_RMS": abs(float(np.mean(source)))
        / source_rms,
        "manufactured_potential_minimum": float(np.min(density)),
        "manufactured_potential_maximum": float(np.max(density)),
    }


def add_physical_deflection(dataset: dict[str, object], config: dict) -> None:
    axis_kpc = np.asarray(dataset["axis_kpc"], dtype=float)
    east_kpc, north_kpc = np.meshgrid(axis_kpc, axis_kpc)
    east_arcsec = east_kpc / float(dataset["kpc_per_arcsec"])
    north_arcsec = north_kpc / float(dataset["kpc_per_arcsec"])
    field = frozen_sky_field(
        str(dataset["cluster"]),
        float(dataset["lens_redshift"]),
        str(config["sample"]["source_model"]),
    )
    alpha_east, alpha_north = field.alpha(
        east_arcsec, north_arcsec, float(config["sample"]["source_redshift"])
    )
    conversion = float(dataset["kpc_per_arcsec"])
    dataset["physical_deflection_east_kpc"] = np.asarray(alpha_east) * conversion
    dataset["physical_deflection_north_kpc"] = np.asarray(alpha_north) * conversion


def correction_templates(
    dataset: dict[str, object],
    *,
    memory_length: float,
    vector_scale: float,
    padding_factor: int,
) -> tuple[dict[str, np.ndarray], object]:
    axis = np.asarray(dataset["axis_kpc"], dtype=float)
    field = projected_vector_stress_source(
        dataset["physical_deflection_east_kpc"],
        dataset["physical_deflection_north_kpc"],
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen Sigma v4B map audit.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v4b_vector_stress_memory_audit.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v4b_vector_stress_memory_audit",
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
        full_rms = float(np.sqrt(np.mean(field.full_source**2)))
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
        "local_stress_directional_derivative": checks[
            "local_stress_directional_derivative_relative_error"
        ]
        <= gates_config["maximum_local_stress_directional_derivative_relative_error"],
        "full_functional_directional_derivative": checks[
            "full_functional_directional_derivative_relative_error"
        ]
        <= gates_config["maximum_full_functional_directional_derivative_relative_error"],
        "conserved_periodic_source": all(
            row["absolute_full_source_mean_over_RMS"]
            <= gates_config["maximum_absolute_source_mean_over_source_rms"]
            for row in source_statistics
        ),
        "signed_source_support": all(
            row["positive_cropped_source_pixel_fraction"]
            >= gates_config["minimum_positive_source_pixel_fraction"]
            and row["negative_cropped_source_pixel_fraction"]
            >= gates_config["minimum_negative_source_pixel_fraction"]
            for row in source_statistics
        ),
        "broad_correction_power": all(
            row["correction_power_fraction_wavelength_ge_50kpc"]
            >= gates_config["minimum_correction_power_fraction_at_wavelength_ge_50kpc"]
            for row in broad_power
        ),
        "positive_physical_action_sign": float(primary["unconstrained_eta_sigma_kpc2"]) > 0.0,
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
    pd.DataFrame(source_statistics).to_csv(args.output / "source_statistics.csv", index=False)
    pd.DataFrame(broad_power).to_csv(args.output / "broad_power_metrics.csv", index=False)

    eta = float(primary["eta_sigma_kpc2"])
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
        correction_limit = float(np.nanpercentile(np.abs(correction), 99.0))
        image = axes[row, 1].imshow(
            correction,
            origin="lower",
            extent=[axis[0], axis[-1], axis[0], axis[-1]],
            cmap="coolwarm",
            vmin=-correction_limit,
            vmax=correction_limit,
        )
        axes[row, 1].set_title("vector-stress correction")
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
        "Sigma v4B vector-stress memory\n"
        f"L={primary['L_sigma_kpc']:.3g} kpc, ell={primary['ell_sigma_kpc']:.3g} kpc, "
        f"eta={eta:.3g} kpc^2, joint NRMSE={primary['normalized_RMSE']:.3f}"
    )
    figure.savefig(args.output / "vector_stress_memory_audit.png", dpi=180)
    plt.close(figure)

    report = {
        "status": "completed Sigma v4B vector-stress memory audit",
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
        "source_statistics": source_statistics,
        "broad_power": broad_power,
        "gates": gates,
        "all_preregistered_gates_pass": all_pass,
        "decision": (
            "advance_to_covariant_vector_stress_completion_before_holdout"
            if all_pass
            else "retire_exact_v4b_vector_stress_memory_source"
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
