from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_causal_polarization import (
    flrw_polarization_mode,
    signed_trace_bandpass,
    static_reduced_kinetic_hessian,
    static_reduced_velocity_lagrangian,
    transition_bandpass_y_derivative,
)


def centered_hessian(function, dimension: int, step: float = 1.0e-4) -> np.ndarray:
    origin = np.zeros(dimension)
    base = function(origin)
    result = np.empty((dimension, dimension))
    for row in range(dimension):
        row_step = np.zeros(dimension)
        row_step[row] = step
        result[row, row] = (
            function(origin + row_step) - 2.0 * base + function(origin - row_step)
        ) / step**2
        for column in range(row):
            column_step = np.zeros(dimension)
            column_step[column] = step
            result[row, column] = result[column, row] = (
                function(origin + row_step + column_step)
                - function(origin + row_step - column_step)
                - function(origin - row_step + column_step)
                + function(origin - row_step - column_step)
            ) / (4.0 * step**2)
    return result


def background_hessian(background: dict, q_sigma: float, weight: float) -> np.ndarray:
    return static_reduced_kinetic_hessian(
        np.asarray(background["weyl_spatial"], dtype=float),
        np.asarray(background["polarization_gradient"], dtype=float),
        np.asarray(background["trace_spatial"], dtype=float),
        float(background["sigma_background"]),
        float(background["anisotropy"]),
        q_sigma=q_sigma,
        polarization_weight=weight,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the cosmological mode and nonlinear kinetic degeneracy of Sigma v5B."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v5b_nonlinear_degeneracy_audit.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v5b_nonlinear_degeneracy_audit",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    normalization = config["normalization"]
    q_sigma = float(normalization["q_sigma"])
    weight = float(normalization["polarization_weight"])

    perturbative = config["perturbative_probe"]
    scale = np.geomspace(
        float(perturbative["lambda_min"]),
        float(perturbative["lambda_max"]),
        int(perturbative["points"]),
    )
    metric_trace_invariant = np.square(scale)
    source = signed_trace_bandpass(metric_trace_invariant)
    source_order = float(np.polyfit(np.log(scale), np.log(source), 1)[0])
    feedback_order = float(np.polyfit(np.log(scale), np.log(np.square(source)), 1)[0])

    flrw = config["flrw_scan"]
    hubble_ratio = np.geomspace(
        float(flrw["hubble_over_q_min"]),
        float(flrw["hubble_over_q_max"]),
        int(flrw["points"]),
    )
    flrw_records = []
    minimum_time = np.inf
    minimum_spatial = np.inf
    maximum_sound_speed = 0.0
    minimum_mass = np.inf
    for anisotropy in flrw["anisotropy_values"]:
        mode = flrw_polarization_mode(hubble_ratio, float(anisotropy))
        record = {
            "anisotropy": float(anisotropy),
            "minimum_time_kinetic": float(np.min(mode["time_kinetic"])),
            "minimum_spatial_gradient": float(np.min(mode["spatial_gradient"])),
            "minimum_sound_speed_squared": float(
                np.min(mode["sound_speed_squared"])
            ),
            "maximum_sound_speed_squared": float(
                np.max(mode["sound_speed_squared"])
            ),
            "minimum_mass_squared_times_L_squared": float(
                np.min(mode["mass_squared_times_L_squared"])
            ),
        }
        flrw_records.append(record)
        minimum_time = min(minimum_time, record["minimum_time_kinetic"])
        minimum_spatial = min(minimum_spatial, record["minimum_spatial_gradient"])
        maximum_sound_speed = max(
            maximum_sound_speed, record["maximum_sound_speed_squared"]
        )
        minimum_mass = min(
            minimum_mass, record["minimum_mass_squared_times_L_squared"]
        )

    hessian_records = {}
    for name, background in config["representative_backgrounds"].items():
        hessian = background_hessian(background, q_sigma, weight)
        eigenvalues = np.linalg.eigvalsh(hessian)
        hessian_records[name] = {
            "hessian": hessian.tolist(),
            "determinant": float(np.linalg.det(hessian)),
            "rank": int(np.linalg.matrix_rank(hessian, tol=1.0e-10)),
            "eigenvalues": eigenvalues.tolist(),
            "negative_eigenvalues": int(np.sum(eigenvalues < -1.0e-10)),
        }

    combined = config["representative_backgrounds"]["combined"]
    analytic = background_hessian(combined, q_sigma, weight)
    weyl = np.asarray(combined["weyl_spatial"], dtype=float)
    gradient = np.asarray(combined["polarization_gradient"], dtype=float)
    trace = np.asarray(combined["trace_spatial"], dtype=float)
    sigma_background = float(combined["sigma_background"])
    anisotropy = float(combined["anisotropy"])
    finite = centered_hessian(
        lambda velocity: static_reduced_velocity_lagrangian(
            velocity,
            weyl,
            gradient,
            trace,
            sigma_background,
            anisotropy,
            q_sigma=q_sigma,
            polarization_weight=weight,
        ),
        3,
    )
    finite_difference_error = float(np.max(np.abs(finite - analytic)))

    source_sign_records = []
    for invariant in config["source_sign_probe_Y"]:
        derivative = float(transition_bandpass_y_derivative(float(invariant)))
        lapse_curvature = -4.0 * 0.2 * derivative / q_sigma**2
        source_sign_records.append(
            {
                "Y": float(invariant),
                "J_Y": derivative,
                "normalized_lapse_hessian": lapse_curvature,
            }
        )

    random_scan = config["random_transport_scan"]
    rng = np.random.default_rng(int(random_scan["seed"]))
    count = int(random_scan["backgrounds"])
    weyl_scan = rng.normal(
        scale=float(random_scan["weyl_standard_deviation"]), size=(count, 3)
    )
    gradient_scan = rng.normal(
        scale=float(random_scan["gradient_standard_deviation"]), size=(count, 3)
    )
    determinants = np.empty(count)
    for index in range(count):
        hessian = static_reduced_kinetic_hessian(
            weyl_scan[index],
            gradient_scan[index],
            np.zeros(3),
            0.0,
            float(random_scan["anisotropy"]),
            q_sigma=q_sigma,
            polarization_weight=weight,
        )
        determinants[index] = np.linalg.det(hessian)
    determinant_threshold = float(random_scan["full_rank_determinant_threshold"])
    full_rank_fraction = float(np.mean(np.abs(determinants) > determinant_threshold))
    degenerate_fraction = 1.0 - full_rank_fraction

    gates_config = config["gates"]
    control = hessian_records["stegr_scalar_control"]
    source_only = hessian_records["source_only_low_field"]
    transport_only = hessian_records["transport_only"]
    combined_record = hessian_records["combined"]
    gates = {
        "quartic_cosmological_source_onset": abs(source_order - 4.0)
        <= float(gates_config["maximum_source_order_error"]),
        "eighth_order_integrated_metric_feedback": abs(feedback_order - 8.0)
        <= float(gates_config["maximum_feedback_order_error"]),
        "healthy_quadratic_flrw_scalar": minimum_time
        >= float(gates_config["minimum_flrw_time_kinetic"])
        and minimum_spatial
        >= float(gates_config["minimum_flrw_spatial_gradient"])
        and minimum_mass > 0.0
        and maximum_sound_speed
        <= float(gates_config["maximum_flrw_sound_speed_squared"]) + 1.0e-14,
        "analytic_hessian_matches_finite_difference": finite_difference_error
        <= float(gates_config["maximum_hessian_finite_difference_error"]),
        "stegr_control_retains_lapse_constraint": control["rank"]
        == int(gates_config["required_control_rank"])
        and abs(control["determinant"])
        <= float(gates_config["maximum_control_determinant"]),
        "source_channel_retains_required_degeneracy": source_only["rank"]
        <= int(gates_config["required_polarized_rank"]),
        "transport_channel_retains_required_degeneracy": transport_only["rank"]
        <= int(gates_config["required_polarized_rank"]),
        "combined_background_retains_required_degeneracy": combined_record["rank"]
        <= int(gates_config["required_polarized_rank"]),
        "generic_transport_degeneracy": degenerate_fraction
        >= float(gates_config["required_random_transport_degenerate_fraction"]),
    }
    gates = {name: bool(value) for name, value in gates.items()}
    nonlinear_gates = [
        "source_channel_retains_required_degeneracy",
        "transport_channel_retains_required_degeneracy",
        "combined_background_retains_required_degeneracy",
        "generic_transport_degeneracy",
    ]
    nonlinear_degeneracy_pass = bool(all(gates[name] for name in nonlinear_gates))
    report = {
        "status": "completed Sigma v5B theory-only nonlinear degeneracy audit",
        "observational_data_accessed": False,
        "raw_holdout_opened": False,
        "perturbative_cosmology": {
            "source_onset_order": source_order,
            "integrated_out_metric_feedback_order": feedback_order,
            "interpretation": "The sigma=0 FLRW branch is GR at linear order; J first appears at fourth metric order and its tree-level feedback at eighth order.",
        },
        "flrw_scalar_mode": {
            "records": flrw_records,
            "minimum_time_kinetic": minimum_time,
            "minimum_spatial_gradient": minimum_spatial,
            "maximum_sound_speed_squared": maximum_sound_speed,
            "minimum_mass_squared_times_L_squared": minimum_mass,
        },
        "static_kinetic_hessians": hessian_records,
        "hessian_finite_difference_maximum_absolute_error": finite_difference_error,
        "source_lapse_sign_probe": source_sign_records,
        "random_transport_scan": {
            "backgrounds": count,
            "full_rank_fraction": full_rank_fraction,
            "degenerate_fraction": degenerate_fraction,
            "minimum_absolute_determinant": float(np.min(np.abs(determinants))),
            "median_absolute_determinant": float(np.median(np.abs(determinants))),
            "maximum_absolute_determinant": float(np.max(np.abs(determinants))),
        },
        "gates": gates,
        "linear_cosmology_pass": bool(
            gates["quartic_cosmological_source_onset"]
            and gates["eighth_order_integrated_metric_feedback"]
            and gates["healthy_quadratic_flrw_scalar"]
        ),
        "nonlinear_degeneracy_pass": nonlinear_degeneracy_pass,
        "all_v5b_gates_pass": bool(all(gates.values())),
        "decision": "retire_exact_v5b_before_observational_fit",
        "reason": "Both the source and orientation transport make the lapse/connection velocity Hessian generically full rank on the polarized static backgrounds the mechanism needs.",
        "data_policy": config["data_policy"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
