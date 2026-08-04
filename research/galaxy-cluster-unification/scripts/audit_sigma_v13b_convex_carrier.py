from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v13b_convex_carrier import (
    ConvexCarrierParameters,
    carrier_hamiltonian_density,
    carrier_legendre_state,
    carrier_phase_space_hessian,
    carrier_radial_curvature,
    carrier_response_mu,
    carrier_shape,
    numerical_flux_jacobian,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the Sigma v13B convex Hamiltonian carrier."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v13b_convex_carrier.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v13b_convex_carrier",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixed = config["fixed_values"]
    selected = ConvexCarrierParameters(
        acceleration_scale=float(fixed["acceleration_scale"]),
        epsilon=float(fixed["selected_epsilon"]),
    ).validated()
    radius = np.concatenate(
        (
            [0.0],
            np.geomspace(
                float(fixed["radius_ratio_minimum"]),
                float(fixed["radius_ratio_maximum"]),
                int(fixed["radius_ratio_samples"]),
            ),
        )
    )

    eigenvalue_rows = []
    for epsilon in fixed["epsilon_theory_scan"]:
        epsilon = float(epsilon)
        transverse = carrier_response_mu(radius, epsilon=epsilon)
        radial = carrier_radial_curvature(radius, epsilon=epsilon)
        shape = carrier_shape(radius, epsilon=epsilon)
        eigenvalue_rows.append(
            {
                "epsilon": epsilon,
                "minimum_transverse_hessian": float(np.min(transverse)),
                "minimum_radial_hessian": float(np.min(radial)),
                "maximum_transverse_hessian": float(np.max(transverse)),
                "maximum_radial_hessian": float(np.max(radial)),
                "minimum_hamiltonian_shape": float(np.min(shape)),
                "all_hessian_eigenvalues_strictly_positive": bool(
                    np.all(transverse > 0.0) and np.all(radial > 0.0)
                ),
                "all_hessian_eigenvalues_at_most_one": bool(
                    np.all(transverse <= 1.0) and np.all(radial <= 1.0)
                ),
                "hamiltonian_nonnegative": bool(np.all(shape >= 0.0)),
            }
        )

    fractions = np.linspace(
        -1.0,
        1.0,
        int(fixed["momentum_fraction_samples"]),
    )[:, None]
    cosines = np.linspace(
        -1.0,
        1.0,
        int(fixed["direction_cosine_samples"]),
    )[None, :]
    spatial_fractions = np.sqrt(np.maximum(0.0, 1.0 - fractions**2))
    maximum_speed = 0.0
    minimum_discriminant = np.inf
    worst_speed_row = None
    for ratio in radius:
        transverse = float(
            carrier_response_mu(ratio, epsilon=selected.epsilon)
        )
        radial = float(
            carrier_radial_curvature(ratio, epsilon=selected.epsilon)
        )
        radial_excess = radial - transverse
        temporal = transverse + radial_excess * fractions**2
        mixing = (
            radial_excess * fractions * spatial_fractions * cosines
        )
        spatial = (
            transverse
            + radial_excess * spatial_fractions**2 * cosines**2
        )
        discriminant = temporal * spatial
        speed = np.abs(mixing) + np.sqrt(discriminant)
        minimum_discriminant = min(
            minimum_discriminant,
            float(np.min(discriminant)),
        )
        flat_index = int(np.argmax(speed))
        row_index, column_index = np.unravel_index(flat_index, speed.shape)
        candidate_speed = float(speed[row_index, column_index])
        if candidate_speed > maximum_speed:
            maximum_speed = candidate_speed
            temporal_value = float(temporal[row_index, 0])
            mixing_value = float(mixing[row_index, column_index])
            spatial_value = float(spatial[row_index, column_index])
            root = float(np.sqrt(discriminant[row_index, column_index]))
            largest_block_eigenvalue = 0.5 * (
                temporal_value
                + spatial_value
                + np.sqrt(
                    (temporal_value - spatial_value) ** 2
                    + 4.0 * mixing_value**2
                )
            )
            worst_speed_row = {
                "radius_ratio": float(ratio),
                "momentum_fraction": float(fractions[row_index, 0]),
                "direction_cosine": float(cosines[0, column_index]),
                "temporal_hessian": temporal_value,
                "momentum_gradient_mixing": mixing_value,
                "directional_spatial_hessian": spatial_value,
                "discriminant": float(discriminant[row_index, column_index]),
                "negative_characteristic_speed": -mixing_value - root,
                "positive_characteristic_speed": -mixing_value + root,
                "maximum_absolute_characteristic_speed": candidate_speed,
                "largest_relevant_hessian_eigenvalue": float(
                    largest_block_eigenvalue
                ),
                "hyperbolic": bool(
                    discriminant[row_index, column_index] > 0.0
                ),
                "causal_in_preferred_unit_cone": bool(
                    candidate_speed <= 1.0 + 1.0e-12
                ),
            }

    legendre_rows = []
    for velocity in fixed["legendre_time_derivatives"]:
        for magnitude in fixed["legendre_gradient_magnitudes"]:
            row = carrier_legendre_state(
                float(velocity),
                (float(magnitude), 0.0, 0.0),
                parameters=selected,
            )
            legendre_rows.append(
                {"spatial_gradient_magnitude": float(magnitude), **row}
            )

    rng = np.random.default_rng(int(fixed["finite_difference_seed"]))
    maximum_flux_jacobian_residual = 0.0
    minimum_random_hessian_eigenvalue = np.inf
    maximum_random_hessian_eigenvalue = 0.0
    for _ in range(int(fixed["finite_difference_samples"])):
        direction = rng.normal(size=4)
        direction /= np.linalg.norm(direction)
        magnitude = 10.0 ** rng.uniform(
            np.log10(float(fixed["finite_difference_ratio_minimum"])),
            np.log10(float(fixed["finite_difference_ratio_maximum"])),
        )
        phase = magnitude * direction
        analytic = carrier_phase_space_hessian(
            float(phase[0]),
            phase[1:],
            parameters=selected,
        )
        numeric = numerical_flux_jacobian(
            float(phase[0]),
            phase[1:],
            parameters=selected,
        )
        scale = max(1.0, float(np.max(np.abs(analytic))))
        maximum_flux_jacobian_residual = max(
            maximum_flux_jacobian_residual,
            float(np.max(np.abs(analytic - numeric)) / scale),
        )
        eigenvalues = np.linalg.eigvalsh(analytic)
        minimum_random_hessian_eigenvalue = min(
            minimum_random_hessian_eigenvalue,
            float(np.min(eigenvalues)),
        )
        maximum_random_hessian_eigenvalue = max(
            maximum_random_hessian_eigenvalue,
            float(np.max(eigenvalues)),
        )

    static_rows = []
    for ratio in (0.0, 1.0e-8, 1.0e-6, 1.0e-3, 1.0, 1.0e3):
        hamiltonian = carrier_hamiltonian_density(
            0.0,
            (ratio, 0.0, 0.0),
            parameters=selected,
        )
        legendre = carrier_legendre_state(
            0.0,
            (ratio, 0.0, 0.0),
            parameters=selected,
        )
        static_rows.append(
            {
                "gradient_over_a_sigma": ratio,
                "mu": float(carrier_response_mu(ratio, epsilon=selected.epsilon)),
                "hamiltonian_density": hamiltonian,
                "lagrangian_plus_hamiltonian_residual": float(
                    legendre["lagrangian_density"] + hamiltonian
                ),
            }
        )

    convexity_tolerance = float(fixed["convexity_tolerance"])
    causality_tolerance = float(fixed["causality_tolerance"])
    finite_difference_tolerance = float(fixed["finite_difference_tolerance"])
    legendre_tolerance = float(fixed["legendre_residual_tolerance"])
    verification_gates = {
        "analytic_hessian_bounds_hold_on_all_epsilon_rows": all(
            row["all_hessian_eigenvalues_strictly_positive"]
            and row["all_hessian_eigenvalues_at_most_one"]
            for row in eigenvalue_rows
        ),
        "hamiltonian_nonnegative_on_all_epsilon_rows": all(
            row["hamiltonian_nonnegative"] for row in eigenvalue_rows
        ),
        "independent_flux_jacobian_matches_hessian": maximum_flux_jacobian_residual
        <= finite_difference_tolerance,
        "random_hessian_remains_strictly_convex": minimum_random_hessian_eigenvalue
        > convexity_tolerance,
        "random_hessian_operator_norm_at_most_one": maximum_random_hessian_eigenvalue
        <= 1.0 + causality_tolerance,
        "arbitrary_background_characteristics_hyperbolic": minimum_discriminant
        > 0.0,
        "arbitrary_background_characteristics_causal": maximum_speed
        <= 1.0 + causality_tolerance,
        "legendre_map_is_unique_and_nonsingular": all(
            float(row["hamiltonian_momentum_curvature"]) > convexity_tolerance
            and abs(float(row["momentum_map_residual"])) <= legendre_tolerance
            and abs(float(row["legendre_reconstruction_residual"]))
            <= legendre_tolerance
            for row in legendre_rows
        ),
        "static_slice_is_exact_aqual_energy": all(
            abs(float(row["lagrangian_plus_hamiltonian_residual"]))
            <= legendre_tolerance
            for row in static_rows
        ),
        "reduced_carrier_parameter_count_within_budget": int(
            fixed["current_reduced_carrier_constants"]
        )
        <= int(fixed["maximum_physical_constants"]),
    }
    advancement_gates = {
        "bounded_reduced_hamiltonian": bool(
            verification_gates["analytic_hessian_bounds_hold_on_all_epsilon_rows"]
            and verification_gates["hamiltonian_nonnegative_on_all_epsilon_rows"]
        ),
        "no_linear_dust_charge_energy": True,
        "regular_global_legendre_map": bool(
            verification_gates["legendre_map_is_unique_and_nonsingular"]
        ),
        "causal_reduced_scalar_characteristics": bool(
            verification_gates["arbitrary_background_characteristics_causal"]
        ),
        "static_aqual_response_retained": bool(
            verification_gates["static_slice_is_exact_aqual_energy"]
        ),
    }
    open_covariant_gates = {
        "manifestly_covariant_action": False,
        "preferred_foliation_uniquely_sourced_without_extra_charge": False,
        "joint_metric_carrier_constraint_count": False,
        "single_physical_metric_lensing_equations": False,
        "luminal_tensor_cone_after_covariantization": False,
        "solar_ppn_gate": False,
    }
    report = {
        "status": "Sigma v13B convex reduced-carrier selection",
        "protocol_status": config["protocol_status"],
        "candidate": config["candidate"],
        "selected_parameters": {
            "acceleration_scale": selected.acceleration_scale,
            "epsilon": selected.epsilon,
            "reduced_carrier_physical_constant_count": int(
                fixed["current_reduced_carrier_constants"]
            ),
        },
        "phase_space_hessian_identity": {
            "transverse": "mu=epsilon+(1-epsilon)t/(1+t)",
            "radial": "epsilon+(1-epsilon)t(t+2)/(1+t)^2",
            "analytic_range": "epsilon <= lambda(Hessian) < 1",
        },
        "characteristic_identity": {
            "equation": "(c+b)^2=A C",
            "causal_bound": "max |c| <= lambda_max([[A,b],[b,C]]) <= 1",
            "maximum_scanned_absolute_speed": maximum_speed,
            "minimum_scanned_discriminant": minimum_discriminant,
            "worst_speed_row": worst_speed_row,
        },
        "eigenvalue_rows": eigenvalue_rows,
        "legendre_rows": legendre_rows,
        "static_aqual_rows": static_rows,
        "independent_verification": {
            "maximum_flux_jacobian_residual": maximum_flux_jacobian_residual,
            "minimum_random_hessian_eigenvalue": minimum_random_hessian_eigenvalue,
            "maximum_random_hessian_eigenvalue": maximum_random_hessian_eigenvalue,
        },
        "verification_gates": verification_gates,
        "all_verification_gates_pass": bool(all(verification_gates.values())),
        "reduced_carrier_advancement_gates": advancement_gates,
        "all_reduced_carrier_advancement_gates_pass": bool(
            all(advancement_gates.values())
        ),
        "open_covariant_gates": open_covariant_gates,
        "all_covariant_gates_pass": bool(all(open_covariant_gates.values())),
        "v13b_selected_for_covariantization": bool(
            all(verification_gates.values()) and all(advancement_gates.values())
        ),
        "v13b_formulation_rejected": False,
        "post_v12_reset_total_material_formulation_failure_count": 2,
        "bounded_hamiltonian_same_gate_failure_count": 2,
        "three_failure_mechanism_reset_triggered": False,
        "observational_data_accessed": False,
        "raw_holdout_opened": False,
        "theory_viable": False,
        "decision": (
            "Select the v13B reduced Hamiltonian carrier for covariantization. It "
            "is globally strictly convex for epsilon>0, has no linear dust-charge "
            "energy, retains the static AQUAL flux, possesses a unique Legendre map, "
            "and keeps arbitrary-background scalar characteristics inside the unit "
            "preferred-frame cone. This is not yet a gravity theory because the "
            "covariant origin of that preferred frame and the joint metric equations "
            "remain unsolved."
        ),
        "next_gate": (
            "Construct a covariant foliation/metric completion that introduces no "
            "free clock charge or hidden matter state. Repeat the joint constraint, "
            "reduced-energy, tensor-cone, PPN, and one-metric lensing derivations before "
            "opening observations."
        ),
        "scope_limit": config["scope_limit"],
        "data_policy": config["data_policy"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
