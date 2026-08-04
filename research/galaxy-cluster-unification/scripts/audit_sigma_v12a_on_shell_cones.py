from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v12a_general_covector import (
    GeneralCovectorBackground,
    boosted_unitary_background,
    lorentz_invariant_background_residuals,
    solve_tilted_constant_branch_scalar_clock,
    unitary_hessian_parity,
)
from voidscreen.sigma_v12a_on_shell_cones import (
    principal_cone_convergence,
    scan_common_time_on_shell_branch,
    screen_on_shell_parameter_pair,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit on-shell constant-background cones for Sigma v12A."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v12a_on_shell_cones.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v12a_on_shell_cones",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixed = config["fixed_values"]
    selected_k_b = float(fixed["selected_k_b"])
    selected_k_2 = float(fixed["selected_k_2"])
    growth_threshold = float(fixed["principal_growth_threshold"])
    polynomial_tolerance = float(fixed["polynomial_residual_tolerance"])

    parity_rows = [
        unitary_hessian_parity(
            scalar_clock_ratio=clock,
            aether_parallel=parallel,
            aether_perpendicular=perpendicular,
            orientation_strength=strength,
            wave_number_ratio=30.0,
            k_b=selected_k_b,
            k_2=selected_k_2,
        )
        for clock, parallel, perpendicular, strength in (
            (0.7, 0.3, 0.2, -1.0),
            (1.0, 0.0, 0.5, 1.0),
            (1.4, -0.4, 0.8, -1.0),
        )
    ]
    maximum_parity_residual = max(
        max(float(value) for value in row.values()) for row in parity_rows
    )

    invariant_clock = solve_tilted_constant_branch_scalar_clock(
        tilt_magnitude=0.5,
        k_b=selected_k_b,
        k_2=selected_k_2,
    )
    invariant_reference = GeneralCovectorBackground(
        scalar_covector=(invariant_clock, 0.0, 0.0, 0.0),
        aether_spatial_covector=(0.0, 0.0, 0.5),
        k_b=selected_k_b,
        k_2=selected_k_2,
    )
    invariant_boosted = boosted_unitary_background(
        scalar_clock_ratio=invariant_clock,
        aether_spatial_covector=(0.0, 0.0, 0.5),
        boost_velocity=(0.1, -0.05, 0.2),
        k_b=selected_k_b,
        k_2=selected_k_2,
    )
    invariant_residuals = lorentz_invariant_background_residuals(
        invariant_reference,
        invariant_boosted,
    )
    maximum_invariant_residual = max(invariant_residuals.values())

    parameter_screens = [
        screen_on_shell_parameter_pair(
            k_b=float(pair[0]),
            k_2=float(pair[1]),
            orientation_strength=float(
                fixed["parameter_screen_orientation_strength"]
            ),
            tilt_magnitudes=tuple(
                float(value)
                for value in fixed["parameter_screen_tilt_magnitudes"]
            ),
            relative_angles_degrees=tuple(
                float(value) for value in fixed["wave_angles_degrees"]
            ),
            wave_number_ratio=float(fixed["parameter_screen_wave_number"]),
            principal_growth_threshold=growth_threshold,
            metric_cone_frequency_tolerance=float(
                fixed["parameter_metric_cone_frequency_tolerance"]
            ),
            polynomial_residual_tolerance=polynomial_tolerance,
            aether_eom_tolerance=float(fixed["aether_eom_tolerance"]),
        )
        for pair in fixed["parameter_pairs"]
    ]
    passing_pairs = [
        [row["k_b"], row["k_2"]]
        for row in parameter_screens
        if row["all_parameter_screen_gates_pass"]
    ]

    common_time_scans = [
        scan_common_time_on_shell_branch(
            k_b=selected_k_b,
            k_2=selected_k_2,
            orientation_strength=float(strength),
            tilt_magnitude=float(tilt),
            boost_velocities=tuple(
                float(value) for value in fixed["boost_velocities"]
            ),
            wave_angles_degrees=tuple(
                float(value) for value in fixed["wave_angles_degrees"]
            ),
            wave_number_ratio=float(fixed["common_time_wave_number"]),
            principal_growth_threshold=growth_threshold,
            metric_cone_frequency_tolerance=float(
                fixed["common_time_metric_cone_frequency_tolerance"]
            ),
            polynomial_residual_tolerance=polynomial_tolerance,
        )
        for strength in fixed["common_time_orientation_strengths"]
        for tilt in fixed["common_time_tilt_magnitudes"]
    ]
    sign_survival = {
        str(float(strength)): all(
            row["common_time_found_at_declared_threshold"]
            for row in common_time_scans
            if row["orientation_strength"] == float(strength)
        )
        for strength in fixed["common_time_orientation_strengths"]
    }

    convergence_rows = [
        principal_cone_convergence(
            k_b=selected_k_b,
            k_2=selected_k_2,
            orientation_strength=-1.0,
            tilt_magnitude=float(sentinel["tilt_magnitude"]),
            boost_velocity=float(sentinel["boost_velocity"]),
            wave_angles_degrees=tuple(
                float(value) for value in fixed["wave_angles_degrees"]
            ),
            wave_numbers=tuple(
                float(value) for value in fixed["convergence_wave_numbers"]
            ),
            principal_growth_threshold=growth_threshold,
            metric_cone_frequency_tolerance=float(
                fixed["common_time_metric_cone_frequency_tolerance"]
            ),
        )
        for sentinel in fixed["convergence_sentinels"]
    ]
    original_k2_two_convergence = principal_cone_convergence(
        k_b=1.0,
        k_2=2.0,
        orientation_strength=-1.0,
        tilt_magnitude=0.5,
        boost_velocity=0.0,
        wave_angles_degrees=(90.0,),
        wave_numbers=(300.0, 1000.0, 3000.0),
        principal_growth_threshold=growth_threshold,
        metric_cone_frequency_tolerance=float(
            fixed["common_time_metric_cone_frequency_tolerance"]
        ),
    )
    maximum_selected_intercept = max(
        abs(float(row["frequency_excess_fit_intercept"]))
        for row in convergence_rows
    )
    maximum_selected_convergence_growth = max(
        float(sample["maximum_normalized_exponential_growth"])
        for row in convergence_rows
        for sample in row["rows"]
    )
    selected_convergence_root_structure = all(
        bool(sample["finite_constraint_root_structure_preserved"])
        for row in convergence_rows
        for sample in row["rows"]
    )

    gates = {
        "manifest_covariant_density_matches_unitary_adm_hessian": (
            maximum_parity_residual <= float(fixed["parity_tolerance"])
        ),
        "boosts_preserve_background_invariants": (
            maximum_invariant_residual
            <= float(fixed["lorentz_invariant_tolerance"])
        ),
        "original_k2_two_row_has_persistent_metric_cone_excess": (
            float(original_k2_two_convergence["frequency_excess_fit_intercept"])
            > 0.002
        ),
        "selected_pair_is_unique_frozen_grid_survivor": passing_pairs
        == [[selected_k_b, selected_k_2]],
        "negative_orientation_has_common_time_on_all_on_shell_tilts": (
            sign_survival.get("-1.0", False)
        ),
        "selected_principal_frequency_converges_to_metric_cone": (
            maximum_selected_intercept
            <= float(fixed["principal_cone_intercept_tolerance"])
        ),
        "selected_convergence_rows_remain_hyperbolic_at_threshold": (
            maximum_selected_convergence_growth <= growth_threshold
        ),
        "selected_convergence_root_structure_preserved": (
            selected_convergence_root_structure
        ),
    }
    report = {
        "status": "Sigma v12A on-shell constant-background common-cone screen",
        "protocol_status": config["protocol_status"],
        "selected_theory_side_row": {
            "k_b": selected_k_b,
            "k_2": selected_k_2,
            "orientation_strength": -1.0,
            "flat_scalar_speed_squared": (2.0 - selected_k_b)
            / (selected_k_2 * selected_k_b),
        },
        "maximum_unitary_parity_residual": maximum_parity_residual,
        "maximum_lorentz_invariant_residual": maximum_invariant_residual,
        "lorentz_invariant_residuals": invariant_residuals,
        "passing_parameter_pairs": passing_pairs,
        "parameter_screens": parameter_screens,
        "common_time_sign_survival": sign_survival,
        "common_time_scans": common_time_scans,
        "principal_convergence": convergence_rows,
        "maximum_selected_frequency_excess_intercept": maximum_selected_intercept,
        "maximum_selected_convergence_growth": maximum_selected_convergence_growth,
        "original_k2_two_convergence": original_k2_two_convergence,
        "gates": gates,
        "all_on_shell_constant_background_gates_pass": bool(all(gates.values())),
        "coordinate_time_energy_is_decisive_gate": False,
        "reduced_physical_hamiltonian_complete": False,
        "nonconstant_background_characteristics_complete": False,
        "theory_viable": False,
        "observational_data_accessed": False,
        "raw_holdout_opened": False,
        "decision": (
            "Reject the original K_B=1,K2=2 frozen row. Provisionally select the existing "
            "K_B=1,K2=4,lambda_D=-1 row for the next theory-only gate if every reported "
            "constant-background gate passes; do not open observations until the reduced "
            "physical Hamiltonian and nonconstant-background characteristics pass."
        ),
        "reason": (
            "The arbitrary off-shell 1.89c warning is not a fair on-shell verdict. After "
            "imposing the projected constant aether equation, K2=2 retains a smaller but "
            "invariant principal cone excess. K2=4 is the unique screened existing-constant "
            "row whose negative orientation has one common sampled time direction and whose "
            "finite-k cone excess extrapolates to zero. Coordinate-time energies remain "
            "negative on some moving backgrounds and require a constrained covariant energy "
            "calculation before this action can survive."
        ),
        "scope_limit": config["scope_limit"],
        "next_kill_gate": (
            "Construct the reduced physical Hamiltonian in the selected common time, then "
            "repeat the characteristic audit with nonzero scalar Hessian, aether gradient, "
            "extrinsic curvature, and local spacetime curvature."
        ),
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
