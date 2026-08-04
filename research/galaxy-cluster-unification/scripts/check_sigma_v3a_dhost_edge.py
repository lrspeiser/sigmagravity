from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_dhost_edge import (
    dhost_degeneracy_residuals,
    luminal_beyond_horndeski_coefficients,
    maximum_smooth_power_law_weyl_fraction,
    smooth_power_law_weyl_fraction,
    spherical_accelerations,
    spherical_xi_coefficients,
    uniform_density_acceleration_ratios,
    weyl_edge_correction_from_density_gradient,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_row(frame: pd.DataFrame, model: str) -> pd.Series:
    rows = frame[(frame.cluster == "ALL") & (frame.model == model)]
    if len(rows) != 1:
        raise RuntimeError(f"expected one ALL row for {model}, found {len(rows)}")
    return rows.iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the Sigma v3A local DHOST edge lane.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v3a_dhost_edge_audit.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v3a_dhost_edge_audit",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    gates = config["gates"]
    alpha_supremum = float(config["predeclared_domain"]["audit_alpha_H_maximum"])

    x = -np.geomspace(1e-6, 1e6, 10000)
    coefficients = luminal_beyond_horndeski_coefficients(
        x, x_background=-1.0, alpha_h=0.2, f0=1.0
    )
    residuals = dhost_degeneracy_residuals(coefficients, x)
    degeneracy_absolute_error = float(
        max(np.max(np.abs(value)) for value in residuals.values())
    )
    degeneracy_relative_error = float(
        max(
            np.max(np.abs(residuals["A1"])),
            np.max(np.abs(residuals["A2"])),
            np.max(
                np.abs(residuals["A4"])
                / np.maximum(np.abs(coefficients["A4"]), np.finfo(float).tiny)
            ),
            np.max(
                np.abs(residuals["A5"])
                / np.maximum(
                    (
                        np.abs(4.0 * coefficients["F_X"])
                        + np.abs(x * coefficients["A3"])
                    )
                    * np.abs(coefficients["A3"])
                    / (2.0 * coefficients["F"]),
                    np.finfo(float).tiny,
                )
            ),
        )
    )
    tensor_speed_squared = coefficients["F"] / (
        coefficients["F"] - x * coefficients["A1"]
    )
    tensor_speed_error = float(np.max(np.abs(np.sqrt(tensor_speed_squared) - 1.0)))

    xi = spherical_xi_coefficients(alpha_supremum)
    alpha_inside = np.nextafter(alpha_supremum, 0.0)
    uniform = uniform_density_acceleration_ratios(alpha_inside)
    uniform_positive = bool(uniform["matter_psi_over_newtonian"] > 0.0)

    slope_grid = np.linspace(
        float(config["predeclared_domain"]["smooth_density_power_law_slope_min"]),
        float(config["predeclared_domain"]["smooth_density_power_law_slope_max"]),
        10001,
    )
    slope_response = np.asarray(
        [smooth_power_law_weyl_fraction(value, alpha_supremum) for value in slope_grid]
    )
    maximizing_index = int(np.argmax(slope_response))
    numerical_maximum = float(slope_response[maximizing_index])
    analytic_maximum = maximum_smooth_power_law_weyl_fraction(alpha_supremum)
    maximum_identity_error = abs(numerical_maximum - analytic_maximum)

    # Verify the published spherical laws reduce to a pure density-gradient
    # Weyl term for a smooth power-law density profile.
    radius = np.geomspace(0.1, 10.0, 2000)
    density_slope = 1.4
    density = np.power(radius, -density_slope)
    density_gradient = -density_slope * density / radius
    mass = 4.0 * np.pi * np.power(radius, 3.0 - density_slope) / (3.0 - density_slope)
    mass_prime = 4.0 * np.pi * np.square(radius) * density
    mass_second = 4.0 * np.pi * (2.0 - density_slope) * radius * density
    acceleration = spherical_accelerations(
        radius,
        mass,
        mass_prime,
        mass_second,
        alpha_h=0.2,
    )
    edge_direct = weyl_edge_correction_from_density_gradient(
        radius, density_gradient, alpha_h=0.2
    )
    edge_from_potentials = acceleration["photon_weyl"] - acceleration["newtonian"]
    edge_identity_error = float(np.max(np.abs(edge_direct - edge_from_potentials)))

    input_config = config["spent_diagnostic_input"]
    summary_path = ROOT / input_config["structure_summary"]
    report_path = ROOT / input_config["cluster_report"]
    structure = pd.read_csv(summary_path)
    base = selected_row(structure, input_config["base_model"])
    optimistic = selected_row(structure, input_config["optimistic_existing_model"])
    newtonian = selected_row(structure, input_config["newtonian_model"])
    halo_kappa = float(base.median_halo_convergence)
    newtonian_kappa = float(newtonian.median_model_convergence)
    base_gap = float(base.median_kappa_needed)
    optimistic_gap = float(optimistic.median_kappa_needed)

    source_scaled_added_kappa = analytic_maximum * newtonian_kappa
    source_scaled_base_gap_closure = source_scaled_added_kappa / base_gap
    # Deliberately generous: pretend the fractional DHOST bound multiplies the
    # full halo convergence rather than its actual Newtonian baryonic source.
    halo_scaled_added_kappa = analytic_maximum * halo_kappa
    generous_base_gap_closure = halo_scaled_added_kappa / base_gap
    generous_optimistic_gap_closure = halo_scaled_added_kappa / optimistic_gap
    generous_best_gap_closure = max(generous_base_gap_closure, generous_optimistic_gap_closure)

    structural_pass = bool(
        degeneracy_relative_error <= gates["degeneracy_identity_max_relative_error"]
        and tensor_speed_error <= gates["tensor_speed_absolute_error_max"]
        and uniform_positive is bool(gates["positive_uniform_core_response_required"])
        and edge_identity_error <= gates["weak_edge_identity_max_absolute_error"]
        and maximum_identity_error <= 1e-10
    )
    amplitude_pass = bool(
        generous_best_gap_closure
        >= gates["minimum_fraction_of_spent_convergence_gap_closed_to_justify_2d_solver"]
    )
    advances = bool(structural_pass and amplitude_pass)

    args.output.mkdir(parents=True, exist_ok=True)
    slope_frame = pd.DataFrame(
        {
            "density_log_slope_n": slope_grid,
            "maximum_alpha_fractional_weyl_correction": slope_response,
        }
    )
    slope_frame.to_csv(args.output / "smooth_power_law_response.csv", index=False)
    report = {
        "status": "completed Sigma v3A local DHOST edge audit",
        "model_id": config["model_id"],
        "input_hashes": {
            "config": sha256(args.config),
            "structure_summary": sha256(summary_path),
            "cluster_report": sha256(report_path),
        },
        "action_structure": {
            "alpha_H_test_value_for_identities": 0.2,
            "degeneracy_max_absolute_error": degeneracy_absolute_error,
            "degeneracy_max_relative_error": degeneracy_relative_error,
            "tensor_speed_absolute_error": tensor_speed_error,
            "tensor_kinetic_F_positive": bool(np.all(coefficients["F"] > 0.0)),
            "matter_minimally_coupled_to_one_metric": True,
            "quadratic_DHOST_degenerate": degeneracy_relative_error
            <= gates["degeneracy_identity_max_relative_error"],
            "full_scalar_background_and_kinetic_health_proved": False,
        },
        "spherical_weak_field": {
            "Xi_at_alpha_supremum": xi,
            "uniform_core_ratios_just_below_alpha_supremum": uniform,
            "uniform_core_matter_response_positive": uniform_positive,
            "edge_identity_max_absolute_error": edge_identity_error,
            "smooth_power_law_maximizing_slope": float(slope_grid[maximizing_index]),
            "maximum_fractional_weyl_correction": analytic_maximum,
            "numerical_vs_analytic_maximum_error": maximum_identity_error,
            "exterior_vacuum_correction": 0.0,
        },
        "spent_amplitude_feasibility": {
            "interpretation": "no-fit smooth spherical proxy; not raw validation",
            "median_newtonian_baryon_convergence": newtonian_kappa,
            "median_sigma_v1_AQUAL_convergence": float(base.median_model_convergence),
            "median_optimistic_existing_convergence": float(
                optimistic.median_model_convergence
            ),
            "median_halo_convergence": halo_kappa,
            "median_sigma_v1_gap": base_gap,
            "median_optimistic_existing_gap": optimistic_gap,
            "source_scaled_maximum_added_convergence": source_scaled_added_kappa,
            "source_scaled_sigma_v1_gap_closure_fraction": source_scaled_base_gap_closure,
            "ultra_generous_halo_scaled_added_convergence": halo_scaled_added_kappa,
            "ultra_generous_sigma_v1_gap_closure_fraction": generous_base_gap_closure,
            "ultra_generous_optimistic_gap_closure_fraction": generous_optimistic_gap_closure,
            "best_ultra_generous_gap_closure_fraction": generous_best_gap_closure,
            "required_gap_closure_fraction": gates[
                "minimum_fraction_of_spent_convergence_gap_closed_to_justify_2d_solver"
            ],
        },
        "gate_results": {
            "degeneracy_tensor_speed_and_local_identities": structural_pass,
            "smooth_amplitude_feasibility": amplitude_pass,
            "full_scalar_health": False,
        },
        "advances_to_full_2d_solver": advances,
        "decision": (
            "advance to scalar-health derivation and a 2D solver"
            if advances
            else "retire the smooth local beta_1=0 DHOST edge term as the sole broad cluster response; its most generous amplitude bound cannot close the preregistered gap"
        ),
        "next_mechanism": (
            "derive a causal baryon-forced nonlocal tidal response that spreads edge information over a universal geometric scale while retaining no free homogeneous halo state"
        ),
        "claim_boundary": config["claim_boundary"],
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["gate_results"], indent=2, sort_keys=True))
    print(json.dumps(report["spent_amplitude_feasibility"], indent=2, sort_keys=True))
    print(report["decision"])


if __name__ == "__main__":
    main()
